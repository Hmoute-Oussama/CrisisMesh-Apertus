"""The CrisisMesh pipeline.

    REPORT -> INGEST -> SEGMENT -> APERTUS EXTRACT -> STRUCTURED EVENT
           -> DEDUP -> CORROBORATE -> CONTRADICTION -> CONFIDENCE
           -> EVIDENCE -> GRAPH

Each stage is a separate function so it can be tested, timed and audited in
isolation. Nothing here resolves a contradiction, and nothing here lets a claim
be written without an EvidenceLink.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from ..apertus import ApertusClient, ApertusUnavailable
from ..config import get_settings
from ..db import (
    AuditLog,
    Conflict,
    DuplicateGroup,
    Event,
    EvidenceLink,
    Report,
    session_scope,
    utcnow,
)
from ..grammar import GRAMMAR_VERSION
from ..guards import (
    REPORT_CLOSE,
    REPORT_OPEN,
    detect_injection,
    model_view,
    normalize_extraction,
    segment,
    uncertainty_notes,
)
from ..taxonomy import VerificationState, severity_from_markers
from . import contradiction as contradiction_svc
from . import dedup
from . import entities


# ---- ingestion ---------------------------------------------------------------

class ReportAlreadyProcessed(Exception):
    """Raised when an already-processed report is submitted again.

    Reprocessing would create a second copy of every event it produced, which
    would silently inflate corroboration counts. Making this an explicit error
    keeps the evidence counts honest.
    """


def ingest_report(
    session,
    raw_text: str,
    *,
    report_id: str | None = None,
    source_language: str = "und",
    source_type: str = "human_report",
    location_hint: str | None = None,
    reporter_id: str | None = None,
    reported_at: datetime | None = None,
    metadata: dict | None = None,
) -> Report:
    """Create an immutable report. raw_text is never mutated after this.

    report_id may be supplied so that a seeded corpus keeps stable identifiers
    across runs; this is what lets an evaluation join predictions back to
    ground-truth annotations. Omit it in normal operator use.
    """
    if not (raw_text or "").strip():
        raise ValueError("raw_text must not be empty")

    if report_id:
        existing = session.get(Report, report_id)
        if existing is not None:
            return existing

    text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    signals = detect_injection(raw_text)
    report = Report(
        report_id=report_id,
        raw_text=raw_text,
        source_language=source_language,
        source_type=source_type,
        location_hint=location_hint,
        reporter_id=reporter_id,
        reported_at=reported_at,
        metadata_=metadata or {},
        text_hash=text_hash,
        suspected_injection=bool(signals),
        injection_signals=signals,
        segment_count=len(segment(model_view(raw_text))),
    )
    session.add(report)
    session.flush()
    _audit(session, "report", report.report_id, "ingested",
           detail={"signals": len(signals), "lang": source_language})
    return report


def _audit(session, entity_type, entity_id, action, *, model_used=None,
           model_version=None, prompt_version=None, input_ref=None,
           output_summary=None, actor=None, detail=None) -> AuditLog:
    """Append one audit row.

    model_used is the identifier that actually served the request; model_version
    is the human-readable label. They are stored separately because an operator
    auditing a claim needs to know "which checkpoint answered", not just "Apertus
    answered" after a model swap.
    """
    entry = AuditLog(
        entity_type=entity_type, entity_id=entity_id, action=action,
        model_used=model_used,
        model_version=model_version if model_version is not None else model_used,
        prompt_version=prompt_version, input_ref=input_ref,
        output_summary=output_summary or {}, actor=actor, detail=detail or {},
    )
    session.add(entry)
    return entry


# ---- extraction --------------------------------------------------------------

def process_report(session, report: Report,
                   client: ApertusClient | None = None) -> dict:
    """Run the full pipeline for one report.

    Raises ApertusUnavailable rather than degrading. An operator who cannot tell
    which model produced a claim cannot audit the claim, so a missing model is a
    visible error, not a silent substitution.
    """
    st = get_settings()
    client = client or ApertusClient(st)

    if report.processed:
        raise ReportAlreadyProcessed(
            f"report {report.report_id} was already processed at "
            f"{report.processed_at.isoformat() if report.processed_at else 'unknown'}; "
            f"reprocessing would duplicate its events and inflate corroboration"
        )

    signals = detect_injection(report.raw_text)
    report.suspected_injection = bool(signals)
    report.injection_signals = signals

    # The model sees a view with instruction-shaped lines removed. The operator
    # always sees the original.
    view = model_view(report.raw_text)
    segments = segment(view)
    report.segment_count = len(segments)

    created: list[Event] = []
    total_latency = 0
    for idx, seg in enumerate(segments):
        try:
            result = client.extract_segment(seg)
        except ApertusUnavailable:
            _audit(session, "report", report.report_id, "extraction_failed",
                   model_used=st.apertus_model_id)
            raise
        total_latency += result.latency_ms

        if not result.events:
            _audit(session, "report", report.report_id, "extraction_empty",
                   model_used=st.apertus_model_id,
                   prompt_version=st.prompt_version_extraction,
                   detail={"segment_index": idx})
            continue

        for raw_ev in result.events:
            cleaned = normalize_extraction(
                event_type=raw_ev.event_type,
                location_text=raw_ev.location_text,
                people_affected=raw_ev.people_affected,
                severity=raw_ev.severity,
                time_reference=raw_ev.time_reference,
                description=raw_ev.description,
                source_segment=seg,
                suspected_injection=bool(signals),
            )
            ev = Event(
                report_id=report.report_id,
                segment_index=idx,
                event_type=cleaned["event_type"],
                description=cleaned["description"],
                location_text=cleaned["location_text"],
                people_affected=cleaned["people_affected"],
                severity=cleaned["severity"],
                time_reference=cleaned["time_reference"],
                uncertainties=uncertainty_notes(cleaned, seg),
                model_confidence=None,          # never taken from the model
                evidence_confidence=0.0,        # filled by corroboration stage
                verification_state=VerificationState.UNVERIFIED.value,
                independent_source_count=1,
                model_used=st.apertus_model_id,
                model_label=st.apertus_model_label,
                prompt_version=st.prompt_version_extraction,
                grammar_version=GRAMMAR_VERSION,
                extraction_latency_ms=result.latency_ms,
                guards_applied=cleaned["guards_applied"],
            )
            session.add(ev)
            session.flush()
            created.append(ev)

            # Hard invariant: no event without provenance. Every factual field
            # that has a value gets a link back to the report it came from.
            for field_path in (
                "event_type", "location_text", "people_affected",
                "severity", "time_reference", "description",
            ):
                if ev_value(ev, field_path) is not None:
                    session.add(EvidenceLink(
                        event_id=ev.event_id, report_id=report.report_id,
                        field_path=field_path, reason="extracted_directly",
                    ))

    report.processed = True
    report.processed_at = utcnow()
    _audit(session, "report", report.report_id, "extracted",
           model_used=st.apertus_model_id,
           model_version=st.apertus_model_label,
           prompt_version=st.prompt_version_extraction,
           input_ref=report.report_id,
           output_summary={"events": len(created), "segments": len(segments),
                           "latency_ms": total_latency},
           detail={"injection_signals": len(signals)})
    return {"report_id": report.report_id, "events": len(created),
            "segments": len(segments), "latency_ms": total_latency}


def ev_value(event, field: str):
    return getattr(event, field, None)


# ---- deduplication & corroboration ------------------------------------------

def apply_deduplication(session) -> dict:
    """Group duplicate reports, then count independent groups per event.

    Corroboration is counted in *groups*, not rows. Ten forwarded copies of one
    report move an event from one source to one source.
    """
    reports = session.query(Report).order_by(Report.ingested_at).all()
    buckets: dict[str, list[Report]] = {}
    for r in reports:
        buckets.setdefault(r.text_hash, []).append(r)

    groups_created = 0
    for fingerprint, members in buckets.items():
        if len(members) < 2:
            continue
        gid = f"DUP-{fingerprint[:10].upper()}"
        existing = session.get(DuplicateGroup, gid)
        if existing:
            continue
        session.add(DuplicateGroup(
            group_id=gid, method="exact_fingerprint", similarity=1.0,
            member_report_ids=[m.report_id for m in members],
            member_event_ids=[e.event_id for m in members
                              for e in m.events],
        ))
        for m in members:
            for e in m.events:
                e.duplicate_group_id = gid
        _audit(session, "duplicate_group", gid, "deduplicated",
               detail={"members": len(members)})
        groups_created += 1

    recount_corroboration(session)
    return {"groups": groups_created}


def resolve_locations(session, client: ApertusClient | None = None) -> dict:
    """Attach a cross-language canonical place to each event.

    Runs after extraction and before contradiction detection, because both need
    to know that 'pont central' and 'الطريق' name the same bridge. Costs one
    model call per new place mention, cached per (mention, candidate set).
    """
    st = get_settings()
    if not st.enable_entity_resolution:
        return {"resolved": 0, "distinct_places": 0, "enabled": False}

    client = client or ApertusClient(st)
    resolver = entities.LocationResolver(client)

    events = session.query(Event).order_by(Event.created_at).all()
    place_events = [e for e in events
                    if entities.is_place_event(e.event_type) and e.location_text]
    if not place_events:
        return {"resolved": 0, "distinct_places": 0, "enabled": True}

    mentions = [e.location_text for e in place_events]
    mapping = resolver.canonical_set(mentions)

    changed = 0
    for ev in place_events:
        canon = mapping.get(ev.location_text, ev.location_text)
        if canon and canon != ev.location_text:
            if ev.resolved_location != canon:
                ev.resolved_location = canon
                ev.location_confidence = 0.8
                changed += 1
        elif ev.resolved_location:
            # A re-run must not leave stale resolutions behind.
            ev.resolved_location = None
            ev.location_confidence = None
    _audit(session, "pipeline", "locations", "entity_resolution",
           model_used=st.apertus_model_id,
           prompt_version=st.prompt_version_normalization,
           output_summary={"distinct_places": len(set(mapping.values())),
                           "updated": changed})
    return {"resolved": changed, "distinct_places": len(set(mapping.values())),
            "enabled": True}


def recount_corroboration(session) -> None:
    """Recompute independent source counts and evidence confidence for all events."""
    st = get_settings()
    reports = {r.report_id: r for r in session.query(Report).all()}
    all_events = session.query(Event).all()

    # Injection-flagged reports are demoted first, and unconditionally. Doing it
    # here rather than inside the grouping loop means an event with no location
    # is covered too: previously such events fell outside the grouping key and
    # kept a default independent_source_count of 1, which is exactly the number
    # an injected claim would need to look corroborated.
    for ev in all_events:
        rep = reports.get(ev.report_id)
        if rep is not None and rep.suspected_injection:
            ev.independent_source_count = 0
            ev.evidence_confidence = 0.0
            if ev.verification_state in (
                VerificationState.CORROBORATED.value,
                VerificationState.DISPUTED.value,
            ):
                ev.verification_state = VerificationState.UNVERIFIED.value

    by_key: dict[tuple, list[Event]] = {}
    for ev in all_events:
        if ev.verification_state == VerificationState.REJECTED.value:
            continue
        place = ev.resolved_location or ev.location_text
        if place:
            key = (ev.event_type, contradiction_svc.canon_location(place))
            by_key.setdefault(key, []).append(ev)

    for events in by_key.values():
        groups: dict[str, list[Event]] = {}
        langs: set[str] = set()
        weights: list[float] = []
        for ev in events:
            rep = reports.get(ev.report_id)
            # A report flagged for prompt injection contributes no independent
            # source. Its text is attacker-controlled, so letting it count
            # would let anyone manufacture corroboration by submitting "the
            # bridge is open" fifty times alongside an injection attempt. The
            # events are still extracted and shown, because hiding them would
            # hide the attempt; they just cannot promote anything.
            if rep is not None and rep.suspected_injection:
                ev.independent_source_count = 0
                ev.evidence_confidence = 0.0
                if ev.verification_state in (
                    VerificationState.CORROBORATED.value,
                    VerificationState.DISPUTED.value,
                ):
                    ev.verification_state = VerificationState.UNVERIFIED.value
                continue
            key = dedup.independence_key(rep) if rep else ev.report_id
            groups.setdefault(key, []).append(ev)
            if rep:
                langs.add(rep.source_language)
                weights.append(dedup.corroboration_weight(rep))
        n_independent = len(groups)
        base = sum(weights) / len(weights) if weights else 1.0
        conf = dedup.evidence_confidence(n_independent, len(langs), base)

        disputed = any(ev.verification_state == VerificationState.DISPUTED.value
                       for ev in events)
        for ev in events:
            ev.independent_source_count = n_independent
            ev.evidence_confidence = conf
            if ev.verification_state == VerificationState.UNVERIFIED.value:
                if disputed:
                    ev.verification_state = VerificationState.DISPUTED.value
                elif n_independent >= 2 and conf >= st.evidence_confidence_floor:
                    ev.verification_state = VerificationState.CORROBORATED.value

        if disputed:
            # A dispute caps evidence strength. Two sources that disagree are
            # not stronger evidence than two that agree.
            for ev in events:
                ev.evidence_confidence = min(ev.evidence_confidence, 0.49)


# ---- contradiction -----------------------------------------------------------

def _conflict_side(ev: Event) -> str:
    """Render one side of a contradiction as the differing values, not prose."""
    parts = [f"type {ev.event_type}"]
    place = ev.resolved_location or ev.location_text
    if place:
        parts.append(f"at {place}")
    if ev.people_affected is not None:
        parts.append(f"people {ev.people_affected}")
    if ev.severity and ev.severity != "unknown":
        parts.append(f"severity {ev.severity}")
    if ev.time_reference:
        parts.append(f"time {ev.time_reference}")
    return ", ".join(parts)


def detect_conflicts(session, client: ApertusClient | None = None) -> dict:
    """Detect contradictions, mark both sides disputed, never resolve."""
    st = get_settings()
    client = client or ApertusClient(st)
    events = session.query(Event).order_by(Event.created_at).all()
    found = contradiction_svc.detect_all(events)

    existing = {
        (c.event_a_id, c.event_b_id, c.conflict_type)
        for c in session.query(Conflict).all()
    }
    created = 0
    for c in found:
        key = (c.event_a.event_id, c.event_b.event_id, c.conflict_type)
        if key in existing or (c.event_b.event_id, c.event_a.event_id,
                               c.conflict_type) in existing:
            continue
        # The model is given the actual differing values, not a prose summary.
        # An earlier version passed only one side's description, and the model
        # dutifully restated it ("blocked"), which is not an explanation of a
        # disagreement.
        explanation = client.explain_conflict(
            _conflict_side(c.event_a), _conflict_side(c.event_b),
            c.conflict_type,
        ) or contradiction_svc.default_explanation(c)
        conflict = Conflict(
            event_a_id=c.event_a.event_id,
            event_b_id=c.event_b.event_id,
            conflict_type=c.conflict_type,
            severity=c.severity,
            explanation=explanation,
            evidence=c.evidence,
            status="unresolved",
            detection_rule=c.rule,
            model_used=st.apertus_model_id if explanation !=
            contradiction_svc.default_explanation(c) else None,
        )
        session.add(conflict)
        session.flush()
        for ev in (c.event_a, c.event_b):
            if ev.verification_state != VerificationState.REJECTED.value:
                ev.verification_state = VerificationState.DISPUTED.value
        _audit(session, "conflict", conflict.conflict_id, "conflict_detected",
               model_used=conflict.model_used,
               prompt_version=st.prompt_version_conflict,
               input_ref=f"{c.event_a.event_id}|{c.event_b.event_id}",
               output_summary={"type": c.conflict_type, "severity": c.severity,
                               "rule": c.rule})
        created += 1
    return {"conflicts": created, "checked_pairs": len(events) * (len(events) - 1) // 2}