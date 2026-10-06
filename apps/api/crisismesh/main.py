"""FastAPI application.

Route groups mirror the operator's mental model rather than the pipeline's:
ingest reports, inspect events, resolve conflicts, read the graph, audit.

Operator-mutating endpoints require a token. Read endpoints are open because
they only expose data that was already submitted to the system.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from . import __version__
from .apertus import ApertusClient, ApertusUnavailable
from .config import get_settings
from .db import (
    AuditLog,
    Conflict,
    DuplicateGroup,
    Event,
    Report,
    init_db,
    session_scope,
)
from .schemas import (
    AuditOut,
    BatchIn,
    ConflictDetail,
    ConflictOut,
    EventDetail,
    EventOut,
    HealthOut,
    MetricsOut,
    ReportIn,
    ReportOut,
    ResolveIn,
    VerifyIn,
)
from .services import pipeline as pipeline_svc
from .services.graph import build_graph
from .taxonomy import ConflictStatus, VerificationState, verification_allowed

app = FastAPI(
    title="CrisisMesh API",
    version=__version__,
    description=(
        "Evidence-aware crisis information intelligence built on the Apertus "
        "model family. Every claim traces to an immutable report. Conflicts "
        "are surfaced, never auto-resolved."
    ),
)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"], allow_headers=["*"],
)


def require_operator(authorization: str | None = Header(default=None)) -> str:
    """Token guard for operator mutations.

    Verification and conflict resolution change what the system asserts about
    the world, so they are authenticated. An empty token in .env disables the
    guard for local development; that is documented rather than hidden.
    """
    expected = get_settings().operator_token
    if not expected:
        return "dev-operator"
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="operator token required")
    return "operator"


@app.on_event("startup")
def _startup() -> None:
    init_db()


# ---- health -----------------------------------------------------------------

@app.get("/health", response_model=HealthOut, tags=["ops"])
def health() -> HealthOut:
    st = get_settings()
    apertus = ApertusClient(st).health()
    with session_scope() as s:
        reports = s.scalar(select(func.count()).select_from(Report)) or 0
        events = s.scalar(select(func.count()).select_from(Event)) or 0
    return HealthOut(
        status="ok" if apertus.get("available") else "degraded",
        version=__version__, apertus=apertus, reports=reports, events=events,
    )


@app.get("/metrics", response_model=MetricsOut, tags=["ops"])
def metrics() -> MetricsOut:
    st = get_settings()
    with session_scope() as s:
        reports = s.scalars(select(Report)).all()
        events = s.scalars(select(Event)).all()
        conflicts = s.scalars(select(Conflict)).all()

        by_state: dict[str, int] = {}
        for e in events:
            by_state[e.verification_state] = by_state.get(e.verification_state, 0) + 1

        # people_affected is summed only over events that state a number.
        # Events where the count is unknown stay unknown in the total rather
        # than being treated as zero, so we report a lower bound.
        stated = [e.people_affected for e in events if e.people_affected is not None]
        unknown_count = sum(
            1 for e in events
            if e.people_affected is None and e.event_type in {
                "injury", "missing_person", "trapped_person", "medical_need",
            }
        )
        locations = {
            e.location_text.strip().lower()
            for e in events
            if e.location_text and e.verification_state != "rejected"
        }
        return MetricsOut(
            reports_total=len(reports),
            reports_processed=sum(1 for r in reports if r.processed),
            events_total=len(events),
            events_unverified=by_state.get(VerificationState.UNVERIFIED.value, 0),
            events_corroborated=by_state.get(VerificationState.CORROBORATED.value, 0),
            events_disputed=by_state.get(VerificationState.DISPUTED.value, 0),
            events_verified_by_operator=by_state.get(
                VerificationState.VERIFIED_BY_OPERATOR.value, 0),
            events_rejected=by_state.get(VerificationState.REJECTED.value, 0),
            events_critical=sum(1 for e in events if e.severity == "critical"),
            people_affected_reported=sum(stated),
            people_affected_lower_bound=sum(stated),
            locations_affected=len(locations),
            conflicts_unresolved=sum(
                1 for c in conflicts
                if c.status == ConflictStatus.UNRESOLVED.value),
            conflicts_resolved=sum(
                1 for c in conflicts
                if c.status != ConflictStatus.UNRESOLVED.value),
            duplicate_groups=s.scalar(select(func.count()).select_from(DuplicateGroup)) or 0,
            independent_source_max=max(
                (e.independent_source_count for e in events), default=0),
            suspected_injection_reports=sum(1 for r in reports if r.suspected_injection),
            model_label=st.apertus_model_label, model_used=st.apertus_model_id,
        )


# ---- reports ----------------------------------------------------------------

@app.get("/reports", response_model=list[ReportOut], tags=["reports"])
def list_reports(
    lang: str | None = None,
    processed: bool | None = None,
    injection: bool | None = None,
    limit: int = Query(200, le=1000),
) -> list[ReportOut]:
    with session_scope() as s:
        q = select(Report).order_by(Report.ingested_at.desc()).limit(limit)
        if lang:
            q = q.where(Report.source_language == lang)
        if processed is not None:
            q = q.where(Report.processed == processed)
        if injection is not None:
            q = q.where(Report.suspected_injection == injection)
        return [ReportOut.model_validate(r) for r in s.scalars(q)]


@app.get("/reports/{report_id}", response_model=ReportOut, tags=["reports"])
def get_report(report_id: str) -> ReportOut:
    with session_scope() as s:
        r = s.get(Report, report_id)
        if not r:
            raise HTTPException(404, f"report {report_id} not found")
        return ReportOut.model_validate(r)


@app.post("/reports", response_model=ReportOut, tags=["reports"], status_code=201)
def create_report(payload: ReportIn) -> ReportOut:
    with session_scope() as s:
        r = pipeline_svc.ingest_report(
            s, payload.message, source_language=payload.language,
            source_type=payload.source_type.value,
            location_hint=payload.location_hint,
            reporter_id=payload.reporter_id,
            reported_at=payload.timestamp, metadata=payload.metadata)
        return ReportOut.model_validate(r)


@app.post("/reports/batch", tags=["reports"], status_code=201)
def create_reports_batch(payload: BatchIn) -> dict:
    with session_scope() as s:
        made = []
        for item in payload.reports:
            r = pipeline_svc.ingest_report(
                s, item.message, source_language=item.language,
                source_type=item.source_type.value,
                location_hint=item.location_hint,
                reporter_id=item.reporter_id, reported_at=item.timestamp,
                metadata=item.metadata)
            made.append(r.report_id)
        return {"created": len(made), "report_ids": made}


# ---- processing -------------------------------------------------------------

@app.post("/process/{report_id}", tags=["processing"])
def process_one(report_id: str) -> dict:
    with session_scope() as s:
        r = s.get(Report, report_id)
        if not r:
            raise HTTPException(404, f"report {report_id} not found")
        try:
            result = pipeline_svc.process_report(s, r)
        except pipeline_svc.ReportAlreadyProcessed as exc:
            raise HTTPException(409, str(exc)) from exc
        except ApertusUnavailable as exc:
            raise HTTPException(503, f"Apertus unavailable: {exc}") from exc
        pipeline_svc.resolve_locations(s)
        pipeline_svc.apply_deduplication(s)
        pipeline_svc.detect_conflicts(s)
        return result


@app.post("/process/all", tags=["processing"])
def process_all(limit: int = Query(500, le=2000)) -> dict:
    """Process every unprocessed report, then re-run the evidence stages."""
    with session_scope() as s:
        pending = s.scalars(
            select(Report).where(Report.processed == False).limit(limit)  # noqa: E712
        ).all()
        processed, failed, events = 0, 0, 0
        for r in pending:
            try:
                res = pipeline_svc.process_report(s, r)
                processed += 1
                events += res["events"]
            except (ApertusUnavailable, pipeline_svc.ReportAlreadyProcessed):
                failed += 1
        locs = pipeline_svc.resolve_locations(s)
        dedup = pipeline_svc.apply_deduplication(s)
        conf = pipeline_svc.detect_conflicts(s)
        return {"processed": processed, "failed": failed, "events": events,
                **locs, **dedup, **conf}


# ---- events -----------------------------------------------------------------

@app.get("/events", response_model=list[EventOut], tags=["events"])
def list_events(
    event_type: str | None = None,
    verification: str | None = None,
    severity: str | None = None,
    location: str | None = None,
    limit: int = Query(500, le=2000),
) -> list[EventOut]:
    with session_scope() as s:
        q = select(Event).order_by(Event.created_at.desc()).limit(limit)
        if event_type:
            q = q.where(Event.event_type == event_type)
        if verification:
            q = q.where(Event.verification_state == verification)
        if severity:
            q = q.where(Event.severity == severity)
        if location:
            q = q.where(Event.location_text.ilike(f"%{location}%"))
        return [EventOut.model_validate(e) for e in s.scalars(q)]


@app.get("/events/{event_id}", response_model=EventDetail, tags=["events"])
def get_event(event_id: str) -> EventDetail:
    with session_scope() as s:
        e = s.get(Event, event_id)
        if not e:
            raise HTTPException(404, f"event {event_id} not found")
        rep = s.get(Report, e.report_id)
        conflicts = s.scalars(
            select(Conflict).where(
                (Conflict.event_a_id == event_id) | (Conflict.event_b_id == event_id)
            )
        ).all()
        detail = EventDetail.model_validate(e)
        detail.source_language = rep.source_language if rep else "und"
        detail.raw_text = rep.raw_text if rep else ""
        detail.conflicts = [ConflictOut.model_validate(c) for c in conflicts]
        return detail


@app.post("/verify/{event_id}", response_model=EventOut,
          tags=["events"], dependencies=[Depends(require_operator)])
def verify_event(event_id: str, payload: VerifyIn) -> EventOut:
    """Record a human verification decision. Every transition is audited with
    the previous state, so who decided what and when is always recoverable."""
    with session_scope() as s:
        e = s.get(Event, event_id)
        if not e:
            raise HTTPException(404, f"event {event_id} not found")
        current = VerificationState(e.verification_state)
        if not verification_allowed(current, payload.state):
            raise HTTPException(
                409,
                f"transition {current} -> {payload.state} is not permitted; "
                f"allowed: {[t.value for t in VerificationState if verification_allowed(current, t)]}",
            )
        previous = e.verification_state
        e.verification_state = payload.state.value
        s.add(AuditLog(
            entity_type="event", entity_id=event_id, action="verified",
            output_summary={"previous": previous, "new": payload.state.value},
            actor=payload.actor or "operator", detail={"note": payload.note or ""},
        ))
        return EventOut.model_validate(e)


# ---- conflicts --------------------------------------------------------------

@app.get("/conflicts", response_model=list[ConflictOut], tags=["conflicts"])
def list_conflicts(status: str = "unresolved") -> list[ConflictOut]:
    with session_scope() as s:
        q = select(Conflict).order_by(Conflict.detected_at.desc())
        if status != "all":
            q = q.where(Conflict.status == status)
        return [ConflictOut.model_validate(c) for c in s.scalars(q)]


@app.get("/conflicts/{conflict_id}", response_model=ConflictDetail,
         tags=["conflicts"])
def get_conflict(conflict_id: str) -> ConflictDetail:
    with session_scope() as s:
        c = s.get(Conflict, conflict_id)
        if not c:
            raise HTTPException(404, f"conflict {conflict_id} not found")
        ea, eb = s.get(Event, c.event_a_id), s.get(Event, c.event_b_id)
        ra, rb = s.get(Report, ea.report_id), s.get(Report, eb.report_id)
        detail = ConflictDetail.model_validate(c)
        detail.event_a = EventOut.model_validate(ea)
        detail.event_b = EventOut.model_validate(eb)
        detail.report_a = ReportOut.model_validate(ra)
        detail.report_b = ReportOut.model_validate(rb)
        return detail


@app.post("/conflicts/{conflict_id}/resolve", response_model=ConflictOut,
          tags=["conflicts"], dependencies=[Depends(require_operator)])
def resolve_conflict(conflict_id: str, payload: ResolveIn) -> ConflictOut:
    """A human closes a dispute. The model never does this.

    Resolving a conflict records who decided, when, and what they concluded.
    It never deletes either side: both original reports remain readable
    forever, because a resolved-but-invisible disagreement teaches nobody.
    """
    with session_scope() as s:
        c = s.get(Conflict, conflict_id)
        if not c:
            raise HTTPException(404, f"conflict {conflict_id} not found")
        if payload.action == ConflictStatus.UNRESOLVED:
            raise HTTPException(409, "action must be resolved_by_operator or dismissed")
        c.status = payload.action.value
        c.resolved_by = payload.actor or "operator"
        c.resolved_at = func.now()
        c.resolution_note = payload.note
        s.add(AuditLog(
            entity_type="conflict", entity_id=conflict_id, action="resolved",
            output_summary={"status": c.status}, actor=c.resolved_by,
            detail={"note": payload.note},
        ))
        s.flush()
        if c.status == "dismissed":
            for e in (s.get(Event, c.event_a_id), s.get(Event, c.event_b_id)):
                if e.verification_state == VerificationState.DISPUTED.value:
                    e.verification_state = VerificationState.UNVERIFIED.value
        pipeline_svc.recount_corroboration(s)
        return ConflictOut.model_validate(c)


# ---- graph, audit -----------------------------------------------------------

@app.get("/graph", tags=["graph"])
def get_graph() -> dict:
    with session_scope() as s:
        return build_graph(
            s.scalars(select(Event)).all(),
            s.scalars(select(Report)).all(),
            s.scalars(select(Conflict)).all(),
        )


@app.get("/audit/{entity_id}", response_model=list[AuditOut], tags=["audit"])
def get_audit(entity_id: str) -> list[AuditOut]:
    with session_scope() as s:
        entries = s.scalars(
            select(AuditLog).where(AuditLog.entity_id == entity_id)
            .order_by(AuditLog.timestamp)
        ).all()
        return [AuditOut.model_validate(a) for a in entries]


from .dashboard import router as dashboard_router
app.include_router(dashboard_router, tags=["ui"])

@app.get("/ui")
def ui_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/")


# root is provided by dashboard router; avoid redefining with missing imports
# ensure dashboard root is mounted
app.include_router(dashboard_router, tags=["ui"], include_in_schema=False)
pass


@app.get("/audit/recent", response_model=list[AuditOut], tags=["audit"])
def recent_audit(limit: int = Query(100, le=1000)) -> list[AuditOut]:
    with session_scope() as s:
        entries = s.scalars(
            select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
        ).all()
        return [AuditOut.model_validate(a) for a in entries]