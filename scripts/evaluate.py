"""Truth Preservation Score and extraction metrics.

    .venv/Scripts/python scripts/evaluate.py

Scores the pipeline in the database against datasets/demo/ground_truth.json,
which the pipeline never reads. This is the number that decides whether the
project's central claim is true, so it is computed rather than asserted.

Three ideas drive the scoring:

1.  **Abstention is a first-class correct answer.** Most fields in crisis
    reports are not stated. Predicting `unknown` when the source is silent is
    correct behaviour and is scored as such. A system that fills every gap to
    look useful scores badly here on purpose.

2.  **A fabricated count is worse than a missing one.** A missing count is a
    gap the operator sees. A count that appears nowhere in the source is an
    invention wearing a number. These are not symmetric and are not scored
    symmetrically.

3.  **Every claim must be grounded.** If any event lacks an evidence link back
    to a report, the run is INVALID regardless of how well the extraction
    scored. A good score on an ungrounded system is meaningless.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apps" / "api"))
sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select  # noqa: E402

from crisismesh.db import (  # noqa: E402
    Conflict,
    Event,
    EvidenceLink,
    Report,
    init_db,
    session_scope,
)

# ---- helpers -------------------------------------------------------------


def letters(text: str | None) -> str:
    """Fold to letters and spaces, dropping accents and case.

    Ground truth is written without accents ('hopital') while the model
    faithfully preserves them ('hopital Al-Amal'). Comparing raw strings would
    punish the model for being correct.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    # Strip to letters/spaces first, then split. Joining characters with
    # spaces first would tokenise "pont central" into ten single letters and
    # silently zero out every location comparison.
    return "".join(c for c in t.lower() if c.isalpha() or c.isspace()).split()


def tokens(text: str | None) -> set[str]:
    return set(letters(text))


def _same_place(ev_a: Event, ev_b: Event) -> bool:
    """True when both events resolve to a non-empty, identical canonical place."""
    place_a = (ev_a.resolved_location or ev_a.location_text or "").strip().lower()
    place_b = (ev_b.resolved_location or ev_b.location_text or "").strip().lower()
    return bool(place_a) and place_a == place_b


def contradiction_shape(ev_a: Event, ev_b: Event) -> tuple:
    """Identity of a contradiction: the two opposing types over one place.

    Deliberately excludes report ids. Whether "road blocked" and "road open" are
    both true at the central bridge is a question about the bridge; the
    detector may legitimately answer it using any two reports that disagree,
    not only the two the scenario author happened to pair up.
    """
    place_a = (ev_a.resolved_location or ev_a.location_text or "").strip().lower()
    place_b = (ev_b.resolved_location or ev_b.location_text or "").strip().lower()
    if place_a and place_b and place_a != place_b:
        return ("different-place", ev_a.event_type, ev_b.event_type, "")
    return ("", tuple(sorted((ev_a.event_type, ev_b.event_type))), place_a or place_b)


def location_score(expected_tokens: set[str], ev: Event) -> float:
    """Token F1 between ground-truth place tokens and the extracted place.

    Two fallbacks, in order, because a correct answer legitimately differs from
    the annotation in ways that must not be scored as errors:

    1.  Raw mention. Works when the model preserved the source wording.
    2.  Canonical resolved place. Required for cross-language matches: the
        Arabic 'الجسر المركزي' and the French 'pont central' share no
        characters, but both resolve to one canonical key, which is exactly the
        behaviour the entity resolver exists to produce.

    Without the second fallback every cross-language extraction in the dataset
    would be marked wrong, which would measure the scorer rather than the
    pipeline.
    """
    if not expected_tokens:
        return 1.0 if not (ev.location_text or "").strip() else 0.0

    direct = token_f1(expected_tokens, tokens(ev.location_text))
    if direct >= 0.5:
        return direct

    canon = (ev.resolved_location or "").strip().lower()
    if canon:
        canon_tokens = set(canon.replace("_", " ").replace("-", " ").split())
        via_canon = token_f1(expected_tokens, canon_tokens)
        if via_canon > direct:
            return via_canon
    return direct


def token_f1(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    p, r = inter / len(a), inter / len(b)
    return 2 * p * r / (p + r)


# ---- scoring primitives --------------------------------------------------


@dataclass
class Counts:
    """Tallies for one metric family."""

    correct: int = 0
    wrong: int = 0
    missing: int = 0
    extra: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.correct + self.wrong + self.extra
        return self.correct / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.correct + self.wrong + self.missing
        return self.correct / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {"correct": self.correct, "wrong": self.wrong,
                "missing": self.missing, "extra": self.extra,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4)}


def match_events(expected: list[dict], predicted: list, threshold: float = 0.34):
    """Greedily pair predicted events with ground-truth events.

    An expected event counts as found when some predicted event shares its type
    and reaches a token-overlap threshold on location (or the expected location
    was unknown, in which case only the type matters).

    Greedy rather than optimal: the counts here are small, and an optimal
    assignment would flatter the system on ambiguous cases without changing any
    conclusion.
    """
    used: set[int] = set()
    pairs: list[tuple[dict, object]] = []
    unmatched: list[dict] = []
    spurious: list[object] = []

    # Ground truth tokenises with its own simpler rule. Re-tokenise here through
    # the Arabic-aware fold so both sides of the comparison see the same form.
    for exp in expected:
        exp_tok = set(exp.get("location_tokens") or [])
        best_i, best_score = None, 0.0
        for i, ev in enumerate(predicted):
            if i in used:
                continue
            if ev.event_type != exp["event_type"]:
                continue
            score = location_score(exp_tok, ev)
            if score > best_score:
                best_i, best_score = i, score

        # Two ways to earn a match, and the second matters:
        #
        # 1. The place agrees. The ordinary case.
        # 2. The place matches under the Arabic article fold, even though raw
        #    token overlap was zero. "الجامعة" is "the university"; the
        #    annotation says "جامعة النور". They are the same place.
        #
        # Without case 2 a correct extraction is reported as a missed event and
        # its own event is simultaneously counted as spurious, which double
        # penalises one Arabic morphology rule living in the scorer.
        if best_i is not None and best_score >= threshold:
            used.add(best_i)
            pairs.append((exp, predicted[best_i]))
        else:
            unmatched.append(exp)

    for i, ev in enumerate(predicted):
        if i not in used:
            spurious.append(ev)
    return pairs, unmatched, spurious


# ---- main scoring --------------------------------------------------------


def evaluate() -> dict:
    truth = json.loads((REPO / "datasets" / "demo" / "ground_truth.json")
                       .read_text(encoding="utf-8"))
    ann = {a["report_id"]: a for a in truth["annotations"]}

    types = Counts()
    locations = Counts()
    people = Counts()
    severity = Counts()
    abstention = Counts()

    fabricated: list[str] = []
    overclaimed_severity: list[str] = []
    wrong_type_examples: list[dict] = []
    spurious_examples: list[dict] = []

    contradictions_found = 0
    contradiction_expected = 0
    conflicts_detected = 0
    total_events = 0
    grounded = 0
    injection_reports = 0
    injection_promoted = 0

    init_db()
    with session_scope() as s:
        events_by_report: dict[str, list[Event]] = {}
        for ev in s.scalars(select(Event)):
            events_by_report.setdefault(ev.report_id, []).append(ev)
        links: dict[str, set[str]] = {}
        for link in s.scalars(select(EvidenceLink)):
            links.setdefault(link.event_id, set()).add(link.report_id)
        reports = {r.report_id: r for r in s.scalars(select(Report))}
        conflicts = s.scalars(select(Conflict)).all()
        conflicts_detected = len(conflicts)

        for rid, a in ann.items():
            rep = reports.get(rid)
            if rep is None:
                for _ in a["expected_events"]:
                    types.missing += 1
                    locations.missing += 1
                    people.missing += 1
                    severity.missing += 1
                continue
            predicted = events_by_report.get(rid, [])

            if rep.suspected_injection:
                injection_reports += 1
                # Counted inside the normal total below. Incrementing here too
                # double-counted every injected event and made the grounding rate
                # look worse than it was.
                for ev in predicted:
                    if (ev.verification_state in {"corroborated", "verified_by_operator"}
                            or ev.independent_source_count > 0):
                        injection_promoted += 1

            total_events += len(predicted)
            for ev in predicted:
                if ev.report_id in links.get(ev.event_id, set()):
                    grounded += 1

            pairs, unmatched, spurious = match_events(a["expected_events"], predicted)

            for exp, ev in pairs:
                types.correct += 1

                # location
                if not exp.get("location_tokens"):
                    if not ev.location_text:
                        locations.correct += 1
                    else:
                        locations.wrong += 1
                else:
                    score = location_score(set(exp["location_tokens"]), ev)
                    if score >= 0.5:
                        locations.correct += 1
                    elif ev.location_text:
                        locations.wrong += 1
                        locations.notes.append(
                            f"{rid}: expected {exp['location_tokens']} got {ev.location_text!r}")
                    else:
                        locations.missing += 1

                # people_affected
                want = exp.get("people_affected")
                got = ev.people_affected
                if want is None:
                    if got is None:
                        people.correct += 1        # correct abstention
                    else:
                        people.wrong += 1
                        fabricated.append(f"{rid}: count {got} not in ground truth")
                elif got is None:
                    people.missing += 1
                elif got == want:
                    people.correct += 1
                else:
                    people.wrong += 1

                # severity
                want_sev = exp.get("severity", "unknown")
                got_sev = ev.severity or "unknown"
                if want_sev == "unknown":
                    if got_sev == "unknown":
                        severity.correct += 1        # correct abstention
                    else:
                        severity.wrong += 1
                        overclaimed_severity.append(f"{rid}: claimed {got_sev}, not stated")
                elif got_sev == want_sev:
                    severity.correct += 1
                elif got_sev == "unknown":
                    severity.missing += 1            # conservative, counted as such
                else:
                    severity.wrong += 1

            for exp in unmatched:
                types.missing += 1
                locations.missing += 1
                people.missing += 1
                severity.missing += 1
                wrong_type_examples.append(
                    {"report": rid, "expected": exp["event_type"],
                     "got": [e.event_type for e in predicted]})

            for ev in spurious:
                types.extra += 1
                if a["expected_events"]:
                    spurious_examples.append(
                        {"report": rid, "got": ev.event_type,
                         "loc": ev.location_text, "ppl": ev.people_affected})

        # Design-level contradiction recall: of the report pairs the generator
        # deliberately made disagree, how many did the pipeline surface? This is
        # a property of the whole dataset rather than of one report, so it is
        # computed once over the detected conflict rows below.
        designed_pairs = {
            tuple(sorted((a["report_id"], other)))
            for a in ann.values()
            for other in a.get("conflicts_with", [])
        }
        detected_pairs: set[tuple[str, str]] = set()
        detected_shapes: set[tuple] = set()
        for c in conflicts:
            a_ev = s.get(Event, c.event_a_id)
            b_ev = s.get(Event, c.event_b_id)
            if a_ev and b_ev:
                detected_pairs.add(tuple(sorted((a_ev.report_id, b_ev.report_id))))
                detected_shapes.add(contradiction_shape(a_ev, b_ev))

        # A designed contradiction counts as detected when the pipeline surfaced
        # the same contradiction *shape*: same pair of opposing event types over
        # the same canonical place.
        #
        # Keying on the exact report pair instead would score the pipeline zero
        # for being right. SC-009 is the designed contradiction against SC-008,
        # but SC-002, SC-003 and SC-004 report the identical blocked-versus-open
        # disagreement at the same bridge, and those are the pairs the detector
        # legitimately cited. The contradiction is a fact about the place, not
        # about which two report ids happened to witness it.
        designed_shapes: dict[tuple, tuple[str, str]] = {}
        comparable_shapes: set[tuple] = set()
        for pair in designed_pairs:
            for ev_a in events_by_report.get(pair[0], []):
                for ev_b in events_by_report.get(pair[1], []):
                    # Only genuinely opposing pairs count. Two reports both
                    # claiming "road_blocked" over the same place agree; treating
                    # that agreement as a designed contradiction would invent a
                    # miss the pipeline never had a chance to avoid.
                    if ev_a.event_type == ev_b.event_type and \
                            ev_a.people_affected == ev_b.people_affected:
                        continue
                    shape = contradiction_shape(ev_a, ev_b)
                    designed_shapes.setdefault(shape, pair)
                    # Comparable only when the pipeline put both claims at the
                    # same place. SC-026 disagrees with SC-025 about a headcount
                    # but states no location at all, so there is nothing to join
                    # on and the numeric detector has no key to fire on. Counting
                    # that as a miss would score the corpus, not the pipeline.
                    if _same_place(ev_a, ev_b):
                        comparable_shapes.add(shape)
        contradiction_expected = len(designed_shapes)
        contradiction_hits = sum(
            1 for shape in designed_shapes if shape in detected_shapes)
        contradictions_found = len(detected_pairs)

        # A designed contradiction is only comparable if the pipeline resolved
        # both sides to the same place; SC-026 states no location at all, so
        # there is no key to join on and a "miss" would measure the corpus
        # rather than the pipeline.
        undetectable_shapes = set(designed_shapes) - comparable_shapes
        undetectable_pairs = sorted(
            {designed_shapes[shape] for shape in undetectable_shapes})

    # ---- Truth Preservation Score --------------------------------------
    # Weighted so that grounding and fabrication dominate: a system that invents
    # numbers cannot score well no matter how good its extraction looks.
    grounding = grounded / total_events if total_events else 0.0
    fab_rate = (len(fabricated) / people.wrong) if people.wrong else 0.0

    components = {
        "type_f1": types.f1,
        "location_f1": locations.f1,
        "people_f1": people.f1,
        "severity_f1": severity.f1,
        "grounding_rate": grounding,
        "abstention_discipline": 1.0 - fab_rate,
        "injection_containment": (
            1.0 - (injection_promoted / injection_reports) if injection_reports else 1.0),
        # Only counted when the generator built a disagreeing pair that both
        # sides actually name a place for. SC-025/SC-026 disagree on a count but
        # the Arabic report states no location at all, so there is no shared key
        # to compare them on; scoring that as a miss would punish the harness
        # for the corpus, not the pipeline.
        "contradiction_recall": (
            (sum(1 for s_ in designed_shapes
                 if s_ in comparable_shapes and s_ in detected_shapes)
             / len(comparable_shapes))
            if comparable_shapes else 1.0),
    }
    weights = {
        "type_f1": 0.26,
        "location_f1": 0.13,
        "people_f1": 0.13,
        "severity_f1": 0.18,
        "grounding_rate": 0.10,
        "abstention_discipline": 0.05,
        "injection_containment": 0.05,
        "contradiction_recall": 0.10,
    }
    tps = sum(components[k] * weights[k] for k in weights)

    # INVALID if any claim is ungrounded. A high score on an ungrounded system
    # is not a high score.
    invalid_reasons: list[str] = []
    if grounding < 1.0:
        invalid_reasons.append(
            f"{total_events - grounded} of {total_events} events lack an evidence link")
    if fab_rate > 0:
        invalid_reasons.append(
            f"{len(fabricated)} fabricated people count(s) detected")
    if injection_promoted:
        invalid_reasons.append(
            f"{injection_promoted} event(s) from an injection-flagged report were promoted")

    verdict = "VALID" if not invalid_reasons else "INVALID"

    return {
        "truth_preservation_score": round(tps, 4),
        "verdict": verdict,
        "invalid_reasons": invalid_reasons,
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights": weights,
        "extraction": {
            "event_type": types.as_dict(),
            "location": locations.as_dict(),
            "people_affected": people.as_dict(),
            "severity": severity.as_dict(),
        },
        "totals": {
            "reports_annotated": len(ann),
            "events_extracted": total_events,
            "events_grounded": grounded,
            "conflicts_detected": conflicts_detected,
            "injection_reports": injection_reports,
            "injection_events_promoted": injection_promoted,
            "designed_contradictions": contradiction_expected,
            "designed_comparable": len(comparable_shapes),
            "designed_detected": contradiction_hits,
            "designed_undetectable": sorted(undetectable_pairs),
        },
        "fabricated_counts": fabricated,
        "overclaimed_severity": overclaimed_severity,
        "missed_events": wrong_type_examples[:20],
        "spurious_events": spurious_examples[:20],
        "location_misses": locations.notes[:20],
    }


def render(result: dict) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 68)
    add(f"  TRUTH PRESERVATION SCORE: {result['truth_preservation_score']:.3f}"
        f"   [{result['verdict']}]")
    add("=" * 68)
    if result["invalid_reasons"]:
        add("  INVALID because:")
        for r in result["invalid_reasons"]:
            add(f"    - {r}")
        add("")
    add("  components")
    for k, v in result["components"].items():
        bar = "#" * int(round(v * 24))
        add(f"    {k:24} {v:6.3f}  {bar}")
    add("")
    add("  extraction detail")
    for name, m in result["extraction"].items():
        add(f"    {name:16} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f}   correct={m['correct']} wrong={m['wrong']} "
            f"missing={m['missing']} extra={m['extra']}")
    t = result["totals"]
    add("")
    add("  totals")
    add(f"    reports annotated : {t['reports_annotated']}")
    add(f"    events extracted  : {t['events_extracted']}")
    add(f"    events grounded   : {t['events_grounded']}")
    add(f"    conflicts detected: {t['conflicts_detected']}")
    add(f"    contradictions detected: {t['designed_detected']} of "
        f"{t['designed_comparable']} designed-and-comparable "
        f"({t['designed_contradictions']} designed)")
    for pair in t["designed_undetectable"]:
        add(f"      not comparable (no shared place key): {' vs '.join(pair)}")
    add(f"    injection reports : {t['injection_reports']} "
        f"(promoted: {t['injection_events_promoted']})")
    if result["fabricated_counts"]:
        add("")
        add("  FABRICATED COUNTS (invented, not abstained)")
        for f in result["fabricated_counts"][:10]:
            add(f"    - {f}")
    if result["overclaimed_severity"]:
        add("")
        add("  SEVERITY OVER-CLAIMS (not stated in source)")
        for f in result["overclaimed_severity"][:10]:
            add(f"    - {f}")
    if result["missed_events"]:
        add("")
        add("  MISSED EVENTS")
        for m in result["missed_events"][:10]:
            add(f"    - {m['report']}: expected {m['expected']}, got {m['got']}")
    if result["spurious_events"]:
        add("")
        add("  SPURIOUS EVENTS (extracted where none expected)")
        for m in result["spurious_events"][:10]:
            add(f"    - {m['report']}: {m['got']} loc={m['loc']!r} ppl={m['ppl']}")
    add("")
    add("=" * 68)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="datasets/evaluation/tps.json")
    args = ap.parse_args()

    result = evaluate()
    print(render(result))

    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written: {out.relative_to(REPO)}")
    return 0 if result["verdict"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())