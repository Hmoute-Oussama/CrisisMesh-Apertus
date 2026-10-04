"""Contradiction detection.

Deliberately split from Apertus. Detection is deterministic code operating on
already-extracted fields; Apertus is consulted only to write a sentence
explaining the disagreement, and is explicitly forbidden from choosing a winner.

This ordering is the whole point. If the model detected conflicts it would
also be deciding which reports to believe, which is the failure the project
exists to prevent. A model that is confidently wrong is more dangerous than no
model at all, so we keep it away from adjudication.

Detected conflict types:
  status   road_blocked vs road_open for the same location
  numeric  differing people_affected for the same event type and location
  temporal differing explicit clock times for the same event and location
"""

from __future__ import annotations

import re
from datetime import datetime

from ..config import get_settings
from ..taxonomy import COUNTABLE_EVENT_TYPES

_CLOCK = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_NORMALIZE_LOCATION = re.compile(r"[^\w؀-ۿ]+", re.UNICODE)

# Events whose real-world truth changes over time are the ones where two
# reports differing in status may be describing different moments rather than
# contradicting each other. We still flag them, but at lower severity and with
# the timestamps shown, because "closed now, open later" is not a contradiction.
VOLATILE_TYPES = {"road_blocked", "road_open", "power_outage", "water_outage",
                  "fire", "flood"}


def canon_location(value: str | None) -> str:
    if not value:
        return ""
    import unicodedata
    t = unicodedata.normalize("NFKC", value).lower()
    return _NORMALIZE_LOCATION.sub(" ", t).strip()


def times_in(text: str | None) -> set[str]:
    if not text:
        return set()
    return {f"{int(h):02d}:{m}" for h, m in _CLOCK.findall(text)}


def _minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


class Contradiction:
    """A detected disagreement between two events. Never self-resolving."""

    __slots__ = ("event_a", "event_b", "conflict_type", "severity",
                 "explanation", "evidence", "rule")

    def __init__(self, event_a, event_b, conflict_type, severity, explanation,
                 evidence, rule):
        self.event_a = event_a
        self.event_b = event_b
        self.conflict_type = conflict_type
        self.severity = severity
        self.explanation = explanation
        self.evidence = evidence
        self.rule = rule

    def __repr__(self) -> str:
        return (f"<Contradiction {self.conflict_type} {self.event_a.event_id} "
                f"vs {self.event_b.event_id} sev={self.severity}>")


def _same_place(a, b) -> bool:
    """Whether two events are about the same place.

    Prefers the resolved (cross-language) location when entity resolution has run,
    and falls back to canonicalised raw text. The fallback matters: with no
    resolver configured, a French and an Arabic report about one bridge stay
    separate, which loses a contradiction but never invents one.
    """
    ra = getattr(a, "resolved_location", None)
    rb = getattr(b, "resolved_location", None)
    if ra and rb:
        return canon_location(ra) == canon_location(rb)
    loc_a, loc_b = canon_location(a.location_text), canon_location(b.location_text)
    return bool(loc_a) and loc_a == loc_b


def detect_pair(a, b) -> Contradiction | None:
    """Compare two events for a contradiction. Pure function of stored fields."""
    same_loc = _same_place(a, b)
    ev = sorted({a.report_id, b.report_id})

    # --- status ------------------------------------------------------------
    st = get_settings()
    pair = (a.event_type, b.event_type)
    if same_loc and pair in st.conflict_pairs:
        # If one report says it was blocked and a later one says it is open,
        # that may be a genuine state change rather than a disagreement.
        volatile = a.event_type in VOLATILE_TYPES and b.event_type in VOLATILE_TYPES
        if volatile and _is_later_change(a, b):
            return Contradiction(
                a, b, "status", "low",
                f"Reports describe '{a.location_text}' as {a.event_type} at "
                f"{a.time_reference or 'an earlier time'} and as {b.event_type} at "
                f"{b.time_reference or 'a later time'}. This may be a genuine "
                f"state change rather than a contradiction. Requires human "
                f"verification.",
                ev, "status_pair_with_temporal_order",
            )
        return Contradiction(
            a, b, "status", "high" if same_loc else "medium",
            f"Two independent reports disagree about whether '{a.location_text}' "
            f"is passable: one reports {a.event_type}, the other reports "
            f"{b.event_type}. Both are shown; neither has been preferred.",
            ev, "status_pair_same_location",
        )

    # --- numeric -----------------------------------------------------------
    if (same_loc and a.event_type == b.event_type
            and a.event_type in COUNTABLE_EVENT_TYPES
            and a.people_affected is not None and b.people_affected is not None
            and a.people_affected != b.people_affected):
        delta = abs(a.people_affected - b.people_affected)
        return Contradiction(
            a, b, "numeric", "high" if delta > 2 else "medium",
            f"Reports disagree on the number of people affected at "
            f"'{a.location_text}': {a.people_affected} versus "
            f"{b.people_affected} for the same {a.event_type}. CrisisMesh does "
            f"not average, pick the larger, or pick the more recent value.",
            ev, "count_mismatch_same_event",
        )

    # --- temporal ----------------------------------------------------------
    if (same_loc and a.event_type == b.event_type
            and a.time_reference and b.time_reference):
        ta, tb = times_in(a.time_reference), times_in(b.time_reference)
        if ta and tb and ta != tb:
            return Contradiction(
                a, b, "temporal", "medium",
                f"Reports place the same {a.event_type} at '{a.location_text}' at "
                f"different times: {'/'.join(sorted(ta))} versus "
                f"{'/'.join(sorted(tb))}. Both timestamps are retained.",
                ev, "clock_time_mismatch",
            )
    return None


def _is_later_change(a, b) -> bool:
    """True when b is unambiguously reported after a, using absolute clocks."""
    ta = times_in(a.time_reference)
    tb = times_in(b.time_reference)
    if not ta or not tb:
        return False
    if not a.time_reference or not b.time_reference:
        return False
    return _minutes(sorted(tb)[0]) > _minutes(sorted(ta)[0])


def detect_all(events: list) -> list[Contradiction]:
    """Find every contradicting pair. O(n^2) is fine at report volumes and
    keeps the logic auditable; an index can come later with evidence it helps."""
    found: list[Contradiction] = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            c = detect_pair(events[i], events[j])
            if c is not None:
                found.append(c)
    return found


def default_explanation(c: Contradiction) -> str:
    """Fallback rationale when Apertus is unavailable.

    We still surface the conflict. Losing the model's one-sentence explanation
    is acceptable; silently dropping a detected contradiction is not.
    """
    return f"[detected by rule: {c.rule}] {c.explanation}"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()