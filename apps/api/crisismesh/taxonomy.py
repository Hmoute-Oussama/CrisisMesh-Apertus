"""Controlled vocabularies.

These are the taxonomy and the state machines. Two rules govern them:

1.  The GBNF grammar is generated from EVENT_TYPES and SEVERITIES, so the model
    physically cannot emit an event type outside the taxonomy. Taxonomy drift
    is a grammar rebuild, not a post-hoc filter.
2.  Verification and conflict transitions are allow-listed here rather than
    checked ad hoc, so the state machine is auditable in one place.
"""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    INJURY = "injury"
    MISSING_PERSON = "missing_person"
    TRAPPED_PERSON = "trapped_person"
    ROAD_BLOCKED = "road_blocked"
    ROAD_OPEN = "road_open"
    FIRE = "fire"
    FLOOD = "flood"
    STRUCTURAL_DAMAGE = "structural_damage"
    POWER_OUTAGE = "power_outage"
    WATER_OUTAGE = "water_outage"
    SHELTER_CAPACITY = "shelter_capacity"
    MEDICAL_NEED = "medical_need"
    FOOD_NEED = "food_need"
    EVACUATION = "evacuation"
    INFRASTRUCTURE_DAMAGE = "infrastructure_damage"
    HAZARD = "hazard"
    RESOURCE_AVAILABLE = "resource_available"
    RESOURCE_NEEDED = "resource_needed"
    OTHER = "other"


class Severity(StrEnum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerificationState(StrEnum):
    UNVERIFIED = "unverified"
    CORROBORATED = "corroborated"
    DISPUTED = "disputed"
    VERIFIED_BY_OPERATOR = "verified_by_operator"
    REJECTED = "rejected"


class ConflictType(StrEnum):
    STATUS = "status"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    LOCATION = "location"


class ConflictStatus(StrEnum):
    UNRESOLVED = "unresolved"
    RESOLVED_BY_OPERATOR = "resolved_by_operator"
    DISMISSED = "dismissed"


class SourceType(StrEnum):
    HUMAN_REPORT = "human_report"
    BATCH = "batch"
    DEMO = "demo"
    AUDIO_TRANSCRIPT = "audio_transcript"


class Language(StrEnum):
    FR = "fr"
    AR = "ar"
    MSA = "msa"
    DARIJA = "dar"
    EN = "en"
    UNKNOWN = "und"


# ---- ordering helpers -------------------------------------------------------

SEVERITY_RANK: dict[str, int] = {
    Severity.UNKNOWN: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

# Event types that assert a condition about a route or path. Used to detect
# status contradictions without understanding any natural language.
STATUS_EVENT_TYPES: frozenset[str] = frozenset(
    {EventType.ROAD_BLOCKED, EventType.ROAD_OPEN}
)

# Event types that assert a condition about a *named place*, and so are the ones
# worth spending model calls on for cross-language entity resolution. Types like
# missing_person also carry a location, but a person is not resolvable across
# languages, so including them would cost calls for no gain.
PLACE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EventType.ROAD_BLOCKED,
        EventType.ROAD_OPEN,
        EventType.STRUCTURAL_DAMAGE,
        EventType.POWER_OUTAGE,
        EventType.WATER_OUTAGE,
        EventType.SHELTER_CAPACITY,
        EventType.FOOD_NEED,
        EventType.MEDICAL_NEED,
        EventType.EVACUATION,
        EventType.INFRASTRUCTURE_DAMAGE,
        EventType.HAZARD,
        EventType.RESOURCE_AVAILABLE,
        EventType.RESOURCE_NEEDED,
    }
)

# Event types where a numeric people_affected disagreement is meaningful.
COUNTABLE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EventType.INJURY,
        EventType.MISSING_PERSON,
        EventType.TRAPPED_PERSON,
        EventType.MEDICAL_NEED,
        EventType.FOOD_NEED,
        EventType.SHELTER_CAPACITY,
        EventType.EVACUATION,
    }
)

# Only explicit lexical markers license a non-unknown severity. Anything else
# stays UNKNOWN. This is the abstention policy enforced after extraction.
SEVERITY_MARKERS: dict[str, frozenset[str]] = {
    Severity.CRITICAL: frozenset(
        {
            "critical", "critique", "critique!", "life-threatening",
            "life threatening", "gravementre", "mortal", "grave danger",
        }
    ),
    Severity.HIGH: frozenset(
        {"grave", "gravementre", "serieux", "serieuse", "sévère", "severe",
         "majeur", "danger", "dangerous", "قاسح", "خطير"}
    ),
    Severity.MEDIUM: frozenset(
        {"modere", "modéré", "important", "urgent", "عاجل", "مهم"}
    ),
    Severity.LOW: frozenset({"leger", "léger", "minor", "خفيفة"}),
}


def severity_from_markers(text: str) -> Severity:
    """Derive severity only from explicit lexical markers in the source text.

    Returns Severity.UNKNOWN when nothing explicitly states seriousness. This is
    deliberately conservative: we would rather report unknown than guess.
    """
    low = (text or "").lower()
    for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
        if SEVERITY_MARKERS[sev] & set(_words(low)):
            return sev
    return Severity.UNKNOWN


def _words(text: str) -> list[str]:
    out, cur = [], []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def verification_allowed(
    current: VerificationState, target: VerificationState
) -> bool:
    """Explicit state machine for human verification.

    An operator may verify or reject anything unverified, and may resolve or
    dismiss a dispute. They may not silently rewrite a rejected event back to
    unverified without an audit trail, and may not promote a claim directly to
    verified_by_operator from an event they have not seen.
    """
    allowed: dict[VerificationState, frozenset[VerificationState]] = {
        VerificationState.UNVERIFIED: frozenset(
            {
                VerificationState.CORROBORATED,
                VerificationState.DISPUTED,
                VerificationState.VERIFIED_BY_OPERATOR,
                VerificationState.REJECTED,
            }
        ),
        VerificationState.CORROBORATED: frozenset(
            {
                VerificationState.DISPUTED,
                VerificationState.VERIFIED_BY_OPERATOR,
                VerificationState.REJECTED,
            }
        ),
        VerificationState.DISPUTED: frozenset(
            {
                VerificationState.VERIFIED_BY_OPERATOR,
                VerificationState.REJECTED,
                VerificationState.CORROBORATED,
            }
        ),
        VerificationState.VERIFIED_BY_OPERATOR: frozenset(
            {VerificationState.DISPUTED, VerificationState.REJECTED}
        ),
        VerificationState.REJECTED: frozenset(
            {VerificationState.UNVERIFIED, VerificationState.VERIFIED_BY_OPERATOR}
        ),
    }
    return target in allowed.get(current, frozenset())