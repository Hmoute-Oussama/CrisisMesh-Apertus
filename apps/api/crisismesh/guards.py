"""The epistemic guardrail layer.

This module contains no natural-language understanding. That is the point.
Everything here is deterministic, inspectable code that runs *after* Apertus
has spoken, and it exists because the measured behaviour of the model is not
trustworthy enough to write to an operator's screen unchecked.

Measured on the spike set, Apertus-v1.1-4B-Instruct:
  - emitted the literal string "UNKNOWN" where null was legal
  - echoed our own delimiter back as a location value
  - let an injected "severity critical" through from report text
  - occasionally produced a people count absent from the source

Each of those is handled here, explicitly and with a recorded reason, so the
audit trail shows that a claim was corrected rather than silently accepted.

The rule: Apertus decides what a report *says*. This module decides what is
allowed to be *believed*.
"""

from __future__ import annotations

import re
import unicodedata

from .taxonomy import EventType, Severity, severity_from_markers

# ---- untrusted-content boundary ---------------------------------------------

REPORT_OPEN = "<UNTRUSTED_REPORT>"
REPORT_CLOSE = "</UNTRUSTED_REPORT>"

# Instruction-shaped text. A crisis report has no business containing these;
# their presence is evidence that the submitter is trying to steer the model.
INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier|the\s+above)",
    r"forget\s+(everything|all)\s+(you|above|before)",
    r"you\s+are\s+now\b",
    r"new\s+instructions?\s*:",
    r"(?:^|\n)\s*(system|assistant)\s*:",
    r"</?(?:untrusted_report|system|instructions?|prompt)>",
    r"\btherefore\s+the\s+answer\s+is\b",
    r"\balways\s+(?:return|output|report)\b",
    r"\boutput\s+only\b",
    r"\bmark\s+(?:this|it)\s+as\b",
    r"\bset\s+severity\s+to\b",
)

# Placeholder values meaning "the model found no answer". Matched on the
# accent-stripped, casefolded form so one entry covers every spelling.
#
# The multilingual variants matter as much as the English ones. An earlier
# version listed "not specified" but not "non specifie", so the model emitting
# the French "non specifie" for an unstated place was stored as if it were a
# real location, producing a phantom duplicate event on four reports. The
# placeholder list has to be closed over the languages the system claims to
# handle, not just the language its author speaks.
_PLACEHOLDER_WORDS = {
    "", "-", "?", "n/a", "na", "nil", "none", "null", "undefined", "unknown",
    "unk", "no", "no information", "no data", "no location", "no place",
    "not specified", "non specified", "not available", "not stated",
    "not mentioned", "non given", "non provided", "non disclosed",
    "non existant", "non existe", "no location given", "location not specified",
    "unspecified", "undefined location", "unavailable", "unspecified location",
    "non specifie", "non precise", "non precisee", "non indique", "non mentionne",
    "non renseigne", "pas d'information", "inconnu", "indetermine",
    "non specifiee", "pas precise", "pas indique", "non defini",
    "غير محدد", "غير معروف", "غير مذكور", "لا يوجد", "لا معلومة",
}

_NULLISH = {
    "", "null", "none", "n/a", "na", "unknown", "unspecified", "not specified",
    "undefined", "nil", "-", "?", "no", "UNK", "UNKNOWN", "Unknown",
}


def _fold_placeholder(value: str) -> str:
    """Accent-stripped, casefolded, whitespace-collapsed form for comparison."""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().split())


_PLACEHOLDER_FOLDED = {_fold_placeholder(p) for p in _PLACEHOLDER_WORDS}

# Values the model emits when it gets confused about its own task. These are
# guardrail rejections, not abstentions, and they are recorded as such.
_DELIMITER_ECHOES = {
    "UNTRUSTED_REPORT", "untrusted_report", "REPORT", "EVENT", "ANSWER",
    "events:", "segment", "SEGMENT", "type", "location", "severity",
    "<UNTRUSTED_REPORT>", "</UNTRUSTED_REPORT>",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟。；])\s+|\n+")
_NUM_RE = re.compile(r"\d+")


# ---- injection defense ------------------------------------------------------

def detect_injection(text: str) -> list[str]:
    """Return the names of instruction-shaped patterns found in report text.

    This runs on raw text, not on a sanitized view: the point is to record what
    the submitter actually wrote.
    """
    return [p for p in INJECTION_PATTERNS if re.search(p, text or "", re.IGNORECASE)]


def model_view(text: str) -> str:
    """Return the text handed to Apertus.

    Instruction-shaped lines are dropped. The original is untouched on disk and
    is always what the operator sees; this view exists only so that an attempt
    to steer the model does not reach it. If every line looks like an
    instruction, we fall back to the original and let the flag stand, because
    silently feeding the model an empty string would be worse than feeding it
    the flagged text alongside a verification_state that forbids belief.
    """
    kept = [
        seg for seg in (s.strip() for s in _SENTENCE_SPLIT.split(text or ""))
        if seg and not detect_injection(seg)
    ]
    return " ".join(kept) if kept else (text or "").strip()


# ---- segmentation -----------------------------------------------------------

def segment(text: str, min_len: int = 8) -> list[str]:
    """Split a report into incident-bearing segments.

    Deterministic on purpose. Rather than asking a 4B model to emit several
    events at once and hoping it does, we hand it one segment at a time. This
    lifted multi-event extraction from 0/4 to 4/4 on the spike set while being
    faster than escalating to the 8B model.
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text or "") if p and p.strip()]
    out = [p for p in parts if len(p) >= min_len]
    return out or ([text.strip()] if (text or "").strip() else [])


# ---- output normalisation ---------------------------------------------------

def _clean_str(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = unicodedata.normalize("NFC", value).strip()
    if v in _NULLISH or v.lower() in _NULLISH:
        return None
    # Folded placeholder check, so "Non Specifié", "NON SPECIFIE" and
    # "non  specifie" are all recognised as abstentions.
    if _fold_placeholder(v) in _PLACEHOLDER_FOLDED:
        return None
    return v or None


def _clean_int(value, source_text: str) -> tuple[int | None, list[str]]:
    """Keep an integer only if it actually appears in the source text.

    This is the single highest-value hallucination guard. A people count that
    is not in the report is fabricated by definition, so it is discarded and
    the event records that it happened.
    """
    guards: list[str] = []
    if value is None or value == "null":
        return None, guards
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None, ["people_unparseable->null"]
    if n <= 0:
        return None, ["people_nonpositive->null"]
    # Arabic-Indic and Eastern Arabic digits are normalized by NFKC above only
    # for str inputs, so also check the raw source numerals.
    src_nums = set(_NUM_RE.findall(source_text or ""))
    if str(n) not in src_nums:
        return None, ["people_not_in_source->null"]
    return n, guards


def normalize_extraction(
    event_type: str | None,
    location_text: str | None,
    people_affected,
    severity: str | None,
    time_reference: str | None,
    description: str | None,
    source_segment: str,
    suspected_injection: bool = False,
) -> dict:
    """Turn one raw extracted event into a guarded, persistable claim.

    Returns a dict with the cleaned fields plus ``guards_applied``: the list of
    corrections made, which is stored on the event and shown in the UI.
    """
    guards: list[str] = []

    etype = _clean_str(event_type)
    if etype not in {e.value for e in EventType}:
        if etype is not None:
            guards.append(f"event_type_out_of_taxonomy({etype})->other")
        etype = EventType.OTHER.value

    loc = _clean_str(location_text)
    if loc and (loc in _DELIMITER_ECHOES or REPORT_OPEN.strip("<>") in loc
                or loc.lower() in _DELIMITER_ECHOES):
        guards.append("location_delimiter_echo->null")
        loc = None

    ppl, g = _clean_int(people_affected, source_segment)
    guards += g

    tref = _clean_str(time_reference)
    if tref and REPORT_OPEN.strip("<>") in tref:
        guards.append("time_delimiter_echo->null")
        tref = None

    desc = _clean_str(description)
    if desc and REPORT_OPEN.strip("<>") in desc:
        guards.append("desc_delimiter_echo->null")
        desc = None

    # Abstention policy: severity may only survive if the source text carries an
    # explicit marker. A model asserting "high" on its own is not evidence.
    sev = _clean_str(severity)
    licensed = severity_from_markers(source_segment)
    if suspected_injection:
        # An attempt to steer the model must never be able to raise severity.
        guards.append("injection_flagged->severity_forced_unknown")
        sev_final = Severity.UNKNOWN.value
    elif licensed is Severity.UNKNOWN:
        if sev and sev != Severity.UNKNOWN.value:
            guards.append(f"severity_unlicensed({sev})->unknown")
        sev_final = Severity.UNKNOWN.value
    else:
        # Source states a marker; keep the model's reading but never above the
        # marker's own ceiling.
        if sev in {s.value for s in Severity} and sev != Severity.UNKNOWN.value:
            from .taxonomy import SEVERITY_RANK
            sev_final = (sev if SEVERITY_RANK[sev] <= SEVERITY_RANK[licensed.value]
                         else licensed.value)
        else:
            sev_final = licensed.value

    return {
        "event_type": etype,
        "location_text": loc,
        "people_affected": ppl,
        "severity": sev_final,
        "time_reference": tref,
        "description": desc,
        "guards_applied": guards,
    }


def uncertainty_notes(cleaned: dict, source_segment: str) -> list[str]:
    """Explicitly record what we do not know about this event."""
    notes: list[str] = []
    if cleaned["people_affected"] is None:
        notes.append("no people count stated in report")
    if cleaned["severity"] == Severity.UNKNOWN.value:
        notes.append("severity not stated in report")
    if not cleaned["location_text"]:
        notes.append("no location stated in report")
    if not cleaned["time_reference"]:
        notes.append("no time reference stated in report")
    return notes