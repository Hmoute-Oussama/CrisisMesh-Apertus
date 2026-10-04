"""Duplicate detection and corroboration counting.

The distinction this module exists to enforce:

    ten forwarded copies of one message are ONE source.

If a WhatsApp group forwards a single incident report ten times and we count
that as ten independent confirmations, we manufacture certainty out of a
single witness. That is the most common way naive crisis dashboards lie.

Method, in tiers, cheapest first:
  1. exact hash of the normalized text                  -> certain duplicate
  2. character-trigram Jaccard similarity              -> candidate
  3. Apertus adjudication of cross-language candidates -> final call

No embeddings and no vector database. At CrisisMesh's report volumes, lexical
similarity plus a model call on the surviving candidates is both cheaper and
far easier to explain to a reviewer than an embedding index.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from ..config import get_settings
from ..guards import detect_injection, segment

_PUNCT = re.compile(r"[^\w؀-ۿ]+", re.UNICODE)
_WS = re.compile(r"\s+")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize(text: str) -> str:
    """Fold a report to a comparison form. Never replaces the stored original."""
    t = unicodedata.normalize("NFKC", text or "").translate(_AR_DIGITS).lower()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def text_fingerprint(text: str) -> str:
    """Order-insensitive fingerprint of the report's segments.

    Two reports with the same sentences in a different order are the same
    report, so we hash the sorted segment set rather than the raw string.
    """
    segs = sorted(normalize(s) for s in segment(text) if s.strip())
    return hashlib.sha256(" || ".join(segs).encode("utf-8")).hexdigest()


def trigrams(text: str) -> set[str]:
    t = normalize(text).replace(" ", "_")
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def similarity(a: str, b: str) -> float:
    return jaccard(trigrams(a), trigrams(b))


def classify(a_text: str, b_text: str) -> tuple[bool, float, str]:
    """Return (is_duplicate, similarity, method).

    Same reporter matters: one person reporting the same thing twice is not two
    independent sources, so the caller uses this together with reporter_id.
    """
    fa, fb = text_fingerprint(a_text), text_fingerprint(b_text)
    if fa == fb:
        return True, 1.0, "exact_fingerprint"
    s = similarity(a_text, b_text)
    st = get_settings()
    if s >= st.dedup_strong_threshold:
        return True, s, "trigram_strong"
    if s >= st.dedup_review_threshold:
        # Similar but not certain. We do not merge automatically: a near-match
        # during an earthquake is often two different injuries on one street.
        return False, s, "trigram_review"
    return False, s, "trigram_weak"


def independence_key(report) -> str:
    """Identity used for counting independent sources.

    Same reporter_id, or same pseudonym plus same text, counts once. Anonymous
    reports are distinguished by their fingerprint.
    """
    if report.reporter_id:
        return f"reporter:{report.reporter_id}"
    return f"anon:{report.text_hash[:16]}"


def corroboration_weight(report) -> float:
    """How much a single report contributes to evidence confidence.

    A flag-anonymous or injection-flagged report is worth less than a named,
    non-flagged one. It still counts: discarding it would be its own distortion.
    """
    w = 1.0
    if not report.reporter_id:
        w *= 0.7
    if report.suspected_injection or detect_injection(report.raw_text):
        w *= 0.3
    return w


def evidence_confidence(
    independent_groups: int,
    distinct_languages: int = 1,
    base_weight: float = 1.0,
) -> float:
    """Evidence strength from independent corroboration.

    Saturating rather than linear: a third independent witness adds real
    information, a thirtieth adds almost none. Cross-language corroboration is
    weighted higher because agreeing in two languages rules out a single
    mistranslation or transcription error, which is a genuine independence
    signal that two reports in one language cannot provide.

    This is deliberately NOT the model's self-reported confidence. It is a
    function of evidence we can count.
    """
    if independent_groups <= 0:
        return 0.0
    from math import log1p

    n = log1p(independent_groups) / log1p(4)      # 1 group -> 0.5
    n = min(1.0, n * 1.6)
    lang_bonus = 1.0 + 0.08 * (distinct_languages - 1)
    return round(min(0.97, n * lang_bonus * base_weight), 3)