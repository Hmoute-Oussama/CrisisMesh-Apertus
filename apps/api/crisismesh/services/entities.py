"""Cross-language entity resolution.

Why this module exists
----------------------
The obvious implementation of contradiction detection is "two reports that
disagree about the same place". Measured on the Al-Nour scenario, that
implementation finds zero contradictions, and it will keep finding zero,
because the two reports are rarely in the same language:

    SC-008 (fr)  "Le pont central est toujours bloque"   -> 'pont central'
    SC-009 (dar) "الطريق محلولة دابا"                    -> 'الطريق'

These are the same bridge. They share no characters, no tokens and no
transliteration of each other, so string equality, fuzzy matching and
trigram similarity all correctly report "no match" and the contradiction is
invisible. A multilingual crisis system whose core feature is spotting
disagreement cannot fail on the most common case there is.

So identity resolution is a semantic task and belongs to Apertus. What belongs
to Python is the part that must not be delegated: deciding whether two
*already-identified-same-place* claims contradict each other, and ever acting
on that conclusion.

The split, stated plainly:

    Apertus decides  "are these two place names the same place?"
    Python decides   "given that they are the same place, do these claims
                      contradict each other, and what do we show the operator?"

The model is never asked which report is true. It only ever answers an identity
question, and a refusal to answer costs us a conflict but never produces a
false accusation of contradiction.

Determinism note: resolution is cached per (normalised mention, canonical set)
in memory and persisted on the Event as ``resolved_location`` and
``location_confidence``. Re-running the same corpus produces the same entities,
because the prompt is versioned and temperature is 0.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from ..taxonomy import PLACE_EVENT_TYPES

# Arabic definite article and Moroccan Darija glue. Stripped before comparison
# so that "الجسر المركزي" and "الجسر" are not treated as unrelated strings, and
# so that a model's redundant article does not split one place into two.
_ARTICLE_PREFIX = re.compile(r"^(?:ال|أل|الـ)\s*")
_NORMALIZE = re.compile(r"[^\w؀-ۿ]+", re.UNICODE)

# Very common generic words that appear as a "location" only because the model
# reached for one. Treating them as place identities would create false links
# between unrelated reports.
_GENERIC_PLACES = {
    "the road", "road", "street", "here", "there", "city", "town", "area",
    "place", "الطريق", "الشارع", "هنا", "هناك", "المدينة", "المنطقة", "بلدة",
    "طريق", "شارع",
}


def normalize_place(value: str | None) -> str:
    """Fold a place mention to a comparison key.

    NFKC normalisation first so Arabic presentation forms and Latin accents
    compare equal, then casefold, then strip punctuation and articles.
    """
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = _ARTICLE_PREFIX.sub("", text)
    return _NORMALIZE.sub(" ", text).strip()


def is_generic(mention: str | None) -> bool:
    """True when a location string is too generic to identify a place."""
    key = normalize_place(mention)
    if not key:
        return True
    return key in _GENERIC_PLACES


# Words too common to be evidence that two names describe one place. 'hospital'
# is not evidence that Al-Amal hospital is the same as any other hospital.
_STOPWORDS = {
    "the", "of", "de", "du", "des", "la", "le", "les", "el", "al", "sidi",
    "city", "town", "ville", "centre", "center", "north", "south", "east",
    "west", "old", "new", "grande", "petite", "saint",
    "al", "ad", "as", "of",
}

# Curated cross-language aliases for the demo scenario's places. This is a
# gazetteer, not a general solution: it encodes that "hopital Al-Amal" and
# "المستشفى" name one facility. It exists so the demo has correct
# cross-language grouping even with a weak resolver, and it is honest about what
# it is. The model path handles places not listed here; these pairs are simply
# trusted without asking.
KNOWN_ALIASES: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"hopital al amal", "al amal hospital", "المستشفى",
                   "المستشفى الأمل"}),
        frozenset({"stade omnisports", "stade", "ملعب", "الملعب الرياضي",
                   "stadium", "sports stadium"}),
        frozenset({"pont central", "central bridge", "الجسر المركزي"}),
        frozenset({"vieille medina", "old medina", "المدينة القديمة",
                   "medina"}),
        frozenset({"rond point nord", "rond-point nord", "north roundabout",
                   "الدوار الشمالي"}),
        frozenset({"zone industrielle", "industrial zone", "المنطقة الصناعية"}),
    }
)

# Alias groups are stored through normalize_place so that article stripping and
# punctuation folding apply to the gazetteer too. Without this, "المستشفى" in
# the table and "مستشفى" from a report never meet.
_ALIAS_INDEX: frozenset[frozenset[str]] = frozenset(
    frozenset(normalize_place(name) for name in group)
    for group in KNOWN_ALIASES
)


def _tokens(value: str) -> set[str]:
    return {w for w in normalize_place(value).split()
            if w and w not in _STOPWORDS and len(w) > 1}


def _shares_content_word(a: str, b: str) -> bool:
    return bool(_tokens(a) & _tokens(b))


def _known_alias(a: str, b: str) -> bool:
    """True when both names belong to the same curated alias group.

    Compares group membership rather than set equality, because a group holds
    every known rendering of a place and a pair is only ever two of them.
    """
    na, nb = normalize_place(a), normalize_place(b)
    if not na or not nb:
        return False
    return any(na in group and nb in group for group in _ALIAS_INDEX)


@dataclass(frozen=True)
class Resolution:

    canonical: str
    same: bool | None          # None means "the model declined to answer"
    confidence: float
    reason: str


class LocationResolver:
    """Groups place mentions into canonical places using Apertus.

    The comparison is done pairwise against existing canonical names rather than
    by asking the model to invent an identifier. Asking for an identifier invites
    the model to return a new string for an existing place every time, which
    fragments the graph; asking "is this the same place as X?" keeps the answer
    space small and checkable.
    """

    def __init__(self, client, enabled: bool = True) -> None:
        self.client = client
        self.enabled = enabled
        self._cache: dict[str, Resolution] = {}

    # ---- cache key -------------------------------------------------------
    @staticmethod
    def _key(mention: str, candidates: tuple[str, ...]) -> str:
        h = hashlib.sha256()
        h.update(mention.encode("utf-8"))
        for c in candidates:
            h.update(b"\x00")
            h.update(c.encode("utf-8"))
        return h.hexdigest()[:16]

    # ---- public API ------------------------------------------------------
    def resolve(self, mention: str | None,
                candidates: list[str]) -> Resolution:
        """Place ``mention`` against ``candidates``.

        Returns a Resolution whose canonical value is either an existing
        candidate or the mention itself. When the model declines, the mention is
        left on its own rather than force-linked: an unresolved place is a
        visible gap, a wrong merge is a silent corruption.
        """
        key_mention = normalize_place(mention)
        if not key_mention or is_generic(mention):
            return Resolution(canonical=mention or "", same=None, confidence=0.0,
                              reason="generic_or_empty_mention")

        if not self.enabled:
            return Resolution(canonical=mention, same=None, confidence=0.0,
                              reason="resolver_disabled")

        live = [c for c in candidates if c and not is_generic(c)]
        cache_key = self._key(key_mention, tuple(sorted(live)))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # First mention of a place establishes the canonical name.
        if not live:
            res = Resolution(canonical=mention, same=None, confidence=1.0,
                             reason="first_mention")
            self._cache[cache_key] = res
            return res

        # Exact normalized identity against an existing candidate, decided
        # before the model is consulted at all. "Pont  Central" and "pont
        # central" are the same string once folded; asking a 4B model to agree
        # with arithmetic only creates a chance to disagree with it. The model
        # supplies spellings for the hard cases, never an identity verdict.
        exact = next((c for c in live if normalize_place(c) == key_mention), None)
        if exact is not None:
            out = Resolution(canonical=exact, same=True, confidence=1.0,
                             reason="exact_normalized_match")
            self._cache[cache_key] = out
            return out

        res = self.client.resolve_place(mention, live)
        if res is None:
            # Unreachable model is not a licence to guess. Keep the place
            # separate and record that we could not check.
            out = Resolution(canonical=mention, same=None, confidence=0.0,
                             reason="model_unavailable")
            self._cache[cache_key] = out
            return out

        # The identity decision is made here, in Python, by comparing the
        # model's canonical spelling against the known names. The model supplies
        # a spelling; it never decides that two reports are the same place.
        target = normalize_place(res.canonical)
        match = next((c for c in live if normalize_place(c) == target), None)

        # ---- the safety gate that matters --------------------------------
        # A grammar-constrained 4B decode of this task has a strong bias toward
        # echoing a candidate. Measured on the Al-Nour corpus it merged
        # 'vieille medina', 'hopital Al-Amal' and 'الجامعة' into 'pont
        # central', which would have manufactured contradictions nobody
        # reported. Two cheap deterministic guards suppress most of that
        # without needing a confidence score from the model:
        #
        #   1. identical after normalization -> merge. No model needed, this is
        #      exact identity.
        #   2. otherwise the mention must share a content word with the match,
        #      or the two must be known cross-language aliases of one another.
        #
# Rule 2 deliberately fails closed: an unmergeable pair stays separate.
        # A missing contradiction is a gap the operator can see; a fabricated
        # one is a lie with a confidence number attached to it.
        if match is None:
            # The model proposed a spelling that is not among the places we
            # already know. That is a new place, not a merge.
            out = Resolution(canonical=res.canonical or mention, same=False,
                             confidence=0.0, reason="apertus_name_is_new")
        elif _shares_content_word(mention, match) or _known_alias(mention, match):
            out = Resolution(canonical=match, same=True, confidence=0.7,
                             reason="apertus_name_with_lexical_support")
        else:
            out = Resolution(canonical=mention, same=False, confidence=0.0,
                             reason="model_proposed_merge_without_lexical_support")
        self._cache[cache_key] = out
        return out

    def canonical_set(self, mentions: list[str]) -> dict[str, str]:
        """Map every mention to its canonical name.

        Deliberately order-dependent and greedy: the first mention of a place
        becomes its canonical name, and later mentions are compared against the
        canonicals accumulated so far. At crisis-report volumes this is a few
        dozen comparisons, which is cheaper than a full clustering pass and is
        far easier to explain to an operator who asks why two reports were
        linked.

        The mapping is only used to group events for corroboration and
        contradiction. The original wording is never overwritten, so a
        disagreement about identity stays visible in the evidence panel.
        """
        canonicals: list[str] = []
        mapping: dict[str, str] = {}
        for mention in mentions:
            if not mention or is_generic(mention):
                mapping[mention or ""] = mention or ""
                continue
            res = self.resolve(mention, canonicals)
            canon = res.canonical or mention
            if not (res.same and canon in canonicals) and canon not in canonicals:
                canonicals.append(canon)
            mapping[mention] = canon
        return mapping


def merge_for_conflict(events) -> dict[str, str]:
    """Assign a comparison key per event id, using resolved_location if present.

    Contradiction detection calls this instead of comparing raw location text,
    so a French and an Arabic report about the same bridge land on the same key
    and the status rule can fire.
    """
    keys: dict[str, str] = {}
    for ev in events:
        base = ev.resolved_location or ev.location_text
        keys[ev.event_id] = normalize_place(base)
    return keys


def is_place_event(event_type: str) -> bool:
    return event_type in PLACE_EVENT_TYPES


__all__ = ["LocationResolver", "Resolution", "normalize_place", "is_generic",
           "merge_for_conflict", "is_place_event"]