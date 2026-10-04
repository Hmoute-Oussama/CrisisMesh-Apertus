"""Tests for contradiction detection and entity-resolution safety.

    .venv/Scripts/python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apps" / "api"))

from crisismesh.services import contradiction as C  # noqa: E402
from crisismesh.services.entities import (  # noqa: E402
    LocationResolver,
    is_generic,
    normalize_place,
)


class FakeEvent:
    """Minimal stand-in with the attributes the detectors actually read."""

    def __init__(self, event_id, event_type, location, people=None,
                 time_reference=None, resolved_location=None,
                 verification_state="unverified"):
        self.event_id = event_id
        self.report_id = f"R-{event_id}"
        self.event_type = event_type
        self.location_text = location
        self.people_affected = people
        self.time_reference = time_reference
        self.resolved_location = resolved_location
        self.verification_state = verification_state
        self.independent_source_count = 1


class StubClient:
    """Stands in for Apertus. Records what it was asked."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def resolve_place(self, mention, candidates):
        self.calls.append((mention, list(candidates)))
        if self.answer is None:
            return None
        return self.answer


def Res(same, canonical, confidence=0.9):  # noqa: N802 - test shorthand
    class R:
        pass
    r = R()
    r.same = same
    r.canonical = canonical
    r.confidence = confidence
    return r


# ---- status contradictions ----------------------------------------------

def test_same_place_opposing_status_is_a_conflict():
    found = C.detect_all([
        FakeEvent("e1", "road_blocked", "pont central"),
        FakeEvent("e2", "road_open", "pont central"),
    ])
    assert found
    assert found[0].conflict_type == "status"


def test_same_type_at_same_place_is_not_a_conflict():
    """Two reports agreeing is not a contradiction."""
    assert not C.detect_all([
        FakeEvent("e1", "road_blocked", "pont central"),
        FakeEvent("e2", "road_blocked", "pont central"),
    ])


def test_opposing_status_at_different_places_is_not_a_conflict():
    """The bridge may be shut while the stadium is open."""
    assert not C.detect_all([
        FakeEvent("e1", "road_blocked", "pont central"),
        FakeEvent("e2", "road_open", "stade"),
    ])


def test_conflict_prefers_resolved_place_over_raw_text():
    """Entity resolution is what lets a cross-language pair be compared.

    "الجسر المركزي" and "pont central" share no characters. Without the
    canonical key the two claims cannot be joined and the contradiction is
    missed entirely.
    """
    found = C.detect_all([
        FakeEvent("e1", "road_blocked", "الجسر المركزي",
                  resolved_location="pont central"),
        FakeEvent("e2", "road_open", "pont central",
                  resolved_location="pont central"),
    ])
    assert found
    assert found[0].conflict_type == "status"


# ---- numeric contradictions ---------------------------------------------

def test_differing_counts_are_a_numeric_conflict():
    found = C.detect_all([
        FakeEvent("e1", "injury", "hopital", people=4),
        FakeEvent("e2", "injury", "hopital", people=2),
    ])
    assert found
    assert found[0].conflict_type == "numeric"


def test_matching_counts_are_not_a_conflict():
    assert not C.detect_all([
        FakeEvent("e1", "injury", "hopital", people=4),
        FakeEvent("e2", "injury", "hopital", people=4),
    ])


def test_no_count_on_one_side_is_not_a_numeric_conflict():
    """One silent side cannot disagree with anything."""
    assert not C.detect_all([
        FakeEvent("e1", "injury", "hopital", people=4),
        FakeEvent("e2", "injury", "hopital", people=None),
    ])


# ---- entity resolution safety gate --------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("Al-Amal Hospital", "al amal hospital"),
    ("الجسر المركزي", "جسر المركزي"),
    (None, ""),
    ("", ""),
])
def test_normalize_place(value, expected):
    assert normalize_place(value) == expected


@pytest.mark.parametrize("text", ["the road", "here", "city", "الشارع", ""])
def test_generic_places_are_rejected(text):
    """Over-merging generic places would fabricate contradictions.

    If every mention of 'road' collapsed into a single place, two unrelated
    roads would appear to disagree. The generic guard is what prevents that.
    """
    assert is_generic(text)


def test_a_real_place_is_not_generic():
    assert not is_generic("hopital Al-Amal")
    assert not is_generic("الجسر المركزي")


def test_first_mention_needs_no_model_call():
    """A place nobody has mentioned before establishes its own canonical name."""
    client = StubClient(answer=None)
    resolver = LocationResolver(client=client, enabled=True)
    res = resolver.resolve("pont central", [])
    assert res.canonical == "pont central"
    assert res.reason == "first_mention"
    assert client.calls == []


def test_generic_mention_is_never_asked_about():
    """Never spend a model call on a placeholder place."""
    client = StubClient(answer=Res("true", "pont central"))
    resolver = LocationResolver(client=client, enabled=True)
    res = resolver.resolve("the road", ["pont central"])
    assert res.same is None
    assert client.calls == []


def test_resolver_fails_closed_without_lexical_support():
    """An unsupported merge is refused, not accepted.

    The 4B decode of this task tends to echo whichever candidate it is shown.
    Taken at face value it merged 'vieille medina', 'hopital Al-Amal' and
    'الجامعة' into 'pont central', manufacturing contradictions nobody
    reported. A missing contradiction is a gap an operator can see; a
    fabricated one is a lie carrying a confidence score.
    """
    client = StubClient(answer=Res("true", "pont central"))
    resolver = LocationResolver(client=client, enabled=True)
    res = resolver.resolve("hopital Al-Amal", ["pont central"])
    assert res.same is False
    assert res.reason == "model_proposed_merge_without_lexical_support"
    assert res.canonical == "hopital Al-Amal"


def test_exact_normalized_match_merges_without_model_agreement():
    """Identity after normalisation needs no model agreement.

    Whitespace and case differences are the same name. The model answering
    "false, it is a different place" is overruled here, because this is exact
    string identity and a 4B model disagreeing with arithmetic would be the
    less reliable of the two.
    """
    client = StubClient(answer=Res("false", "elsewhere"))
    resolver = LocationResolver(client=client, enabled=True)
    res = resolver.resolve("Pont  Central", ["pont central"])
    assert res.same is True
    assert res.reason == "exact_normalized_match"
    assert res.confidence == 1.0


def test_lexically_supported_merge_is_accepted():
    client = StubClient(answer=Res("true", "pont central"))
    resolver = LocationResolver(client=client, enabled=True)
    res = resolver.resolve("pont central vieux", ["pont central"])
    assert res.same is True
    assert res.reason == "apertus_name_with_lexical_support"


def test_curated_cross_language_alias_merges():
    """The lexicon is what licenses merging across scripts."""
    client = StubClient(answer=Res("true", "pont central"))
    resolver = LocationResolver(client=client, enabled=True)
    res = resolver.resolve("الجسر المركزي", ["pont central"])
    assert res.same is True


def test_unreachable_model_does_not_license_a_guess():
    """No model, no merge.

    ``same`` is None rather than False here, and that distinction matters:
    None means "we could not find out", False means "we checked and they
    differ". Collapsing them would let a model outage be misread as evidence
    that two places are different.
    """
    client = StubClient(answer=None)
    resolver = LocationResolver(client=client, enabled=True)
    res = resolver.resolve("hopital Al-Amal", ["pont central"])
    assert res.same is None
    assert res.reason == "model_unavailable"
    assert res.confidence == 0.0
    assert res.canonical == "hopital Al-Amal"


def test_disabled_resolver_abstains():
    client = StubClient(answer=Res("true", "pont central"))
    resolver = LocationResolver(client=client, enabled=False)
    res = resolver.resolve("الجسر المركزي", ["pont central"])
    assert res.same is None
    assert res.reason == "resolver_disabled"
    assert client.calls == []


def test_canonical_set_keeps_original_wording_available():
    """Resolution never rewrites the source; it only groups."""
    client = StubClient(answer=Res("true", "pont central"))
    resolver = LocationResolver(client=client, enabled=True)
    mapping = resolver.canonical_set(["pont central", "الجسر المركزي"])
    assert mapping["pont central"] == "pont central"
    assert mapping["الجسر المركزي"] == "pont central"
    # The Arabic key is still present, so the original wording remains
    # recoverable for the evidence panel.
    assert "الجسر المركزي" in mapping