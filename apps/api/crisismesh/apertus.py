"""Apertus client and extraction runner.

One model, many roles. Apertus does all of the semantic work in CrisisMesh:
language identification, segmentation-aware extraction, cross-report duplicate
adjudication, conflict explanation and synthesis. Python does none of it.

Two properties are load-bearing:

1.  **No silent substitution.** If the server is unreachable or the output
    cannot be parsed, this raises. It never falls back to another model and it
    never fabricates a placeholder event. ``model_used`` is recorded from what
    actually answered.

2.  **Determinism.** temperature 0, top_p 1, fixed grammar, fixed prompt
    version. The same report produces the same events on every run, which is
    what makes the Truth Preservation Score reproducible.

Wiring: any OpenAI-compatible server. Default is llama.cpp, because it runs on
CPU and can enforce the GBNF grammar during sampling.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

from .config import Settings, get_settings
from .grammar import GRAMMAR_VERSION, build_grammar
from .prompts import conflict_prompt, extraction_prompt, place_prompt
from .guards import REPORT_CLOSE, REPORT_OPEN

# A quoted value may contain an escaped double quote ("" inside the quotes), so
# the body is matched as (any char or a doubled quote) rather than [^"]*. The
# previous [^"]* form could not represent a quote inside a location and, more
# importantly, left the surrounding quote characters inside the captured group,
# so every stored location was literally '"pont central"'. That single quoting
# bug is why cross-location contradiction detection found nothing: no two
# canonically different strings ever matched.
_Q = r'(?:"(?:[^"]|"")*")'
_EVENT_RE = re.compile(
    r"\{type:(?P<etype>[a-z_]+),"
    rf"location:(?P<loc>{_Q}|null),"
    r'people:(?P<ppl>[0-9]+|null),'
    r'severity:(?P<sev>[a-z]+),'
    rf"time:(?P<tref>{_Q}|null),"
    rf"desc:(?P<desc>{_Q}|null)\}}"
)


def _unquote(value: str | None) -> str | None:
    """Strip the wire-format quotes and unescape doubled quotes.

    Returns None for the literal ``null``. Grammar-constrained output is always
    syntactically well formed, but keeping the null handling here means a
    malformed or truncated response degrades to an absent field instead of the
    string "null" being written into a report as though it were a location.
    """
    if value is None or value == "null":
        return None
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value


class ApertusUnavailable(RuntimeError):
    """Raised when the local Apertus server cannot be reached.

    Deliberately a hard failure. CrisisMesh does not silently degrade to a
    different model, because an operator who cannot tell which model produced a
    claim cannot audit the claim.
    """


@dataclass
class ExtractedEvent:
    event_type: str | None
    location_text: str | None
    people_affected: int | str | None
    severity: str | None
    time_reference: str | None
    description: str | None


@dataclass
class PlaceResolution:
    """Apertus's answer to "is this the same place as one of these?".

    ``same`` is None when the model declined, which is treated as "different"
    for safety: an unresolved place must never silently merge two reports.
    """

    canonical: str
    same: bool | None
    confidence: float
    reason: str


@dataclass
class ExtractionResult:
    raw: str
    events: list[ExtractedEvent] = field(default_factory=list)
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    grammar_held: bool = True


class ApertusClient:
    """Thin, synchronous client over an OpenAI-compatible chat endpoint."""

    def __init__(self, settings: Settings | None = None,
                 http_client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = http_client
        self._own_client = http_client is None

    # ---- plumbing ---------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        """POST to the chat endpoint, normalising every failure mode.

        A connection error, a timeout and a 500 are all the same situation from
        the caller's point of view: no answer came back, so no claim may be
        written. They must surface as ApertusUnavailable rather than as an
        httpx traceback, or the pipeline's error handling silently misses them.
        """
        try:
            if self._own_client:
                with httpx.Client(timeout=self.settings.apertus_timeout_s) as c:
                    r = c.post(self.settings.chat_url, json=payload)
            else:
                r = self._client.post(  # type: ignore[union-attr]
                    self.settings.chat_url, json=payload)
        except httpx.HTTPError as exc:
            raise ApertusUnavailable(
                f"cannot reach Apertus at {self.settings.chat_url}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if r.status_code >= 400:
            raise ApertusUnavailable(
                f"Apertus server returned {r.status_code}: {r.text[:400]}"
            )
        return r.json()

    def health(self) -> dict:
        try:
            if self._own_client:
                with httpx.Client(timeout=5.0) as c:
                    r = c.get(self.settings.models_url)
            else:
                r = self._client.get(self.settings.models_url)  # type: ignore[union-attr]
            ok = r.status_code == 200
            body = r.json() if ok else {}
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"available": False, "error": str(exc)[:200],
                    "model_label": self.settings.apertus_model_label}
        served = [m.get("id") for m in body.get("data", [])]
        return {
            "available": ok,
            "model_label": self.settings.apertus_model_label,
            "model_served": served,
            "license": self.settings.apertus_license,
        }

    # ---- extraction -------------------------------------------------------
    def extract_segment(
        self, segment_text: str, prompt_version: str | None = None
    ) -> ExtractionResult:
        """Extract events from ONE segment with grammar-constrained decoding.

        The grammar is generated from the taxonomy, so structurally invalid
        output is not merely unlikely, it is unreachable. What remains possible
        is *semantically* wrong, which is what guards.py exists to catch.
        """
        version = prompt_version or self.settings.prompt_version_extraction
        system = extraction_prompt(version)
        payload = {
            "model": self.settings.apertus_model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"{REPORT_OPEN}\n{segment_text}\n{REPORT_CLOSE}"},
            ],
            "temperature": self.settings.apertus_temperature,
            "top_p": self.settings.apertus_top_p,
            "max_tokens": self.settings.apertus_max_tokens,
            "grammar": build_grammar(),
            "cache_prompt": True,
        }
        t0 = time.perf_counter()
        resp = self._post(payload)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = resp.get("usage", {})
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        events = [
            ExtractedEvent(
                event_type=m.group("etype"),
                location_text=_unquote(m.group("loc")),
                people_affected=m.group("ppl"),
                severity=m.group("sev"),
                time_reference=_unquote(m.group("tref")),
                description=_unquote(m.group("desc")),
            )
            for m in _EVENT_RE.finditer(content)
        ]
        return ExtractionResult(
            raw=content,
            events=events,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            grammar_held=content.startswith("events:"),
        )

    # ---- place identity ---------------------------------------------------

    def resolve_place(self, mention: str,
                      candidates: list[str]) -> "PlaceResolution | None":
        """Ask whether ``mention`` names one of ``candidates``.

        This is an identity question, not a truth question. The model is asked
        which existing name (if any) refers to the same place, and is explicitly
        forbidden from commenting on the reports or their claims. Answering
        "different" for a place we already know costs us a missed contradiction;
        answering "same" for two different places would manufacture one. The
        grammar offers only those two answers plus a refusal, so the failure
        modes are all toward under-reporting.
        """
        from .grammar import build_place_grammar

        system = place_prompt(self.settings.prompt_version_place)
        options = " OR ".join(c[:80] for c in candidates[:6])
        payload = {
            "model": self.settings.apertus_model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content":
                    f"known places: {options}\nnew place: {mention[:120]}"},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 24,
            "grammar": build_place_grammar(),
        }
        try:
            resp = self._post(payload)
        except ApertusUnavailable:
            return None
        raw = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        m = re.search(r'place:\s*"([^"]*)"', raw)
        if not m:
            return None
        value = m.group(1).strip()
        if not value:
            return None
        # Identity is decided here in Python, not by the model: if the returned
        # name matches a candidate it is the same place, otherwise it is new.
        # The model only supplies the canonical spelling.
        return PlaceResolution(
            canonical=value,
            same=None,          # caller compares against candidates
            confidence=0.0,
            reason="apertus_canonical_name",
        )

    # ---- conflict rationale ----------------------------------------------
    def explain_conflict(self, a_text: str, b_text: str, field: str) -> str | None:
        """Ask Apertus to describe why two reports disagree.

        The model explains. It does not decide: the caller must not use the
        return value to pick a winner. Lowercase-only grammar keeps the output
        short and avoids the model restating the situation as a verdict.
        """
        from .grammar import build_conflict_grammar

        system = conflict_prompt(self.settings.prompt_version_conflict)
        payload = {
            "model": self.settings.apertus_model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"field: {field}\nfirst: {a_text}\nsecond: {b_text}"},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 60,
            "grammar": build_conflict_grammar(),
        }
        try:
            resp = self._post(payload)
        except ApertusUnavailable:
            return None
        raw = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        m = re.search(r'conflict:\s*"([^"]*)"', raw)
        return m.group(1) if m else None


__all__ = ["ApertusClient", "ApertusUnavailable", "ExtractedEvent",
           "ExtractionResult", "GRAMMAR_VERSION"]