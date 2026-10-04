"""GBNF grammar for grammar-constrained extraction.

Why a grammar instead of "prompt and hope for JSON":

  With llama.cpp, a GBNF grammar is enforced during sampling, so the model
  cannot emit a character that would break the structure. Measured on
  Apertus-v1.1-4B this gave 100% parseable output across French, Arabic,
  Darija and English, with no repair retries.

  The grammar is generated from taxonomy.py, so adding an event type changes
  the legal output space at the same moment it changes the code. The model
  cannot invent "giraffe_accident": it is not in the grammar.

The wire format is a flat tagged DSL rather than nested JSON. Flat key:value
pairs measurably reduced failure modes on the 4B model (no escaping hazards,
fixed key order, easy to regex-parse deterministically) and kept output short,
which matters at ~12 t/s on CPU.
"""

from __future__ import annotations

from .taxonomy import EventType, Severity

GRAMMAR_VERSION = "gbnf_v1"


def _alt(values) -> str:
    return " | ".join(f'"{v}"' for v in values)


def build_grammar() -> str:
    """Return the extraction GBNF grammar.

    Field order is fixed by the grammar, which is why the parser is trivial and
    why partial output is still recoverable.
    """
    etypes = _alt([e.value for e in EventType])
    sevs = _alt([s.value for s in Severity])
    return f"""
root   ::= "events:" arr
arr    ::= "[" event ("," event)* "]"
event  ::= "{{type:" etype ",location:" qstr ",people:" nint ",severity:" sev ",time:" qstr ",desc:" qstr "}}"
etype  ::= {etypes}
sev    ::= {sevs}
qstr   ::= "\\\"" qchar* "\\\"" | "null"
qchar  ::= [^"\\\\]
nint   ::= [0-9]{{1,4}} | "null"
""".strip()


def build_conflict_grammar() -> str:
    """Grammar for conflict rationale. Model explains, it never decides.

    The minimum length is load-bearing. Without a lower bound, constrained
    decoding collapses to the shortest legal string: an earlier version produced
    explanations of "blocked" and "road", which restate one side instead of
    describing the disagreement. A lower bound of 20 characters makes a
    one-word answer unreachable.

    Single alternative only: two alternatives that both begin with a quoted
    literal are rejected by llama.cpp's GBNF parser (see
    scripts/probe_grammar_bisect2.py).
    """
    return (
        'root ::= "conflict:" "\\"" expl "\\""\n'
        "expl ::= [a-z0-9 ,.']{20,150}\n"
    )


def build_place_grammar() -> str:
    """Grammar for cross-language place identity.

    Deliberately has NO abstention token. An earlier version offered a bare
    ``| "different"`` alternative, and Apertus-v1.1-4B answered
    ``place:different`` for every input, including a control where the new name
    was character-identical to a known one (scripts/probe_place2.py).

    The reason is constrained-decoding bias: the sentinel is reachable in one
    token while a name costs three, so the shortest legal continuation is always
    abstention. Prompt engineering could not fix it, because the failure is in
    the grammar's shape, not in the instructions.

    So the model is only ever asked for the canonical name of the place, in the
    same language as the known names. "Different" is expressed by returning a
    name that is not in the list, and the comparison happens in Python. That
    puts the actual decision in deterministic code and leaves the model with one
    job it can do.

    llama.cpp GBNF constraints honoured here, both found by bisection
    (scripts/probe_grammar_bisect2.py):
      - two alternatives both starting with a quoted literal are rejected;
      - ``\\uXXXX`` escapes are unreliable inside a character class, so the
        Arabic range is written with literal characters.
    """
    return (
        'root ::= "place:" "\\"" name "\\""\n'
        "name ::= [a-zA-Z0-9 _"
        "\u0600-\u06FF"
        ".\\-]+\n"
    )


def build_normalization_grammar() -> str:
    """Grammar for operator-facing normalization. Never replaces the original."""
    return """
root   ::= "lang:" lang " meaning:" meaning
lang   ::= "en" | "fr" | "ar" | "dar"
meaning::= "\\"" [a-z ]* "\\""
""".strip()