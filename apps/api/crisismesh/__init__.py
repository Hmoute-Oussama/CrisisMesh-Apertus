"""CrisisMesh backend package.

CrisisMesh is an evidence-aware crisis information intelligence layer built
around the Apertus model family (Swiss AI Initiative, Apache-2.0).

Architecture in one line:

    Apertus  = the perception and interpretation engine (all semantics)
    Python   = the epistemic guardrail (all provenance, no semantics)

Apertus is never asked whether something is true. It is only asked what a
report says. Every claim it produces must be traceable to an immutable report,
and conflicts are surfaced rather than resolved.
"""

__version__ = "0.1.0"