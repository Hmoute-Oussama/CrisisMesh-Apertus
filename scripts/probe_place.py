"""Probe place identity directly against the running Apertus server.

    .venv/Scripts/python scripts/probe_place.py

Prints the raw model reply for a few (mention, candidates) pairs so we can see
whether identity resolution works or whether the model always answers
"different". Useful when tuning; not part of the pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))
sys.stdout.reconfigure(encoding="utf-8")

from crisismesh.apertus import ApertusClient  # noqa: E402
from crisismesh.config import get_settings  # noqa: E402
from crisismesh.grammar import build_place_grammar  # noqa: E402

CASES = [
    ("الجسر المركزي", ["pont central", "hopital al amal", "medina"]),
    ("الطريق", ["pont central", "hopital al amal", "medina"]),
    ("stadium", ["pont central", "medina", "stade omnisports"]),
    ("الملعب الرياضي", ["pont central", "stade omnisports", "medina"]),
    ("المستشفى", ["hopital al amal", "stade omnisports", "medina"]),
]

client = ApertusClient(get_settings())
print("grammar:")
print(build_place_grammar())
print()

for mention, cands in CASES:
    res = client.resolve_place(mention, cands)
    print(f"{mention!r} vs {cands}")
    print(f"   -> same={res.same if res else None} "
          f"canonical={(res.canonical if res else None)!r} "
          f"reason={(res.reason if res else 'UNAVAILABLE')!r}")
    print()