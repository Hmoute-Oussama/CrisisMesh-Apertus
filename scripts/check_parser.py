"""Parser check against realistic llama.cpp output.

Run: .venv/Scripts/python scripts/check_parser.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))
sys.stdout.reconfigure(encoding="utf-8")

from crisismesh.apertus import _EVENT_RE  # noqa: E402

CASES = [
    ('events:[{type:road_blocked,location:"pont central",people:null,'
     'severity:unknown,time:null,desc:"bloque par debris"}]'),
    # a location that itself contains a double quote
    ('events:[{type:road_blocked,location:"pont ""est""",people:null,'
     'severity:unknown,time:null,desc:"x"}]'),
    # numeric people
    ('events:[{type:injury,location:"hopital",people:4,'
     'severity:medium,time:null,desc:"quatre blesses"}]'),
    # arabic, null everything
    ('events:[{type:power_outage,location:"المدينة",people:null,'
     'severity:unknown,time:null,desc:null}]'),
    # two events
    ('events:[{type:water_outage,location:"a",people:null,'
     'severity:unknown,time:null,desc:"d1"},'
     '{type:food_need,location:"b",people:null,'
     'severity:unknown,time:null,desc:"d2"}]'),
]

fail = 0
for raw in CASES:
    events = list(_EVENT_RE.finditer(raw))
    label = raw[:58].replace("\n", " ")
    if not events:
        print(f"FAIL  no match: {label}")
        fail += 1
        continue
    for m in events:
        print(f"ok    type={m.group('etype')}")
        print(f"      loc ={m.group('loc')!r}")
        print(f"      ppl ={m.group('ppl')!r}  sev={m.group('sev')!r}")
        print(f"      desc={m.group('desc')!r}")

print()
print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)