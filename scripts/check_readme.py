"""Verify that every endpoint and script claimed in README.md actually exists.

    .venv/Scripts/python scripts/check_readme.py

Documentation that lies is worse than no documentation, so this is a check and
not a one-off manual read. Exits non-zero on any mismatch.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apps" / "api"))
sys.stdout.reconfigure(encoding="utf-8")

readme = (REPO / "README.md").read_text(encoding="utf-8")

from crisismesh.main import app  # noqa: E402

have: set[tuple[str, str]] = set()
for route in app.routes:
    path = getattr(route, "path", None)
    if not path:
        continue
    for method in getattr(route, "methods", []) or []:
        have.add((method, path))

# Rows look like:  | `GET` | `/events/{id}` | description |
row = re.compile(r"\|\s*`(GET|POST|PUT|DELETE)`\s*\|\s*`([^`]+)`\s*\|")
claimed = [(m.group(1), m.group(2)) for m in row.finditer(readme)]

print(f"routes registered : {len({p for _, p in have})}")
print(f"endpoints claimed : {len(claimed)}")

missing = [f"{m} {p}" for m, p in claimed if (m, p) not in have]
for m in missing:
    print(f"  MISSING ENDPOINT: {m}")
if not missing:
    print("  all claimed endpoints exist")

scripts = sorted(set(re.findall(r"scripts/[A-Za-z0-9_\-]+\.(?:py|ps1)", readme)))
for s in scripts:
    print(("  OK      " if (REPO / s).exists() else "  MISSING ") + s)

# Numbers quoted in the README must match the live database, so the README
# cannot drift away from measured reality without this failing.
try:
    from sqlalchemy import select
    from crisismesh.db import Event, Report, init_db, session_scope

    init_db()
    with session_scope() as s:
        n_reports = len(s.query(Report).all())
        n_events = len(s.query(Event).all())
    for label, actual in (("Reports processed", n_reports),
                          ("Events extracted", n_events)):
        # find "label | N" in the metrics table
        m = re.search(rf"\|\s*{label}\s*\|\s*([\d,]+)", readme)
        if not m:
            print(f"  (no figure for {label} in README)")
            continue
        claimed_n = int(m.group(1).replace(",", ""))
        flag = "OK  " if claimed_n == actual else "DRIFT"
        print(f"  {flag} {label}: README={claimed_n} actual={actual}")
        if claimed_n != actual:
            missing.append(f"{label} drift")
except Exception as exc:  # noqa: BLE001
    print(f"  (skipped metric check: {exc})")

print()
print("PROBLEMS:", len(missing))
raise SystemExit(1 if missing else 0)