"""Load the Al-Nour scenario and run the full CrisisMesh pipeline.

    python scripts/seed_demo.py --reset     # wipe, ingest, process everything
    python scripts/seed_demo.py             # continue an interrupted run

Reports are ingested with their scenario id as the primary key, so a seeded run
is reproducible: the same scenario always produces the same Report rows, and the
evaluation harness can join predictions to ground truth on that key.

Progress is printed per report because on this hardware extraction is real work:
several seconds per report with Apertus-v1.1-4B, so a full 34-report scenario
lands in minutes. That is the honest cost of local inference, and we print it
instead of hiding it behind a cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apps" / "api"))

from crisismesh.apertus import ApertusClient, ApertusUnavailable  # noqa: E402
from crisismesh.config import get_settings  # noqa: E402
from crisismesh.db import (  # noqa: E402
    Base,
    Conflict,
    DuplicateGroup,
    Event,
    EvidenceLink,
    Report,
    get_engine,
    init_db,
    session_scope,
)
from crisismesh.services import pipeline as pipeline_svc  # noqa: E402
from crisismesh.services.graph import build_graph  # noqa: E402


def reset() -> None:
    """Drop and recreate the schema, and remove the SQLite file."""
    eng = get_engine()
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    print("reset: schema recreated")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="datasets/demo/scenario.json")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    init_db()
    if args.reset:
        reset()

    st = get_settings()
    client = ApertusClient(st)
    health = client.health()
    if not health.get("available"):
        print(f"ERROR: Apertus not reachable at {st.apertus_base_url}")
        print(f"       {health.get('error', '')}")
        print("       start it with:  powershell scripts/start_apertus.ps1")
        return 2
    print(f"model    : {st.apertus_model_label} ({st.apertus_quant})")
    print(f"prompt   : {st.prompt_version_extraction}")
    print(f"grammar  : enforced\n")

    data = json.loads((REPO / args.scenario).read_text(encoding="utf-8"))
    reports = data["reports"][: args.limit] if args.limit else data["reports"]
    print(f"scenario : {data['scenario']}  ({len(reports)} reports)\n")

    # ---- ingest ----------------------------------------------------------
    with session_scope() as s:
        added = 0
        for r in reports:
            already = s.get(Report, r["report_id"]) is not None
            pipeline_svc.ingest_report(
                s,
                r["raw_text"],
                report_id=r["report_id"],
                source_language=r["source_language"],
                source_type=r["source_type"],
                reporter_id=r.get("reporter_id"),
                reported_at=datetime.fromisoformat(r["reported_at"]),
                metadata=r.get("metadata", {}),
            )
            if not already:
                added += 1
        print(f"ingested : {added} new, {len(reports) - added} already present")

    # ---- extract ---------------------------------------------------------
    t0 = time.perf_counter()
    made = 0
    for i, spec in enumerate(reports, 1):
        with session_scope() as s:
            rep = s.get(Report, spec["report_id"])
            if rep is None:
                print(f"[{i:02d}/{len(reports)}] {spec['report_id']} MISSING")
                continue
            if rep.processed:
                print(f"[{i:02d}/{len(reports)}] {rep.report_id} "
                      f"[{spec['source_language']}] already processed")
                continue
            try:
                res = pipeline_svc.process_report(s, rep, client)
            except ApertusUnavailable as exc:
                print(f"[{i:02d}/{len(reports)}] {rep.report_id} "
                      f"APERTUS FAIL: {exc}")
                return 3
            made += res["events"]
            types = [e.event_type for e in rep.events]
            guards = sum(len(e.guards_applied) for e in rep.events)
            flag = "  [INJECTION]" if rep.suspected_injection else ""
            g = f" guards={guards}" if guards else ""
            print(f"[{i:02d}/{len(reports)}] {rep.report_id} "
                  f"[{spec['source_language']}] {res['latency_ms']:>6}ms  "
                  f"{res['events']}ev  {','.join(types) or '(none)'}{g}{flag}")
    elapsed = time.perf_counter() - t0

    # ---- evidence stages -------------------------------------------------
    print("\n--- evidence stages ---")
    with session_scope() as s:
        locs = pipeline_svc.resolve_locations(s, client)
        print(f"places   : {locs['distinct_places']} distinct canonical place(s)"
              f" ({locs['resolved']} event(s) newly resolved)"
              if locs["enabled"] else "places   : entity resolution disabled")
        dedup = pipeline_svc.apply_deduplication(s)
        conf = pipeline_svc.detect_conflicts(s, client)
        print(f"dedup    : {dedup['groups']} duplicate group(s)")
        print(f"conflict : {conf['conflicts']} found "
              f"of {conf['checked_pairs']} pairs checked")
        for c in s.scalars(select(Conflict)).all():
            print(f"            {c.conflict_id} {c.conflict_type}/{c.severity} "
                  f"[{c.detection_rule}] {c.explanation[:90]}")

        events = s.scalars(select(Event)).all()
        reports_all = s.scalars(select(Report)).all()
        links = s.scalars(select(EvidenceLink)).all()
        conflicts = s.scalars(select(Conflict)).all()
        dupes = s.scalars(select(DuplicateGroup)).all()

        states: dict[str, int] = {}
        for e in events:
            states[e.verification_state] = states.get(e.verification_state, 0) + 1
        print(f"events   : {len(events)}")
        for k, v in sorted(states.items(), key=lambda x: -x[1]):
            print(f"            {k:<22} {v}")

        # The invariant that makes the whole project honest.
        by_event: dict[str, set[str]] = {}
        for l in links:
            by_event.setdefault(l.event_id, set()).add(l.report_id)
        ungrounded = [e.event_id for e in events
                      if e.report_id not in by_event.get(e.event_id, set())]
        print(f"evidence : {len(links)} links across {len(by_event)} events")
        print(f"grounding invariant: "
              f"{'PASS' if not ungrounded else 'FAIL ' + str(ungrounded[:5])}")

        max_src = max((e.independent_source_count for e in events), default=0)
        print(f"max independent sources on one event: {max_src}")

        graph = build_graph(events, reports_all, conflicts)
        out = REPO / "datasets" / "demo" / "graph.json"
        out.write_text(json.dumps(graph, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"graph    : {graph['stats']['nodes']} nodes / "
              f"{graph['stats']['edges']} edges -> {out.relative_to(REPO)}")

    print(f"\nelapsed  : {elapsed:.1f}s extraction "
          f"({elapsed / max(1, len(reports)):.1f}s/report, live Apertus)")
    print(f"events created: {made}")
    print("\nNext:  uvicorn crisismesh.main:app --app-dir apps/api --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())