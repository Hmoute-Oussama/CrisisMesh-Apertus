"""Situation graph.

Nodes: events, reports, places, infrastructure, hazards.
Edges: reported_by, occurred_at, affects, contradicts, corroborates, near,
        related_to.

NetworkX rather than Neo4j: the graph is a few hundred nodes at CrisisMesh's
scale, it is rebuilt from the database on demand, and a real graph database
would add a service to a 16 GB laptop for no benefit. The interface returns
plain JSON so the frontend is not coupled to the library.

The graph is a view. It owns no state, so it cannot drift from the evidence
tables that do.
"""

from __future__ import annotations

import networkx as nx

from ..taxonomy import EventType

# Event types that denote a place-bearing condition rather than a person.
_PLACE_TYPES = {
    EventType.ROAD_BLOCKED, EventType.ROAD_OPEN, EventType.FIRE,
    EventType.FLOOD, EventType.STRUCTURAL_DAMAGE, EventType.POWER_OUTAGE,
    EventType.WATER_OUTAGE, EventType.INFRASTRUCTURE_DAMAGE,
    EventType.SHELTER_CAPACITY, EventType.HAZARD,
}

# Colour contract, mirrored in the frontend. Kept here so the API and UI cannot
# disagree about what "disputed" looks like.
STATUS_COLOR = {
    "verified_by_operator": "#22c55e",
    "corroborated": "#22c55e",
    "disputed": "#f97316",
    "unverified": "#eab308",
    "rejected": "#6b7280",
}


def build_graph(events, reports, conflicts) -> dict:
    g = nx.DiGraph()

    for r in reports:
        g.add_node(
            f"report:{r.report_id}",
            kind="report", id=r.report_id, label=f"Report {r.report_id}",
            lang=r.source_language, suspected_injection=r.suspected_injection,
        )

    for e in events:
        nid = f"event:{e.event_id}"
        g.add_node(
            nid,
            kind="event",
            id=e.event_id,
            label=e.event_type.replace("_", " "),
            event_type=e.event_type,
            verification_state=e.verification_state,
            severity=e.severity,
            location=e.location_text,
            people_affected=e.people_affected,
            sources=e.independent_source_count,
            color=STATUS_COLOR.get(e.verification_state, "#eab308"),
        )
        g.add_edge(nid, f"report:{e.report_id}", rel="reported_by")

        if e.location_text:
            pid = f"place:{e.location_text.strip().lower()}"
            g.add_node(pid, kind="place", id=e.location_text, label=e.location_text)
            g.add_edge(nid, pid, rel="occurred_at")

        if e.people_affected:
            g.add_node(
                f"people:{e.people_affected}", kind="people",
                id=str(e.people_affected),
                label=f"{e.people_affected} people",
            )
            g.add_edge(nid, f"people:{e.people_affected}", rel="affects")

    # Contradictions are first-class edges, not a footnote. A picture in which
    # the dispute is invisible is a picture that misleads.
    for c in conflicts:
        a, b = f"event:{c.event_a_id}", f"event:{c.event_b_id}"
        if g.has_node(a) and g.has_node(b):
            g.add_edge(a, b, rel="contradicts", conflict_id=c.conflict_id,
                       conflict_type=c.conflict_type,
                       resolved=c.status != "unresolved")

    # Corroboration links: events of the same type and place describing one thing.
    by_key: dict[tuple, list] = {}
    for e in events:
        if not e.location_text:
            continue
        key = (e.event_type, e.location_text.strip().lower())
        by_key.setdefault(key, []).append(e)
    for members in by_key.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda x: x.created_at)
        for prev, nxt in zip(members, members[1:]):
            if prev.report_id != nxt.report_id:
                g.add_edge(
                    f"event:{prev.event_id}", f"event:{nxt.event_id}",
                    rel="corroborates",
                )

    nodes = [{"id": n, **{"data": d}} for n, d in g.nodes(data=True)]
    edges = [{"source": u, "target": v, **d} for u, v, d in g.edges(data=True)]
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "events": sum(1 for n in g.nodes if n.startswith("event:")),
            "places": sum(1 for n in g.nodes if n.startswith("place:")),
            "contradictions": sum(1 for _, _, d in g.edges(data=True)
                                  if d.get("rel") == "contradicts"),
        },
    }