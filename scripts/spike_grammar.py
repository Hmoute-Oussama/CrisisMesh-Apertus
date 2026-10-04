"""Standalone GBNF grammar validation. Fast iteration on the extraction grammar."""

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8090"

EVENT_TYPES = [
    "injury", "missing_person", "trapped_person", "road_blocked", "road_open",
    "fire", "flood", "structural_damage", "power_outage", "water_outage",
    "shelter_capacity", "medical_need", "food_need", "evacuation",
    "infrastructure_damage", "hazard", "resource_available", "resource_needed",
    "other",
]
SEVERITIES = ["unknown", "low", "medium", "high", "critical"]


def alt(items: list[str]) -> str:
    """Quoted alternation of literals."""
    return " | ".join(f'"{i}"' for i in items)


GRAMMAR = f"""
root   ::= "events:" arr
arr    ::= "[" event ("," event)* "]"
event  ::= "{{" etype "," loc "," ppl "," sev "," tref "," desc "}}"
etype  ::= {alt(EVENT_TYPES)}
loc    ::= "location:" qstr
ppl    ::= "people:" nint
sev    ::= "severity:" {alt(SEVERITIES)}
tref   ::= "time:" qstr
desc   ::= "desc:" qstr
qstr   ::= "\\"" [^"]* "\\"" | "null"
nint   ::= [0-9]{{1,4}} | "null"
""".strip()


def try_request(grammar: str) -> tuple[bool, str]:
    body = {
        "model": "apertus",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "grammar": grammar,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return True, json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", "replace")[:400]


if __name__ == "__main__":
    print(GRAMMAR)
    print("=" * 60)
    ok, out = try_request(GRAMMAR)
    print("grammar valid:", ok)
    print("output:", out)