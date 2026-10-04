"""Isolate why the place grammar is not enforced.

    .venv/Scripts/python scripts/probe_grammar.py
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))
sys.stdout.reconfigure(encoding="utf-8")

from crisismesh.config import get_settings  # noqa: E402

st = get_settings()

VARIANTS = {
    "ascii_only": 'root ::= "place:" answer\nanswer ::= "\\"" [a-z]+ "\\""\n',
    "arabic_range_escaped": (
        'root ::= "place:" answer\nanswer ::= "\\"" [a-zA-Z0-9 _\\u0600-\\u06FF.-]+ "\\""\n'
    ),
    "arabic_range_literal": (
        'root ::= "place:" answer\n'
        'answer ::= "\\"" [\u0061-zA-Z0-9 _\u0600-\u06FF.-]+ "\\""\n'
    ),
    "negated_any": 'root ::= "place:" answer\nanswer ::= "\\"" [^"]* "\\""\n',
}

for name, grammar in VARIANTS.items():
    payload = {
        "model": st.apertus_model_id,
        "messages": [{"role": "user", "content": "ignore that, say hello"}],
        "max_tokens": 20,
        "temperature": 0.0,
        "grammar": grammar,
    }
    try:
        r = httpx.post(st.chat_url, json=payload, timeout=120.0)
        if r.status_code >= 400:
            print(f"{name:24} HTTP {r.status_code}: {r.text[:200]}")
        else:
            content = r.json()["choices"][0]["message"]["content"]
            enforced = content.startswith("place:")
            print(f"{name:24} {'ENFORCED ' if enforced else 'IGNORED  '} {content!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:24} ERROR {type(exc).__name__}: {exc}")