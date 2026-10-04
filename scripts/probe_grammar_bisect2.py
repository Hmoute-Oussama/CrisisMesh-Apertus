"""Second bisect: is the two-alternative form the problem?

    .venv/Scripts/python scripts/probe_grammar_bisect2.py
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))
sys.stdout.reconfigure(encoding="utf-8")

from crisismesh.config import get_settings  # noqa: E402

st = get_settings()

VARIANTS = {
    "1_one_branch": 'root ::= "place:" answer\nanswer ::= "\\"" [a-z]+ "\\""\n',
    "2_two_branches": 'root ::= "place:" answer\nanswer ::= "\\"" [a-z]+ "\\"" | "\\"" other "\\""\n',
    "3_alt_with_space": 'root ::= "place:" answer\nanswer ::= "\\"" [a-z]+ "\\"" | "other"\n',
    "4_multiline_alt": (
        'root ::= "place:" answer\n'
        'answer ::= "\\"" [a-z]+ "\\""\n'
        '       | "other"\n'
    ),
    "5_named_rule_alt": (
        'root ::= "place:" answer\n'
        'answer ::= named | other\n'
        'named ::= "\\"" [a-z]+ "\\""\n'
        'other ::= "\\"" different "\\""\n'
    ),
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
            print(f"{name:20} HTTP {r.status_code}")
        else:
            print(f"{name:20} OK   {r.json()['choices'][0]['message']['content']!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:20} ERROR {type(exc).__name__}: {exc}")