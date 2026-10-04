"""Bisect the place grammar to find the construct llama.cpp rejects.

    .venv/Scripts/python scripts/probe_grammar_bisect.py
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))
sys.stdout.reconfigure(encoding="utf-8")

from crisismesh.config import get_settings  # noqa: E402

st = get_settings()
CHAR = r"[a-zA-Z0-9 _؀-ۿ.-]"

VARIANTS = {
    "A_separate_rule": f'root ::= "place:" answer\nanswer ::= "\\"" name "\\"" | "\\"" different "\\""\nname ::= {CHAR}+\n',
    "B_inlined": f'root ::= "place:" answer\nanswer ::= "\\"" {CHAR}+ "\\"" | "\\"" different "\\""\n',
    "C_three_rules": f'root ::= "place:" answer\nanswer ::= quoted | literal\nquoted ::= "\\"" {CHAR}+ "\\""\nliteral ::= "\\"" different "\\""\n',
    "D_no_dash_in_class": f'root ::= "place:" answer\nanswer ::= "\\"" name "\\"" | "\\"" different "\\""\nname ::= [a-zA-Z0-9 _؀-ۿ.]+\n',
    "E_no_dot_dash": f'root ::= "place:" answer\nanswer ::= "\\"" name "\\"" | "\\"" different "\\""\nname ::= [a-zA-Z0-9 _؀-ۿ]+\n',
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
            print(f"{name:22} HTTP {r.status_code} {r.text[:120]}")
        else:
            print(f"{name:22} OK   {r.json()['choices'][0]['message']['content']!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:22} ERROR {type(exc).__name__}: {exc}")