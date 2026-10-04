"""Is the 4B model able to answer 'same' at all, or is it stuck on 'different'?

    .venv/Scripts/python scripts/probe_place2.py

Includes an exact-string control. If the control fails, the problem is the model
or the prompt framing, not multilingual difficulty.
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))
sys.stdout.reconfigure(encoding="utf-8")

from crisismesh.apertus import ApertusClient  # noqa: E402
from crisismesh.config import get_settings  # noqa: E402
from crisismesh.prompts import place_prompt  # noqa: E402

st = get_settings()
client = ApertusClient(st)
system = place_prompt(st.prompt_version_place)

CASES = [
    ("CONTROL exact same string", "pont central", ["pont central", "medina"]),
    ("CONTROL case/format only", "Pont Central", ["pont central", "medina"]),
    ("fr->ar bridge", "الجسر المركزي", ["pont central", "hopital al amal"]),
    ("en->fr stadium", "stadium", ["stade omnisports", "medina"]),
    ("ar->fr hospital", "المستشفى", ["hopital al amal", "stade omnisports"]),
    ("genuinely different", "حي النخيل", ["pont central", "hopital al amal"]),
]

for label, mention, cands in CASES:
    options = " OR ".join(cands)
    payload = {
        "model": st.apertus_model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content":
                f"known places: {options}\nnew place: {mention}"},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 24,
        "grammar": __import__(
            "crisismesh.grammar", fromlist=["x"]).build_place_grammar(),
    }
    r = httpx.post(st.chat_url, json=payload, timeout=180.0)
    raw = (r.json()["choices"][0]["message"]["content"] if r.status_code < 400
           else f"HTTP {r.status_code}")
    res = client.resolve_place(mention, cands)
    print(f"{label:26} raw={raw!r:34} same={res.same if res else None}")