"""Zero-build offline dashboard for CrisisMesh Apertus.

Mounted under the FastAPI app so it can be run entirely offline (http://127.0.0.1:8000/).
No Node, no npm, no JS build step. Uses vanilla HTML + CSS + a little inline JS to
poll the API and render events, evidence, contradictions, and audit history.

The point is transparency: an operator should be able to answer "why is this event
shown?" in under ten seconds by clicking through to its evidence and audit trail.
"""

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> str:
    html_path = Path(__file__).resolve().parent / "templates" / "dashboard.html"
    html = html_path.read_text(encoding="utf-8")
    return html
