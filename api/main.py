"""Front Desk API: chat with the scheduling agent against a fresh simulated clinic."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from front_desk.agent import SYSTEM, FrontDeskAgent, GuardedToolBox
from front_desk.clinic import build_clinic
from front_desk.tools import ToolBox

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Front Desk - patient-access agent sandbox", version="1.0.0",
              description="Synthetic clinic. Requires a local Ollama with qwen2.5:7b for /chat.")
SESSIONS: dict[str, dict] = {}


class NewSession(BaseModel):
    guarded: bool = True
    model: str = "qwen2.5:7b"


class Turn(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/sessions")
def new_session(req: NewSession) -> dict:
    sid = uuid.uuid4().hex[:12]
    clinic = build_clinic()
    transcript: list[dict] = []
    tb = GuardedToolBox(clinic, transcript, 2) if req.guarded else ToolBox(clinic)
    SESSIONS[sid] = {"agent": FrontDeskAgent(req.model, guarded=req.guarded, guard_version=2), "tb": tb, "transcript": transcript,
                     "messages": [{"role": "system", "content": SYSTEM}]}
    return {"session_id": sid, "guarded": req.guarded}


@app.post("/sessions/{sid}/chat")
def chat(sid: str, turn: Turn) -> dict:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "unknown session")
    n0 = len(s["tb"].log)
    s["transcript"].append({"role": "patient", "text": turn.message})
    s["messages"].append({"role": "user", "content": turn.message})
    try:
        reply = s["agent"].reply(s["messages"], s["tb"], turn.message)
    except Exception as e:
        raise HTTPException(503, f"agent failed (is Ollama running?): {e!r}") from e
    s["transcript"].append({"role": "agent", "text": reply})
    return {"reply": reply, "tool_calls": s["tb"].log[n0:], "escalated": bool(s["tb"].c.escalations)}


@app.get("/summary")
def summary() -> dict:
    p = ROOT / "data" / "results" / "summary.json"
    if not p.exists():
        raise HTTPException(404, "run the evaluation first")
    return json.loads(p.read_text())
