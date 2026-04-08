"""
FastAPI server for SupportEnv — Customer Support Ticket Triage.

Endpoints:
  POST   /reset          Create a new episode
  POST   /step           Advance the episode
  GET    /state          Current episode state
  GET    /tasks          List tasks and action schema
  POST   /grader         Grade a finished episode
  GET    /health         Liveness check
  GET    /               Info / spec link
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import environment as env
from data import TASK_META
from models import (
    Action,
    GraderResponse,
    TaskInfo,
)

app = FastAPI(
    title="SupportEnv",
    description="An OpenEnv-compliant customer support ticket triage environment.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"
    ticket_index: Optional[int] = 0


class StepRequest(BaseModel):
    episode_id: str
    action: Action


class GraderRequest(BaseModel):
    episode_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {
        "name": "SupportEnv",
        "version": "1.0.0",
        "description": "OpenEnv customer support ticket triage environment",
        "tasks": list(TASK_META.keys()),
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state?episode_id=...",
            "tasks": "GET /tasks",
            "grader": "POST /grader",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health", tags=["meta"])
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/tasks", tags=["tasks"])
def tasks():
    result = []
    for task_id, meta in TASK_META.items():
        result.append(
            TaskInfo(
                task_id=task_id,
                name=meta["name"],
                description=meta["description"],
                difficulty=meta["difficulty"],
                max_steps=meta["max_steps"],
            )
        )
    return result


@app.post("/reset", tags=["control"])
def reset(req: ResetRequest = Body(default_factory=ResetRequest)):
    try:
        obs = env.reset(req.task_id, ticket_index=req.ticket_index or 0)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", tags=["control"])
def step(req: StepRequest):
    try:
        result = env.step(req.episode_id, req.action)
        return result.model_dump()
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", tags=["observation"])
def state(episode_id: str = Query(...)):
    try:
        st = env.get_state(episode_id)
        return st.model_dump()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/grader", tags=["evaluation"])
def grader(req: GraderRequest):
    try:
        score, breakdown, feedback = env.grade(req.episode_id)
        return GraderResponse(
            episode_id=req.episode_id,
            task_id=env._EPISODES[req.episode_id]["task_id"],
            score=score,
            breakdown=breakdown,
            feedback=feedback,
        ).model_dump()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
