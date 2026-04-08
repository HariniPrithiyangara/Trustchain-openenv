# Trigger build: v1.0.2
"""
app.py — TrustChain OpenEnv HTTP server
Exposes /reset, /step, /state, /tasks, /health endpoints.
Run with: uvicorn app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from trustchain_env import TrustChainEnvironment
from trustchain_env.graders import (
    VerifyCredentialGrader,
    TraceProvenanceGrader,
    AuditAnomalyGrader,
)

app = FastAPI(title="TrustChain OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = TrustChainEnvironment()

_GRADERS = {
    "verify_credential": VerifyCredentialGrader(),
    "trace_provenance": TraceProvenanceGrader(),
    "audit_anomaly": AuditAnomalyGrader(),
}


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    task: Optional[str] = None  # alias for compatibility


class StepRequest(BaseModel):
    action_type: str
    target_id: Optional[str] = None
    verdict: Optional[str] = None
    reason: Optional[str] = None
    remediation: Optional[str] = None


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> dict:
    task_id = req.task_id or req.task
    return _env.reset(task_id=task_id)


@app.post("/step")
def step(req: StepRequest) -> dict:
    action = req.dict(exclude_none=True)
    result = _env.step(action)

    if result.get("done"):
        state = result.get("state", {})
        task_id = state.get("task_id", "verify_credential")
        grader = _GRADERS.get(task_id)
        score = grader(state) if grader else 0.0
        result["score"] = round(score, 4)

    return result


@app.get("/state")
def state() -> dict:
    return _env.state()


@app.get("/tasks")
def list_tasks() -> dict:
    return {
        "tasks": [
            {
                "id": "verify_credential",
                "difficulty": "easy",
                "grader": "trustchain_env.graders:VerifyCredentialGrader",
            },
            {
                "id": "trace_provenance",
                "difficulty": "medium",
                "grader": "trustchain_env.graders:TraceProvenanceGrader",
            },
            {
                "id": "audit_anomaly",
                "difficulty": "hard",
                "grader": "trustchain_env.graders:AuditAnomalyGrader",
            },
        ]
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
