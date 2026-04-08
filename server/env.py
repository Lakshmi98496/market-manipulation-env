"""
Market Manipulation Detection — OpenEnv Environment
====================================================
Implements the full OpenEnv spec:
  POST /reset   → ManipulationObservation
  POST /step    → StepResult
  GET  /state   → current episode state

KEY FIXES vs v1:
  1. Per-session isolation via SessionStore — safe for concurrent agents
  2. Rich narrative observation (context_hint) — forces LLM to reason
  3. Background stale-session cleanup
  4. Session ID via X-Session-ID header (auto-generated if absent)
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.models import ManipulationAction, StepResult
from server.session import SessionStore, TASKS, DEFAULT_TASK
from server.reward import compute_reward
from server.narrative import build_narrative

# ---------------------------------------------------------------------------
# App + session store
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Market Manipulation Detection — OpenEnv",
    description=(
        "RL environment for detecting spoofing, layering, and wash trading "
        "in a simulated live order book. Mirrors SEC/SEBI surveillance systems."
    ),
    version="2.0.0",
)

store = SessionStore()

# CORS — allows dashboard.html opened from browser to talk to server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def start_cleanup():
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)
            removed = store.cleanup_stale()
            if removed:
                print(f"[cleanup] Removed {removed} stale sessions. "
                      f"Active: {store.active_count}", flush=True)
    asyncio.create_task(_cleanup_loop())


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helper: enrich observation with narrative
# ---------------------------------------------------------------------------

def _enrich_obs(obs, session_id: str) -> dict:
    d = obs.dict()
    d["context_hint"] = build_narrative(obs)
    d["session_id"] = session_id
    return d


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(
    req: ResetRequest = ResetRequest(),
    x_session_id: Optional[str] = Header(default=None),
):
    """
    Reset the environment.

    Headers:
        X-Session-ID: optional. Auto-generated and returned if absent.

    Body (optional JSON):
        task: spoofing_detection | layering_wash_detection | adaptive_adversary_detection
        seed: integer for reproducibility
    """
    task = req.task if req.task and req.task in TASKS else DEFAULT_TASK
    seed = req.seed if req.seed is not None else random.randint(0, 999_999)

    session = store.get_or_create(x_session_id)
    session.reset(task_name=task, seed=seed)

    obs = session.simulator.reset(seed=seed)
    obs_dict = _enrich_obs(obs, session.session_id)

    return JSONResponse(content={
        "observation": obs_dict,
        "reward": 0.0,
        "done": False,
        "session_id": session.session_id,
        "info": {
            "task": task,
            "seed": seed,
            "max_steps": session.max_steps,
            "difficulty": TASKS[task]["difficulty"],
            "description": TASKS[task]["description"],
            "active_sessions": store.active_count,
        },
    })


@app.post("/step")
async def step(
    req: StepRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    """
    Take one action in the environment.

    Headers:
        X-Session-ID: session identifier returned by /reset.

    Body JSON:
        action: {decision, pattern_type, confidence}
    """
    # Look up session — support both header-based and headerless (single-agent)
    session = store.get(x_session_id) if x_session_id else None
    if session is None:
        # Headerless fallback: get or create default session
        session = store.get_or_create(None)
        if session.simulator is None or session.done:
            raise HTTPException(
                status_code=400,
                detail="Episode not started or already done. Call POST /reset first."
            )

    if session.done:
        raise HTTPException(
            status_code=400,
            detail="Episode finished. Call POST /reset to start a new episode."
        )

    # Parse and validate action
    last_action_error: Optional[str] = None
    try:
        action = ManipulationAction(**req.action)
    except Exception as exc:
        action = ManipulationAction(
            decision="ignore", pattern_type="none", confidence=0.0
        )
        last_action_error = str(exc)

    # Advance simulator
    obs, true_pattern = session.simulator.step(agent_decision=action.decision)
    obs_dict = _enrich_obs(obs, session.session_id)

    # Compute reward
    reward = compute_reward(
        decision=action.decision,
        pattern_type=action.pattern_type,
        confidence=action.confidence,
        true_pattern=true_pattern,
    )

    session.step_count += 1
    session.rewards.append(reward)
    session.decisions.append(action.decision)
    session.true_patterns.append(true_pattern)

    done = session.step_count >= session.max_steps
    session.done = done

    result = StepResult(
        observation=obs,
        reward=reward,
        done=done,
        true_pattern=true_pattern,
        last_action_error=last_action_error,
        info={
            "step": session.step_count,
            "max_steps": session.max_steps,
            "true_pattern": true_pattern,
            "episode_score_so_far": session.episode_score,
            "session_id": session.session_id,
        },
    )

    result_dict = result.dict()
    result_dict["observation"] = obs_dict
    return JSONResponse(content=result_dict)


@app.get("/state")
async def state(x_session_id: Optional[str] = Header(default=None)):
    """Return current episode state."""
    session = store.get(x_session_id) if x_session_id else None
    if session is None or session.simulator is None:
        return JSONResponse(content={
            "status": "not_started",
            "active_sessions": store.active_count,
        })

    return JSONResponse(content={
        "session_id": session.session_id,
        "task": session.task_name,
        "step": session.step_count,
        "max_steps": session.max_steps,
        "done": session.done,
        "rewards": session.rewards,
        "episode_score": session.episode_score,
        "decisions": session.decisions,
        "true_patterns": session.true_patterns,
        "elapsed_seconds": round(time.time() - session.started_at, 2),
        "active_sessions": store.active_count,
    })


@app.get("/tasks")
async def list_tasks():
    return JSONResponse(content={"tasks": TASKS})


@app.get("/health")
async def health():
    return JSONResponse(content={
        "status": "ok",
        "active_sessions": store.active_count,
        "version": "2.0.0",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
