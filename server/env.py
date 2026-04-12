"""
Market Manipulation Detection — OpenEnv Environment
====================================================
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.models import ManipulationAction, StepResult
from server.session import SessionStore, TASKS, DEFAULT_TASK
from server.reward import compute_reward
from server.narrative import build_narrative

app = FastAPI(
    title="Market Manipulation Detection — OpenEnv",
    description="RL environment for detecting spoofing, layering, and wash trading",
    version="2.0.0",
)

store = SessionStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def start_cleanup():
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)
            store.cleanup_stale()

    asyncio.create_task(_cleanup_loop())


class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


def _enrich_obs(obs, session_id: str) -> dict:
    d = obs.dict()
    d["context_hint"] = build_narrative(obs)
    d["session_id"] = session_id
    return d


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest(), x_session_id: Optional[str] = Header(default=None)):
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
        "info": {"task": task, "seed": seed},
    })


@app.post("/step")
async def step(req: StepRequest, x_session_id: Optional[str] = Header(default=None)):
    session = store.get(x_session_id) if x_session_id else store.get_or_create(None)

    if session.simulator is None or session.done:
        raise HTTPException(status_code=400, detail="Call /reset first")

    action = ManipulationAction(**req.action)

    obs, true_pattern = session.simulator.step(agent_decision=action.decision)
    obs_dict = _enrich_obs(obs, session.session_id)

    reward = compute_reward(
        decision=action.decision,
        pattern_type=action.pattern_type,
        confidence=action.confidence,
        true_pattern=true_pattern,
    )

    session.step_count += 1
    done = session.step_count >= session.max_steps
    session.done = done

    result = StepResult(
        observation=obs,
        reward=reward,
        done=done,
        true_pattern=true_pattern,
    )

    result_dict = result.dict()
    result_dict["observation"] = obs_dict

    return JSONResponse(content=result_dict)


@app.get("/state")
async def state(x_session_id: Optional[str] = Header(default=None)):
    session = store.get(x_session_id)
    if session is None or session.simulator is None:
        return {"status": "not_started"}

    return {
        "task": session.task_name,
        "step": session.step_count,
    }


@app.get("/tasks")
async def list_tasks():
    tasks_list = []
    for name, info in TASKS.items():
        tasks_list.append({
            "name": name,
            "max_steps": info["max_steps"],
            "grader": info.get("grader", ""),
            "reward_range": info.get("reward_range", [0, 1]),
        })
    return {"tasks": tasks_list}
@app.get("/tasks/{task_name}/grade")
@app.post("/tasks/{task_name}/grade")
async def grade_task_endpoint(task_name: str):
    from tasks.graders import grade_task

    score = grade_task(task_name, seed=42)

    return {
        "task": task_name,
        "score": score,
        "success": score >= 0.25,
        "mean_reward": score,
        "steps": 0,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
