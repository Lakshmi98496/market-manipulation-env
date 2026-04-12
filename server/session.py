"""
Session Manager
===============
Replaces the dangerous module-level global _episode with a proper
per-session state store keyed by session_id.

The hackathon harness (Phase 2) runs multiple concurrent episodes —
one per agent instance. A global state object causes all sessions to
share and overwrite each other's episode data, producing silent
wrong scores.

This module provides thread-safe, per-session isolation.
"""
"""Session Manager - per-session state store."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional

from server.simulator import OrderBookSimulator
from server.reward import compute_episode_score


TASKS = {
    "spoofing_detection": {
        "difficulty": "easy",
        "max_steps": 15,
        "description": "Detect single-pattern spoofing in a clean order book.",
        "grader": "tasks.graders.grade_easy",
        "reward_range": [0.01, 0.99],
    },
    "layering_wash_detection": {
        "difficulty": "medium",
        "max_steps": 20,
        "description": "Identify layering and wash trading mixed with HFT noise.",
        "grader": "tasks.graders.grade_medium",
        "reward_range": [0.01, 0.99],
    },
    "adaptive_adversary_detection": {
        "difficulty": "hard",
        "max_steps": 25,
        "description": "Track an adaptive manipulator through a regime shift.",
        "grader": "tasks.graders.grade_hard",
        "reward_range": [0.01, 0.99],
    },
}

DEFAULT_TASK = "spoofing_detection"
SESSION_TTL_SECONDS = 3600


class EpisodeSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.task_name: str = DEFAULT_TASK
        self.simulator: Optional[OrderBookSimulator] = None
        self.step_count: int = 0
        self.max_steps: int = 15
        self.rewards: List[float] = []
        self.decisions: List[str] = []
        self.true_patterns: List[str] = []
        self.done: bool = False
        self.seed: int = 42
        self.started_at: float = time.time()
        self.last_active: float = time.time()

    def reset(self, task_name: str, seed: int) -> None:
        self.task_name = task_name
        self.seed = seed
        self.max_steps = TASKS[task_name]["max_steps"]
        self.simulator = OrderBookSimulator(task_name=task_name, seed=seed)
        self.step_count = 0
        self.rewards = []
        self.decisions = []
        self.true_patterns = []
        self.done = False
        self.started_at = time.time()
        self.last_active = time.time()

    def touch(self) -> None:
        self.last_active = time.time()

    def is_stale(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS

    @property
    def episode_score(self) -> float:
        return compute_episode_score(self.rewards)


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, EpisodeSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: Optional[str] = None) -> EpisodeSession:
        with self._lock:
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                s.touch()
                return s
            new_id = session_id or str(uuid.uuid4())
            s = EpisodeSession(new_id)
            self._sessions[new_id] = s
            return s

    def get(self, session_id: str) -> Optional[EpisodeSession]:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.touch()
            return s

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_stale(self) -> int:
        with self._lock:
            stale = [k for k, v in self._sessions.items() if v.is_stale()]
            for k in stale:
                del self._sessions[k]
            return len(stale)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


store = SessionStore()
