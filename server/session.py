"""Session Manager - per-session state store."""
from __future__ import annotations
import threading, time, uuid
from typing import Dict, List, Optional
from server.simulator import OrderBookSimulator
from server.reward import compute_episode_score

TASKS = {
    "spoofing_detection": {
        "max_steps": 15,
        "grader": "tasks.graders.grade_easy",
        "reward_range": [0, 1],
    },
    "layering_wash_detection": {
        "max_steps": 20,
        "grader": "tasks.graders.grade_medium",
        "reward_range": [0, 1],
    },
    "adaptive_adversary_detection": {
        "max_steps": 25,
        "grader": "tasks.graders.grade_hard",
        "reward_range": [0, 1],
    },
}

DEFAULT_TASK = "spoofing_detection"
SESSION_TTL_SECONDS = 3600

class EpisodeSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.task_name = DEFAULT_TASK
        self.simulator = None
        self.step_count = 0
        self.max_steps = 15
        self.rewards = []
        self.decisions = []
        self.true_patterns = []
        self.done = False
        self.seed = 42
        self.started_at = time.time()
        self.last_active = time.time()

    def reset(self, task_name, seed):
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

    def touch(self): 
        self.last_active = time.time()

    def is_stale(self): 
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS

    @property
    def episode_score(self): 
        return compute_episode_score(self.rewards)


class SessionStore:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id=None):
        with self._lock:
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                s.touch()
                return s
            new_id = session_id or str(uuid.uuid4())
            s = EpisodeSession(new_id)
            self._sessions[new_id] = s
            return s

    def get(self, session_id):
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.touch()
            return s

    def delete(self, session_id):
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_stale(self):
        with self._lock:
            stale = [k for k, v in self._sessions.items() if v.is_stale()]
            for k in stale:
                del self._sessions[k]
            return len(stale)

    @property
    def active_count(self):
        with self._lock:
            return len(self._sessions)


store = SessionStore()
