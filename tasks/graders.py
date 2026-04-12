"""
Task Graders
============
One grader per task. Each grader runs a full episode with a rule-based
policy and returns a score strictly in (0.01, 0.99).
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from server.simulator import OrderBookSimulator
from server.reward import compute_reward, compute_episode_score


def _policy_easy(obs_dict: dict, rng: random.Random) -> Tuple[str, str, float]:
    imbalance   = obs_dict.get("order_imbalance", 0.0)
    cancel_rate = obs_dict.get("cancel_rate", 0.0)
    if cancel_rate > 0.45 and abs(imbalance) > 0.40:
        return "escalate", "spoofing", 0.85
    if cancel_rate > 0.38 or abs(imbalance) > 0.55:
        return "escalate", "spoofing", 0.60
    if rng.random() < 0.12:
        return "soft_flag", "spoofing", 0.35
    return "ignore", "none", 0.90


def _policy_medium(obs_dict: dict, rng: random.Random) -> Tuple[str, str, float]:
    imbalance   = obs_dict.get("order_imbalance", 0.0)
    cancel_rate = obs_dict.get("cancel_rate", 0.0)
    spread      = obs_dict.get("spread", 0.04)
    tape        = obs_dict.get("trade_tape", [])
    sizes = [t.get("size", 0) for t in tape[:6]] if tape else []
    repeated_sizes = len(sizes) != len(set(sizes)) if sizes else False
    if repeated_sizes and spread < 0.035:
        conf = 0.65 + rng.uniform(-0.1, 0.1)
        return "soft_flag", "wash_trading", max(0.1, min(0.9, conf))
    if cancel_rate > 0.42 and abs(imbalance) > 0.30:
        conf = 0.70 + rng.uniform(-0.15, 0.10)
        return "escalate", "layering", max(0.1, min(0.9, conf))
    if abs(imbalance) > 0.45:
        pattern = rng.choice(["layering", "spoofing"])
        return "soft_flag", pattern, 0.45
    if rng.random() < 0.08:
        return "soft_flag", "wash_trading", 0.30
    return "ignore", "none", 0.80


def _policy_hard(obs_dict: dict, step: int, rng: random.Random, regime_switched: bool) -> Tuple[str, str, float]:
    imbalance   = obs_dict.get("order_imbalance", 0.0)
    cancel_rate = obs_dict.get("cancel_rate", 0.0)
    context     = obs_dict.get("context_hint", "")
    tape        = obs_dict.get("trade_tape", [])
    in_volatile = "volatile" in context or regime_switched
    cancel_thresh    = 0.55 if in_volatile else 0.40
    imbalance_thresh = 0.55 if in_volatile else 0.38
    sizes = [t.get("size", 0) for t in tape[:8]] if tape else []
    wash_signal = sum(sizes.count(s) >= 2 for s in set(sizes)) >= 2 if sizes else False

    if wash_signal and not in_volatile:
        conf = 0.60 + rng.uniform(-0.1, 0.1)
        return "soft_flag", "wash_trading", max(0.1, min(0.9, conf))

    if cancel_rate > cancel_thresh and abs(imbalance) > imbalance_thresh:
        pattern = "spoofing" if step < 12 else rng.choice(["layering", "spoofing"])
        conf = 0.65 + rng.uniform(-0.15, 0.10)
        return "escalate", pattern, max(0.1, min(0.9, conf))

    if abs(imbalance) > 0.60 and not in_volatile:
        return "soft_flag", "layering", 0.50

    if rng.random() < 0.06:
        return "soft_flag", rng.choice(["spoofing", "layering"]), 0.25

    return "ignore", "none", 0.85


def _run_grader(task_name: str, seed: int) -> Dict:
    rng = random.Random(seed + 7)
    sim = OrderBookSimulator(task_name=task_name, seed=seed)

    max_steps = {
        "spoofing_detection": 15,
        "layering_wash_detection": 20,
        "adaptive_adversary_detection": 25,
    }[task_name]

    obs = sim.reset(seed=seed)
    rewards: List[float] = []
    regime_switched = False

    for step in range(1, max_steps + 1):
        obs_dict = obs.dict()

        if "volatile" in obs_dict.get("context_hint", ""):
            regime_switched = True

        if task_name == "spoofing_detection":
            decision, pattern_type, confidence = _policy_easy(obs_dict, rng)
        elif task_name == "layering_wash_detection":
            decision, pattern_type, confidence = _policy_medium(obs_dict, rng)
        else:
            decision, pattern_type, confidence = _policy_hard(obs_dict, step, rng, regime_switched)

        obs, true_pattern = sim.step(agent_decision=decision)

        reward = compute_reward(
            decision=decision,
            pattern_type=pattern_type,
            confidence=confidence,
            true_pattern=true_pattern,
        )

        rewards.append(reward)

    score = compute_episode_score(rewards)

    return {
        "task": task_name,
        "score": score,
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.01,
        "steps": len(rewards),
        "success": score >= 0.25,
    }


def grade_easy(seed: int = 42) -> float:
    return _run_grader("spoofing_detection", seed)["score"]


def grade_medium(seed: int = 42) -> float:
    return _run_grader("layering_wash_detection", seed)["score"]


def grade_hard(seed: int = 42) -> float:
    return _run_grader("adaptive_adversary_detection", seed)["score"]


def grade_task(task_name: str, seed: int = 42) -> float:
    return _run_grader(task_name, seed)["score"]
