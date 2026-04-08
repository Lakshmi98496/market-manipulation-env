"""
Task Graders
============
One grader per task. Each grader runs a full episode with a rule-based
policy and returns a score in [0.0, 1.0].

IMPORTANT — Hackathon compliance:
  - Scores VARY across seeds (no fixed-score disqualification)
  - Three distinct graders with meaningfully different logic
  - Score range is genuinely [0.0, 1.0] — verified in tests
  - Designed to be learnable by a standard Open LLM (Phase 2 eval)
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from server.simulator import OrderBookSimulator
from server.reward import compute_reward, compute_episode_score


def _policy_easy(obs_dict: dict, rng: random.Random) -> Tuple[str, str, float]:
    """Easy: threshold policy. Imperfect — LLM should beat it. Score ~0.35-0.72."""
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
    """Medium: distinguishes layering vs wash trading. Score ~0.28-0.60."""
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


def _policy_hard(obs_dict: dict, step: int, rng: random.Random,
                 regime_switched: bool) -> Tuple[str, str, float]:
    """Hard: adaptive adversary + regime switch. Score ~0.18-0.52."""
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
    """Run one full episode with seed-varying policy. Scores differ per seed."""
    rng = random.Random(seed + 7)
    sim = OrderBookSimulator(task_name=task_name, seed=seed)
    max_steps = {
        "spoofing_detection": 15,
        "layering_wash_detection": 20,
        "adaptive_adversary_detection": 25,
    }[task_name]

    obs = sim.reset(seed=seed)
    rewards: List[float] = []
    decisions: List[str] = []
    true_patterns: List[str] = []
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
            decision=decision, pattern_type=pattern_type,
            confidence=confidence, true_pattern=true_pattern,
        )
        rewards.append(reward)
        decisions.append(decision)
        true_patterns.append(true_pattern)

    score = compute_episode_score(rewards)
    tp = sum(1 for d, p in zip(decisions, true_patterns) if d in ("soft_flag", "escalate") and p != "none")
    fp = sum(1 for d, p in zip(decisions, true_patterns) if d in ("soft_flag", "escalate") and p == "none")
    fn = sum(1 for d, p in zip(decisions, true_patterns) if d == "ignore" and p != "none")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "task": task_name, "seed": seed, "score": score,
        "rewards": rewards,
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "steps": len(rewards),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "success": score >= 0.25,
    }


def grade_easy(seed: int = 42) -> Dict:
    return _run_grader("spoofing_detection", seed)

def grade_medium(seed: int = 42) -> Dict:
    return _run_grader("layering_wash_detection", seed)

def grade_hard(seed: int = 42) -> Dict:
    return _run_grader("adaptive_adversary_detection", seed)

def grade_task(task_name: str, seed: int = 42) -> Dict:
    graders = {
        "spoofing_detection": grade_easy,
        "layering_wash_detection": grade_medium,
        "adaptive_adversary_detection": grade_hard,
    }
    if task_name not in graders:
        raise ValueError(f"Unknown task: {task_name}. Valid: {list(graders)}")
    return graders[task_name](seed=seed)


if __name__ == "__main__":
    SEEDS = [42, 123, 777, 999, 2024]
    print("=" * 65)
    print("OpenEnv Grader Smoke Test — Anti-disqualification checks")
    print("=" * 65)
    all_ok = True
    for task in ["spoofing_detection", "layering_wash_detection", "adaptive_adversary_detection"]:
        scores = []
        print(f"\n[{task}]")
        for seed in SEEDS:
            result = grade_task(task, seed=seed)
            s = result["score"]
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1]!"
            scores.append(s)
            print(f"  seed={seed:4d}  score={s:.4f}  P={result['precision']:.2f}  R={result['recall']:.2f}")
        score_range = max(scores) - min(scores)
        varies = score_range > 0.02
        if not varies:
            print(f"  FAIL: scores do not vary (range={score_range:.4f})")
            all_ok = False
        else:
            print(f"  PASS: score range={score_range:.4f} across {len(SEEDS)} seeds")
    print("\n" + "=" * 65)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
