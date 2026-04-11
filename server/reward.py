"""
Reward Function
===============
Computes step-level reward for the manipulation detection task.

Design goals:
  - Meaningful partial credit at every step (not just 0/1)
  - Penalise false positives (flagging legit market makers)
  - Reward confidence calibration
  - All outputs strictly in (0.01, 0.99) -- never exactly 0 or 1

Scoring breakdown:
  Base reward for correct decision       0.00 - 0.60
  Bonus for correct pattern_type         0.00 - 0.20
  Confidence calibration bonus           0.00 - 0.10
  False positive penalty                -0.30
  Escalation precision bonus             +0.10
  -------------------------------------------------
  Total (clamped)                        0.01 - 0.99
"""
from __future__ import annotations


# Correct action for each true pattern
EXPECTED_DECISION = {
    "spoofing":     "escalate",   # clear crime -> escalate
    "layering":     "escalate",   # clear crime -> escalate
    "wash_trading": "soft_flag",  # harder to prove -> soft flag acceptable
    "none":         "ignore",     # no pattern -> ignore
}

# Partial credit matrix  decision_made -> true_pattern -> score
PARTIAL_CREDIT = {
    #                   spoofing  layering  wash_trading  none
    "escalate":    {"spoofing": 0.60, "layering": 0.60, "wash_trading": 0.25, "none": 0.00},
    "soft_flag":   {"spoofing": 0.20, "layering": 0.20, "wash_trading": 0.60, "none": 0.10},
    "ignore":      {"spoofing": 0.00, "layering": 0.00, "wash_trading": 0.00, "none": 0.60},
}

# Pattern identification bonus (only if decision was correct)
PATTERN_BONUS = 0.20
FALSE_POSITIVE_PENALTY = 0.30
ESCALATION_BONUS = 0.10
MAX_CONFIDENCE_BONUS = 0.10


def compute_reward(
    decision: str,
    pattern_type: str,
    confidence: float,
    true_pattern: str,
) -> float:
    """
    Compute normalised reward in [0.0, 1.0].

    Args:
        decision:     agent's decision (ignore/soft_flag/escalate)
        pattern_type: agent's claimed pattern type
        confidence:   agent's stated confidence (0.0-1.0)
        true_pattern: ground truth from simulator

    Returns:
        reward: float in [0.0, 1.0]
    """
    # 1. Base partial credit
    reward = PARTIAL_CREDIT.get(decision, {}).get(true_pattern, 0.0)

    expected = EXPECTED_DECISION[true_pattern]

    # 2. Pattern identification bonus
    if decision == expected and pattern_type == true_pattern:
        reward += PATTERN_BONUS

    # 3. False positive penalty
    if decision in ("soft_flag", "escalate") and true_pattern == "none":
        reward -= FALSE_POSITIVE_PENALTY

    # 4. Escalation precision bonus (correctly escalated real crime)
    if decision == "escalate" and true_pattern in ("spoofing", "layering"):
        reward += ESCALATION_BONUS

    # 5. Confidence calibration bonus
    #    High confidence on a correct decision -> bonus
    #    High confidence on a wrong decision -> no bonus (already penalised by base)
    if decision == expected:
        reward += confidence * MAX_CONFIDENCE_BONUS
    else:
        # Penalise over-confident wrong answers
        reward -= (confidence - 0.5) * 0.05 if confidence > 0.5 else 0.0

    return round(max(0.0, min(1.0, reward)), 4)


def compute_episode_score(rewards: list) -> float:
    """
    Aggregate step rewards into a single episode score strictly in (0.0, 1.0).
    Uses mean, with a small bonus for consistency (low variance).
    Scores are clamped to (0.01, 0.99) -- never exactly 0 or 1 as required
    by the hackathon validator.
    """
    if not rewards:
        return 0.01
    mean = sum(rewards) / len(rewards)
    # Consistency bonus: reduce by variance penalty
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    score = mean - 0.05 * variance
    # Strictly between 0 and 1 as required by hackathon validator
    return round(max(0.01, min(0.99, score)), 4)
