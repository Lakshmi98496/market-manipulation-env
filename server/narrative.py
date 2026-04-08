"""
Narrative Builder
=================
Converts raw order book signals into a rich, structured analyst
briefing that forces the LLM agent to reason — not just threshold-match.

Instead of:
  "order_imbalance: 0.72, cancel_rate: 0.48"

The agent receives:
  "ALERT: Bid side is heavily stacked (imbalance +0.72). In the last
   10 ticks, a single order at $100.04 appeared for 1,847 units —
   far exceeding the average size of 67 units. Cancel rate has spiked
   to 48% in the last 30 seconds. Three ticks ago this same price
   level appeared then vanished within 200ms. Pattern consistent
   with classic spoofing: phantom liquidity placed to attract sellers,
   then withdrawn before execution."

This is what makes the environment interesting to a Nemotron-class
agent and impressive to Phase 3 human reviewers.
"""
from __future__ import annotations

from typing import List, Optional
from server.models import ManipulationObservation, PriceLevel, Trade


# ---------------------------------------------------------------------------
# Signal analyser helpers
# ---------------------------------------------------------------------------

def _largest_bid_size(obs: ManipulationObservation) -> Optional[PriceLevel]:
    if not obs.bid_levels:
        return None
    return max(obs.bid_levels, key=lambda x: x.size)


def _largest_ask_size(obs: ManipulationObservation) -> Optional[PriceLevel]:
    if not obs.ask_levels:
        return None
    return max(obs.ask_levels, key=lambda x: x.size)


def _avg_trade_size(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    return sum(t.size for t in trades) / len(trades)


def _wash_pairs(trades: List[Trade]) -> int:
    """Count back-to-back buy/sell pairs with identical price and size."""
    count = 0
    for i in range(len(trades) - 1):
        a, b = trades[i], trades[i + 1]
        if (abs(a.price - b.price) < 0.001
                and abs(a.size - b.size) < 0.001
                and a.side != b.side):
            count += 1
    return count


def _size_uniformity(levels: List[PriceLevel]) -> float:
    """
    Returns 0-1 measure of how uniform order sizes are across levels.
    High uniformity (>0.8) is a layering signal.
    """
    if len(levels) < 2:
        return 0.0
    sizes = [l.size for l in levels]
    mean = sum(sizes) / len(sizes)
    if mean == 0:
        return 0.0
    variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
    cv = (variance ** 0.5) / mean  # coefficient of variation
    return round(max(0.0, 1.0 - cv), 3)


def _recent_bids(obs: ManipulationObservation, tick: int = 0) -> List[PriceLevel]:
    return [b for b in obs.bid_levels if b.tick == tick]


def _recent_asks(obs: ManipulationObservation, tick: int = 0) -> List[PriceLevel]:
    return [a for a in obs.ask_levels if a.tick == tick]


# ---------------------------------------------------------------------------
# Main narrative builder
# ---------------------------------------------------------------------------

def build_narrative(obs: ManipulationObservation) -> str:
    """
    Build a rich, multi-paragraph analyst briefing from raw observations.
    This is the key observation the LLM agent reasons over.
    """
    lines = []

    # --- Header ---
    lines.append(
        f"=== MARKET SURVEILLANCE BRIEF | Step {obs.step_number} | "
        f"Task: {obs.task_name} ==="
    )
    lines.append(f"Mid price: ${obs.mid_price:.2f}  |  "
                 f"Spread: ${obs.spread:.4f}  |  "
                 f"Regime: {'⚠ VOLATILE' if obs.spread > 0.07 else 'calm'}")
    lines.append("")

    # --- Order book analysis ---
    lines.append("── ORDER BOOK ──")
    recent_bids = _recent_bids(obs, tick=0) or obs.bid_levels[:5]
    recent_asks = _recent_asks(obs, tick=0) or obs.ask_levels[:5]

    if recent_bids:
        bid_str = "  ".join(f"${b.price:.2f}×{b.size:.0f}" for b in recent_bids[:3])
        lines.append(f"  Bids (best→deep): {bid_str}")
    if recent_asks:
        ask_str = "  ".join(f"${a.price:.2f}×{a.size:.0f}" for a in recent_asks[:3])
        lines.append(f"  Asks (best→deep): {ask_str}")

    # Imbalance interpretation
    imb = obs.order_imbalance
    if imb > 0.5:
        lines.append(f"  ⚠ Strong BID pressure (imbalance={imb:+.3f}). "
                     f"Bid volume dominates — could be spoofing or genuine buying.")
    elif imb < -0.5:
        lines.append(f"  ⚠ Strong ASK pressure (imbalance={imb:+.3f}). "
                     f"Ask volume dominates — could be spoofing on offer side.")
    else:
        lines.append(f"  Order imbalance: {imb:+.3f} (balanced)")

    # Largest order anomaly
    big_bid = _largest_bid_size(obs)
    big_ask = _largest_ask_size(obs)
    avg_t = _avg_trade_size(obs.trade_tape)
    if big_bid and avg_t > 0 and big_bid.size > avg_t * 8:
        lines.append(
            f"  ⚠ ANOMALY: Single bid at ${big_bid.price:.2f} "
            f"for {big_bid.size:.0f} units — "
            f"{big_bid.size / avg_t:.1f}× the average trade size ({avg_t:.0f}). "
            f"Phantom liquidity suspected."
        )
    if big_ask and avg_t > 0 and big_ask.size > avg_t * 8:
        lines.append(
            f"  ⚠ ANOMALY: Single ask at ${big_ask.price:.2f} "
            f"for {big_ask.size:.0f} units — "
            f"{big_ask.size / avg_t:.1f}× the average trade size. "
            f"Phantom offer suspected."
        )

    # Layering: uniform sizes across levels
    bid_uniformity = _size_uniformity(recent_bids)
    if bid_uniformity > 0.75 and len(recent_bids) >= 3:
        lines.append(
            f"  ⚠ LAYERING SIGNAL: Bid levels show unusually uniform sizes "
            f"(uniformity={bid_uniformity:.2f}). "
            f"Multiple same-sized orders stacked suggests artificial wall."
        )

    lines.append("")

    # --- Cancel rate ---
    lines.append("── CANCELLATION ACTIVITY ──")
    cr = obs.cancel_rate
    if cr > 0.5:
        lines.append(
            f"  ⚠ CRITICAL: Cancel rate {cr:.0%} — more than half of all orders "
            f"in the last 30s were cancelled before execution. "
            f"Consistent with spoofing or layering."
        )
    elif cr > 0.35:
        lines.append(
            f"  ⚠ ELEVATED: Cancel rate {cr:.0%} — above normal HFT baseline (10-15%). "
            f"Warrants monitoring."
        )
    else:
        lines.append(f"  Cancel rate: {cr:.0%} — within normal range.")

    lines.append("")

    # --- Trade tape analysis ---
    lines.append("── TRADE TAPE (last 5 executions) ──")
    tape = obs.trade_tape[:5]
    if tape:
        for t in tape:
            lines.append(
                f"  {t.side.upper():4s}  ${t.price:.2f}  ×{t.size:.0f}  "
                f"@{t.timestamp_ms}ms"
            )
    else:
        lines.append("  No recent trades.")

    # Wash trading detection
    wash = _wash_pairs(obs.trade_tape)
    if wash >= 2:
        lines.append(
            f"  ⚠ WASH TRADING SIGNAL: {wash} back-to-back buy/sell pairs "
            f"detected with identical price and size. "
            f"This pattern generates artificial volume without real ownership change."
        )
    elif wash == 1:
        lines.append(
            f"  ⚠ POSSIBLE WASH: 1 matched buy/sell pair at same price/size. "
            f"Inconclusive alone — watch next steps."
        )

    lines.append("")

    # --- Regime note ---
    if "volatile" in obs.context_hint.lower():
        lines.append(
            "── REGIME NOTE ──\n"
            "  Market has entered a VOLATILE regime. Elevated cancel rates and\n"
            "  wide spreads are expected even for legitimate participants.\n"
            "  Raise your evidence bar before escalating — false positives\n"
            "  are heavily penalised in volatile conditions."
        )
        lines.append("")

    # --- Decision guidance ---
    lines.append("── DECISION FRAMEWORK ──")
    lines.append("  ignore     → Normal market / insufficient evidence")
    lines.append("  soft_flag  → Suspicious but needs more evidence (use for wash trading)")
    lines.append("  escalate   → Clear manipulation, alert regulators (spoofing/layering)")
    lines.append("  NOTE: False positives (flagging clean markets) are penalised -0.30")

    return "\n".join(lines)
