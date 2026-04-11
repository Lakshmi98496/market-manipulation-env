"""
Pydantic models for Market Manipulation Detection environment.
Defines Action, Observation, and StepResult with full type safety.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ManipulationAction(BaseModel):
    """
    Agent's decision for a single order-book event.

    decision:     ignore | soft_flag | escalate
    pattern_type: which manipulation pattern the agent believes it sees
    confidence:   how confident the agent is (0.0 – 1.0)
    """
    decision: str = Field(
        ...,
        description="Agent ruling: 'ignore', 'soft_flag', or 'escalate'",
    )
    pattern_type: str = Field(
        default="none",
        description="Detected pattern: 'spoofing', 'layering', 'wash_trading', or 'none'",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent confidence in its decision (0.0–1.0)",
    )

    @field_validator("decision", mode="before")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        allowed = {"ignore", "soft_flag", "escalate"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"decision must be one of {allowed}, got '{v}'")
        return v

    @field_validator("pattern_type", mode="before")
    @classmethod
    def validate_pattern_type(cls, v: str) -> str:
        allowed = {"spoofing", "layering", "wash_trading", "none"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"pattern_type must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Order Book helpers
# ---------------------------------------------------------------------------

class PriceLevel(BaseModel):
    """A single price/size entry in the order book."""
    price: float
    size: float
    tick: int  # which historical tick (0 = most recent)


class Trade(BaseModel):
    """An individual trade from the tape."""
    price: float
    size: float
    side: str          # "buy" or "sell"
    timestamp_ms: int  # simulated millisecond timestamp


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ManipulationObservation(BaseModel):
    """
    Full state visible to the agent at each step.
    Mirrors a real L2 order book feed with derived signals.
    """
    # Order book snapshot — 5 levels × last 10 ticks
    bid_levels: List[PriceLevel] = Field(
        default_factory=list,
        description="Top-5 bid levels over last 10 ticks (50 entries total)",
    )
    ask_levels: List[PriceLevel] = Field(
        default_factory=list,
        description="Top-5 ask levels over last 10 ticks (50 entries total)",
    )

    # Trade tape — last 20 trades
    trade_tape: List[Trade] = Field(
        default_factory=list,
        description="Last 20 executed trades",
    )

    # Derived signals
    order_imbalance: float = Field(
        default=0.0,
        description="(bid_vol - ask_vol) / (bid_vol + ask_vol), range [-1, 1]",
    )
    cancel_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of orders cancelled in last 30s",
    )
    spread: float = Field(
        default=0.01,
        description="Current bid-ask spread in dollars",
    )
    mid_price: float = Field(
        default=100.0,
        description="Current mid-price in dollars",
    )

    # Episode context
    step_number: int = Field(default=0, description="Current step index")
    task_name: str = Field(default="", description="Active task name")
    context_hint: str = Field(
        default="",
        description="Plain-English summary of observed market behaviour",
    )


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Returned by env.step() after each agent action."""
    observation: ManipulationObservation
    reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised reward for this step [0.0, 1.0]",
    )
    done: bool = Field(default=False)
    info: dict = Field(default_factory=dict)

    # Ground truth (revealed after step for logging; not used in reward calc)
    true_pattern: Optional[str] = Field(
        default=None,
        description="Actual injected pattern, revealed post-step",
    )
    last_action_error: Optional[str] = Field(default=None)
