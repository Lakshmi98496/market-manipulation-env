"""
Order Book Simulator
====================
Generates realistic L2 order book snapshots and injects manipulation
patterns (spoofing, layering, wash trading) for the agent to detect.

Designed to be deterministic given a seed (for reproducibility) while
supporting an adaptive adversary mode for the hard task.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from server.models import ManipulationObservation, PriceLevel, Trade
from server.data_loader import RealDataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_PRICE = 100.0
TICK_SIZE = 0.01
NUM_LEVELS = 5
HISTORY_TICKS = 10
TAPE_LENGTH = 20


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

@dataclass
class MarketRegime:
    """Controls background market noise parameters."""
    volatility: float = 0.05       # price std per step
    base_spread: float = 0.04      # normal bid-ask spread
    hft_cancel_rate: float = 0.15  # legitimate cancel rate
    name: str = "calm"


CALM_REGIME = MarketRegime(volatility=0.03, base_spread=0.03,
                            hft_cancel_rate=0.10, name="calm")
VOLATILE_REGIME = MarketRegime(volatility=0.12, base_spread=0.08,
                                hft_cancel_rate=0.30, name="volatile")


# ---------------------------------------------------------------------------
# Pattern injector
# ---------------------------------------------------------------------------

class PatternInjector:
    """
    Injects a manipulation pattern into a given order book state.
    Returns modified bid/ask levels and a ground-truth label.
    """

    @staticmethod
    def inject_spoofing(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        rng: random.Random,
        step: int,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], str]:
        """
        Spoofing: large phantom order on one side to push price,
        then cancel before fill. Appears as a sudden large bid or ask.
        """
        phantom_size = rng.uniform(800, 2000)
        side = "bid" if step % 3 != 0 else "ask"

        if side == "bid":
            # Insert a huge phantom bid just below best ask
            phantom_price = bids[0][0] + rng.uniform(0.01, 0.04)
            bids = [(phantom_price, phantom_size)] + bids[:4]
        else:
            phantom_price = asks[0][0] - rng.uniform(0.01, 0.04)
            asks = [(phantom_price, phantom_size)] + asks[:4]

        return bids, asks, "spoofing"

    @staticmethod
    def inject_layering(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        rng: random.Random,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], str]:
        """
        Layering: multiple orders stacked at successive price levels on
        one side, creating an artificial wall to move price.
        """
        side = rng.choice(["bid", "ask"])
        layer_sizes = [rng.uniform(300, 700) for _ in range(5)]

        if side == "bid":
            bids = [(bids[i][0] + 0.01 * i, layer_sizes[i]) for i in range(5)]
        else:
            asks = [(asks[i][0] - 0.01 * i, layer_sizes[i]) for i in range(5)]

        return bids, asks, "layering"

    @staticmethod
    def inject_wash_trading(
        tape: List[Tuple[float, float, str, int]],
        rng: random.Random,
        mid_price: float,
        base_ts: int,
    ) -> Tuple[List[Tuple[float, float, str, int]], str]:
        """
        Wash trading: self-matched buy+sell pairs at same price to
        create artificial volume. Appears as back-to-back B/S of
        identical size at the same price.
        """
        wash_price = round(mid_price + rng.uniform(-0.02, 0.02), 2)
        wash_size = round(rng.uniform(50, 200), 0)
        ts = base_ts

        # Insert pairs at the front of the tape
        wash_trades = [
            (wash_price, wash_size, "buy", ts),
            (wash_price, wash_size, "sell", ts + 1),
            (wash_price, wash_size, "buy", ts + 2),
            (wash_price, wash_size, "sell", ts + 3),
        ]
        return wash_trades + tape[:16], "wash_trading"


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class OrderBookSimulator:
    """
    Simulates a live order book for one episode.

    task_name controls which patterns are injected:
        spoofing_detection        → easy   (spoofing only)
        layering_wash_detection   → medium (layering + wash trading)
        adaptive_adversary_detection → hard (adaptive + regime shift)
    """

    def __init__(self, task_name: str, seed: int = 42):
        self.task_name = task_name
        self.rng = random.Random(seed)
        self.step_count = 0
        self.mid_price = BASE_PRICE
        self.regime = CALM_REGIME
        self.regime_switched = False

        # Adaptive adversary state (hard task)
        self.agent_flag_history: List[str] = []
        self.adversary_pattern: str = "spoofing"

        # Real data loader (used for hard task)
        self._data_loader = RealDataLoader(seed=seed)
        self._use_real_data = (
            task_name == "adaptive_adversary_detection"
            and self._data_loader.available()
        )

        # Running tape
        self._tape: List[Tuple[float, float, str, int]] = []
        self._tick_history: List[Tuple[List, List]] = []

        self._warm_up()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> ManipulationObservation:
        if seed is not None:
            self.rng = random.Random(seed)
        self.step_count = 0
        self.mid_price = BASE_PRICE
        self.regime = CALM_REGIME
        self.regime_switched = False
        self.agent_flag_history = []
        self.adversary_pattern = "spoofing"
        self._tape = []
        self._tick_history = []
        if hasattr(self, '_data_loader'):
            self._data_loader.reset(seed=seed)
        self._warm_up()
        return self._build_observation(true_pattern=None)

    def step(self, agent_decision: str = "ignore") -> Tuple[ManipulationObservation, float, bool, str]:
        """
        Advance the simulation one step.

        Returns: (observation, reward_components_dict, done, true_pattern)
        """
        self.step_count += 1

        # Possibly switch regime mid-episode (hard task, step 12)
        if (self.task_name == "adaptive_adversary_detection"
                and self.step_count == 12
                and not self.regime_switched):
            self.regime = VOLATILE_REGIME
            self.regime_switched = True

        # Update price — use real data tick for hard task if available
        real_tick = None
        if getattr(self, '_use_real_data', False):
            real_tick = self._data_loader.get_signals()

        if real_tick:
            self.mid_price = real_tick["mid_price"]
        else:
            self.mid_price += self.rng.gauss(0, self.regime.volatility)
            self.mid_price = max(self.mid_price, 50.0)

        # Build base book
        bids, asks = self._generate_base_book()

        # Inject pattern based on task
        true_pattern = "none"
        inject_prob = self._injection_probability()

        # For hard task: use real data label if available
        if real_tick and real_tick.get("true_label") not in (None, ""):
            true_pattern = real_tick["true_label"]
            if true_pattern != "none":
                bids, asks, _ = self._inject_for_task(bids, asks)
        elif self.rng.random() < inject_prob:
            bids, asks, true_pattern = self._inject_for_task(bids, asks)

        # Build trade tape
        tape = self._generate_tape(true_pattern)

        # Store tick history
        self._tick_history.append((bids[:], asks[:]))
        if len(self._tick_history) > HISTORY_TICKS:
            self._tick_history.pop(0)

        obs = self._build_observation(true_pattern=true_pattern)

        # Record agent decision for adaptive adversary
        self.agent_flag_history.append(agent_decision)

        return obs, true_pattern

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warm_up(self):
        """Pre-populate history before episode starts."""
        for _ in range(HISTORY_TICKS):
            bids, asks = self._generate_base_book()
            self._tick_history.append((bids, asks))
            self._tape.extend(self._generate_tape("none")[:2])
        self._tape = self._tape[-TAPE_LENGTH:]

    def _injection_probability(self) -> float:
        if self.task_name == "spoofing_detection":
            return 0.6
        elif self.task_name == "layering_wash_detection":
            return 0.7
        else:  # adaptive hard
            # Adversary learns: if agent flags a lot, go quiet; if agent ignores, attack
            recent = self.agent_flag_history[-5:] if self.agent_flag_history else []
            flag_rate = sum(1 for d in recent if d in ("soft_flag", "escalate")) / max(len(recent), 1)
            return 0.3 if flag_rate > 0.6 else 0.75

    def _inject_for_task(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Tuple[List, List, str]:
        if self.task_name == "spoofing_detection":
            return PatternInjector.inject_spoofing(bids, asks, self.rng, self.step_count)

        elif self.task_name == "layering_wash_detection":
            # Alternate between layering and wash trading
            if self.step_count % 2 == 0:
                return PatternInjector.inject_layering(bids, asks, self.rng)
            else:
                tape, pattern = PatternInjector.inject_wash_trading(
                    self._tape, self.rng, self.mid_price,
                    base_ts=self.step_count * 1000
                )
                self._tape = tape
                return bids, asks, pattern

        else:  # adaptive_adversary_detection
            # Adaptive: change pattern based on what got flagged last
            recent_flags = self.agent_flag_history[-3:]
            if "escalate" in recent_flags and self.adversary_pattern == "spoofing":
                self.adversary_pattern = "layering"
            elif "escalate" in recent_flags and self.adversary_pattern == "layering":
                self.adversary_pattern = "wash_trading"

            if self.adversary_pattern == "spoofing":
                return PatternInjector.inject_spoofing(bids, asks, self.rng, self.step_count)
            elif self.adversary_pattern == "layering":
                return PatternInjector.inject_layering(bids, asks, self.rng)
            else:
                tape, pattern = PatternInjector.inject_wash_trading(
                    self._tape, self.rng, self.mid_price,
                    base_ts=self.step_count * 1000
                )
                self._tape = tape
                return bids, asks, pattern

    def _generate_base_book(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        half_spread = self.regime.base_spread / 2
        bids = [
            (
                round(self.mid_price - half_spread - i * TICK_SIZE, 2),
                round(self.rng.uniform(10, 150), 1),
            )
            for i in range(NUM_LEVELS)
        ]
        asks = [
            (
                round(self.mid_price + half_spread + i * TICK_SIZE, 2),
                round(self.rng.uniform(10, 150), 1),
            )
            for i in range(NUM_LEVELS)
        ]
        return bids, asks

    def _generate_tape(self, true_pattern: str) -> List[Tuple[float, float, str, int]]:
        trades = []
        n = self.rng.randint(1, 4)
        for i in range(n):
            price = round(self.mid_price + self.rng.uniform(-0.02, 0.02), 2)
            size = round(self.rng.uniform(5, 80), 1)
            side = self.rng.choice(["buy", "sell"])
            ts = self.step_count * 1000 + i
            trades.append((price, size, side, ts))
        self._tape = (trades + self._tape)[:TAPE_LENGTH]
        return self._tape

    def _compute_signals(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Tuple[float, float, float]:
        bid_vol = sum(s for _, s in bids)
        ask_vol = sum(s for _, s in asks)
        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

        # Cancel rate elevated for manipulation patterns
        base_cancel = self.regime.hft_cancel_rate
        cancel_rate = min(base_cancel + self.rng.uniform(0, 0.1), 1.0)

        spread = asks[0][0] - bids[0][0] if asks and bids else self.regime.base_spread
        return imbalance, cancel_rate, spread

    def _build_observation(self, true_pattern: Optional[str]) -> ManipulationObservation:
        if not self._tick_history:
            bids_now, asks_now = self._generate_base_book()
        else:
            bids_now, asks_now = self._tick_history[-1]

        # Flatten history into PriceLevel list
        bid_levels = []
        ask_levels = []
        for tick_idx, (bids_t, asks_t) in enumerate(self._tick_history[-HISTORY_TICKS:]):
            for price, size in bids_t[:NUM_LEVELS]:
                bid_levels.append(PriceLevel(price=price, size=size, tick=tick_idx))
            for price, size in asks_t[:NUM_LEVELS]:
                ask_levels.append(PriceLevel(price=price, size=size, tick=tick_idx))

        trade_tape = [
            Trade(price=p, size=s, side=sd, timestamp_ms=ts)
            for p, s, sd, ts in self._tape[:TAPE_LENGTH]
        ]

        imbalance, cancel_rate, spread = self._compute_signals(bids_now, asks_now)

        hint = self._build_hint(imbalance, cancel_rate, spread)

        return ManipulationObservation(
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            trade_tape=trade_tape,
            order_imbalance=round(imbalance, 4),
            cancel_rate=round(cancel_rate, 4),
            spread=round(spread, 4),
            mid_price=round(self.mid_price, 4),
            step_number=self.step_count,
            task_name=self.task_name,
            context_hint=hint,
        )

    def _build_hint(self, imbalance: float, cancel_rate: float, spread: float) -> str:
        parts = []
        if abs(imbalance) > 0.4:
            side = "bid" if imbalance > 0 else "ask"
            parts.append(f"strong {side}-side imbalance ({imbalance:+.2f})")
        if cancel_rate > 0.35:
            parts.append(f"elevated cancel rate ({cancel_rate:.0%})")
        if spread < 0.02:
            parts.append("unusually tight spread")
        if spread > 0.10:
            parts.append(f"wide spread (${spread:.3f})")

        # Wash trading hint: repeated same-size trades
        sizes = [t.size for t in []]  # placeholder; real hint from tape
        if not parts:
            parts.append("normal market conditions")

        regime_note = f" [regime: {self.regime.name}]" if self.regime.name != "calm" else ""
        return "Observed: " + "; ".join(parts) + regime_note
