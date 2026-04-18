"""
Exit management: market-structure stop loss + 4-phase Chandelier trailing stop.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# =============================================================================
# PILLAR 2A — Market-Structure Stop Loss
# =============================================================================


class MarketStructureSL:
    """
    Places SL behind the nearest significant swing point rather than a fixed
    ATR multiple. Typically tighter and more structurally sound — only
    invalidated when market structure actually breaks.

    Add 0.3xATR buffer to avoid exact-level stop hunts.
    Fallback: 1.2xATR if no clean swing is detected.
    """

    def compute_sl(
        self,
        direction: int,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: float,
        lookback: int = 30,
    ) -> float:
        n = len(close)
        price = float(close[-1])

        if n < lookback:
            return price - direction * 1.2 * atr

        if direction > 0:
            recent = low[-lookback:]
            swings = [
                float(recent[i])
                for i in range(2, len(recent) - 1)
                if recent[i] < recent[i-1] and recent[i] < recent[i-2] and recent[i] <= recent[i+1]
            ]
            if swings:
                sl = max(swings) - 0.3 * atr
                return float(np.clip(sl, price - 3.0 * atr, price - 0.5 * atr))
            return price - 1.2 * atr

        else:
            recent = high[-lookback:]
            swings = [
                float(recent[i])
                for i in range(2, len(recent) - 1)
                if recent[i] > recent[i-1] and recent[i] > recent[i-2] and recent[i] >= recent[i+1]
            ]
            if swings:
                sl = min(swings) + 0.3 * atr
                return float(np.clip(sl, price + 0.5 * atr, price + 3.0 * atr))
            return price + 1.2 * atr


# =============================================================================
# PILLAR 2B — 4-Phase Chandelier Trailing Stop
# =============================================================================

_CHANDELIER_PHASES: List[dict] = [
    {"threshold_r": 1.0, "action": "breakeven",  "trail_atr_mult": None},
    {"threshold_r": 1.5, "action": "chandelier", "trail_atr_mult": 2.5},
    {"threshold_r": 2.5, "action": "chandelier", "trail_atr_mult": 1.5},
    {"threshold_r": 4.0, "action": "chandelier", "trail_atr_mult": 1.0},
]


def chandelier_update(
    direction: int,
    entry_price: float,
    initial_risk: float,
    current_high: float,
    current_low: float,
    current_atr: float,
    current_stop: float,
    highest: float,
    lowest: float,
) -> Tuple[float, int]:
    """
    Compute new chandelier stop and current phase index.

    Phase progression (for longs):
      0 → Hold initial SL
      1 → Move to breakeven (+0.2R) after +1.0R unrealised
      2 → Chandelier at 2.5xATR  after +1.5R
      3 → Chandelier at 1.5xATR  after +2.5R
      4 → Chandelier at 1.0xATR  after +4.0R

    Stops can only move in the favourable direction (never widen).
    """
    if direction > 0:
        unrealised_r = (highest - entry_price) / (initial_risk + 1e-10)
    else:
        unrealised_r = (entry_price - lowest) / (initial_risk + 1e-10)

    active = -1
    for i, p in enumerate(_CHANDELIER_PHASES):
        if unrealised_r >= p["threshold_r"]:
            active = i

    if active < 0:
        return current_stop, 0

    phase = _CHANDELIER_PHASES[active]

    if direction > 0:
        if phase["action"] == "breakeven":
            new_sl = entry_price + 0.2 * initial_risk
        else:
            new_sl = highest - phase["trail_atr_mult"] * current_atr  # type: ignore[operator]
        return max(current_stop, new_sl), active + 1
    else:
        if phase["action"] == "breakeven":
            new_sl = entry_price - 0.2 * initial_risk
        else:
            new_sl = lowest + phase["trail_atr_mult"] * current_atr  # type: ignore[operator]
        return min(current_stop, new_sl), active + 1


class ChandelierExit:
    """Object-oriented wrapper around chandelier_update for backtest use."""

    def update_stop(
        self,
        direction: int,
        entry_price: float,
        initial_risk: float,
        current_high: float,
        current_low: float,
        current_atr: float,
        current_stop: float,
        highest_since_entry: float,
        lowest_since_entry: float,
    ) -> Tuple[float, int]:
        return chandelier_update(
            direction, entry_price, initial_risk,
            current_high, current_low, current_atr,
            current_stop, highest_since_entry, lowest_since_entry,
        )
