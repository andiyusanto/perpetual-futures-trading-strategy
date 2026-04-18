"""
Limit order entry management.
Computes maker-friendly entry prices and tracks pending fills.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.core.volatility import VolatilityPhase


class LimitOrderManager:
    """
    Prefers limit (maker) orders over market (taker) orders.

    Expected impact:
      ~70% fill rate (crypto usually retests within 3 bars)
      Save ~3.5 bp per filled trade (5.5 bp taker -> 2.0 bp maker)
      Better average entry by ~0.2xATR
    """

    MAKER_FEE_BPS: float = 2.0
    TAKER_FEE_BPS: float = 5.5

    def compute_limit_price(
        self,
        direction: int,
        current_price: float,
        atr: float,
        vol_phase: VolatilityPhase,
    ) -> float:
        """
        Regime-adaptive limit offset:
          EXPANSION   -> 0.15xATR  (don't miss the move)
          NORMAL      -> 0.25xATR
          COMPRESSION -> 0.40xATR  (likely to retrace; wait for it)
        """
        if vol_phase == VolatilityPhase.EXPANSION:
            mult = 0.15
        elif vol_phase == VolatilityPhase.COMPRESSION:
            mult = 0.40
        else:
            mult = 0.25

        offset = atr * mult
        return current_price - direction * offset

    def simulate_fill(
        self,
        limit_price: float,
        direction: int,
        future_lows: np.ndarray,
        future_highs: np.ndarray,
        max_wait_bars: int = 3,
    ) -> Tuple[bool, int]:
        """
        Backtest helper: check whether a limit would fill in the next N bars.
        Returns (filled, bar_index).
        """
        for i in range(min(max_wait_bars, len(future_lows))):
            if direction > 0 and future_lows[i] <= limit_price:
                return True, i
            if direction < 0 and future_highs[i] >= limit_price:
                return True, i
        return False, -1
