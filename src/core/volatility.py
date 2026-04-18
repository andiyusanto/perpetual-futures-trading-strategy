"""
Volatility regime classification (TTM Squeeze / Bollinger-Keltner method).
Single canonical implementation shared by backtest and production.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd


class VolatilityPhase(Enum):
    COMPRESSION = "compression"   # BB inside KC — avoid trading
    EXPANSION = "expansion"       # BB just exited KC — high-probability entries
    NORMAL = "normal"             # Standard volatility environment
    EXTREME = "extreme"           # Vol spike — widen stops or skip


class VolatilityClassifier:
    """
    Differentiates compression from expansion via Bollinger Band Width
    relative to Keltner Channels (TTM Squeeze concept).

    The key insight: volatility is mean-reverting — low-vol compression
    eventually fires into a directional expansion. We skip compression
    (whipsaw / death-by-1000-cuts) and enter on the expansion.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        lookback: int = 120,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.lookback = lookback

    def classify(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Tuple[VolatilityPhase, float, bool]:
        """
        Returns (phase, bbw_percentile, squeeze_just_fired).

        squeeze_just_fired = True when BB exits KC within the last 5 bars —
        the highest-probability entry window for momentum trades.
        """
        n = len(close)
        if n < self.lookback:
            return VolatilityPhase.NORMAL, 50.0, False

        # Bollinger Bands
        sma = pd.Series(close).rolling(self.bb_period).mean().values
        std = pd.Series(close).rolling(self.bb_period).std().values
        bb_upper = sma + self.bb_std * std
        bb_lower = sma - self.bb_std * std
        bbw = (bb_upper - bb_lower) / (sma + 1e-10)

        valid_bbw = bbw[-self.lookback:]
        valid_bbw = valid_bbw[~np.isnan(valid_bbw)]
        if len(valid_bbw) < 20:
            return VolatilityPhase.NORMAL, 50.0, False

        bbw_pctile = float(np.sum(valid_bbw < bbw[-1]) / len(valid_bbw) * 100)

        # Keltner Channels
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1)),
            ),
        )
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=self.kc_period, adjust=False).mean().values
        kc_upper = sma + self.kc_mult * atr
        kc_lower = sma - self.kc_mult * atr

        # Squeeze = BB fully inside KC
        squeeze = np.zeros(n, dtype=bool)
        for j in range(self.bb_period, n):
            if not np.isnan(bb_upper[j]) and not np.isnan(kc_upper[j]):
                squeeze[j] = bb_upper[j] < kc_upper[j] and bb_lower[j] > kc_lower[j]

        squeeze_fired = bool(n >= 5 and any(squeeze[-5:-1]) and not squeeze[-1])

        if bbw_pctile > 90:
            phase = VolatilityPhase.EXTREME
        elif squeeze[-1] or bbw_pctile < 20:
            phase = VolatilityPhase.COMPRESSION
        elif squeeze_fired or 50 < bbw_pctile < 90:
            phase = VolatilityPhase.EXPANSION
        else:
            phase = VolatilityPhase.NORMAL

        return phase, bbw_pctile, squeeze_fired
