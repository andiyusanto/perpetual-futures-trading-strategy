"""
Market regime classification and risk-of-ruin calculation.
RegimeClassifier reconstructed from usage in BacktestEngineV3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.core.utils import compute_adx, compute_atr


class MarketRegime(Enum):
    TRENDING    = "trending"
    MEAN_REVERT = "mean_reverting"
    HIGH_VOL    = "high_vol"
    LOW_LIQ     = "low_liq"


@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    adx: float = 0.0
    atr_pct: float = 0.0


class RegimeClassifier:
    """
    Classifies the current market regime using ADX + ATR percentile.

    ADX > 25  -> TRENDING
    ADX < 20  -> MEAN_REVERT
    ATR > 85th percentile -> HIGH_VOL
    Volume < 15% of mean  -> LOW_LIQ
    """

    def __init__(self, adx_period: int = 14, atr_lookback: int = 120) -> None:
        self.adx_period = adx_period
        self.atr_lookback = atr_lookback

    def classify(self, df: pd.DataFrame) -> RegimeState:
        """Classify regime from a OHLCV DataFrame slice."""
        if len(df) < max(self.adx_period * 2, 30):
            return RegimeState(regime=MarketRegime.TRENDING, confidence=0.5)

        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        v = df.get("volume", pd.Series(np.ones(len(df)))).values if "volume" in df.columns else np.ones(len(df))

        adx, plus_di, minus_di = compute_adx(h, l, c, self.adx_period)
        atr_arr = compute_atr(h, l, c, self.adx_period)

        current_adx = float(adx[-1])
        current_atr = float(atr_arr[-1])
        atr_pct_val = current_atr / (float(c[-1]) + 1e-10) * 100

        recent_atrs = (
            atr_arr[-self.atr_lookback:] / (c[-self.atr_lookback:] + 1e-10) * 100
        )
        atr_percentile = float(np.sum(recent_atrs < atr_pct_val) / len(recent_atrs) * 100)

        vol_ma = float(np.mean(v[-min(720, len(v)):]))
        low_liq = float(np.mean(v[-6:])) < vol_ma * 0.15

        if low_liq:
            return RegimeState(
                regime=MarketRegime.LOW_LIQ, confidence=0.7,
                adx=current_adx, atr_pct=atr_pct_val,
            )

        if atr_percentile > 85:
            return RegimeState(
                regime=MarketRegime.HIGH_VOL,
                confidence=min(0.9, atr_percentile / 100),
                adx=current_adx, atr_pct=atr_pct_val,
            )

        if current_adx > 25:
            regime = MarketRegime.TRENDING
            confidence = min(0.9, current_adx / 50)
        elif current_adx < 20:
            regime = MarketRegime.MEAN_REVERT
            confidence = min(0.85, (25 - current_adx) / 25)
        else:
            regime = MarketRegime.TRENDING
            confidence = 0.55

        return RegimeState(
            regime=regime, confidence=confidence,
            adx=current_adx, atr_pct=atr_pct_val,
        )


class RiskManager:
    """Portfolio-level risk checks and analytical utilities."""

    @staticmethod
    def risk_of_ruin(
        win_rate: float,
        rr: float,
        risk_per_trade: float,
        ruin_fraction: float,
    ) -> float:
        """
        Classic gambler's ruin approximation for fixed-fraction betting.

        Uses: RoR ~= ((1-edge)/(1+edge))^(Z/a)
        where edge = win_rate * rr - (1 - win_rate),
              Z    = ruin_fraction, a = risk_per_trade.
        """
        if win_rate <= 0 or rr <= 0 or risk_per_trade <= 0:
            return 1.0

        edge = win_rate * rr - (1.0 - win_rate)
        if edge <= 0:
            return 1.0

        edge_norm = edge / (1.0 + edge + 1e-10)
        n_units = ruin_fraction / risk_per_trade

        try:
            ror = ((1.0 - edge_norm) / (1.0 + edge_norm + 1e-10)) ** n_units
            return float(np.clip(ror, 0.0, 1.0))
        except (OverflowError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def max_position_size(
        capital: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = 0.01,
    ) -> float:
        """Return max position size in base currency units."""
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0.0
        return capital * risk_pct / risk_per_unit

    @staticmethod
    def check_drawdown(
        current_capital: float, peak_capital: float, max_dd_pct: float
    ) -> bool:
        """True if current drawdown is within the allowed limit."""
        dd = (peak_capital - current_capital) / peak_capital * 100
        return dd < max_dd_pct
