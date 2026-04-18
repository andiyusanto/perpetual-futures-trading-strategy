"""
Kelly-inspired position sizer with drawdown and volatility scaling.
Reconstructed from usage signatures in BacktestEngineV3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.risk.risk_manager import MarketRegime


class SignalType(Enum):
    STRONG_LONG  = "strong_long"
    LONG         = "long"
    NEUTRAL      = "neutral"
    SHORT        = "short"
    STRONG_SHORT = "strong_short"


@dataclass
class TradeSignal:
    """Minimal signal descriptor fed to PositionSizer."""
    direction: SignalType
    confidence: float
    trend_score: float
    momentum_score: float
    orderflow_score: float
    regime: MarketRegime
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size_pct: float
    timeframe_alignment: float
    edge_quality: str           # A+, A, B, C
    filters_passed: bool


class PositionSizer:
    """
    Fractional Kelly position sizing with four dynamic adjustments:
      1. Edge quality  — A+ gets full size; B gets 60%
      2. Drawdown      — reduces size linearly as DD approaches max
      3. Volatility    — reduces size when current vol > baseline
      4. Regime        — boosts size in trending regimes
    """

    def __init__(self, base_risk_pct: float = 0.01, max_size_pct: float = 0.10) -> None:
        self.base_risk_pct = base_risk_pct
        self.max_size_pct = max_size_pct

    def compute_size(
        self,
        signal: TradeSignal,
        capital: float,
        current_dd_pct: float,
        base_win_rate: float,
        base_rr: float,
        baseline_vol: float,
        current_vol: float,
    ) -> float:
        """Return position size as fraction of capital (e.g. 0.08 = 8%)."""
        # Half-Kelly
        edge = base_win_rate * base_rr - (1 - base_win_rate)
        kelly = edge / base_rr if base_rr > 0 else 0.0
        half_kelly = max(0.0, kelly * 0.5)

        grade_mult = {"A+": 1.0, "A": 0.85, "B": 0.60, "C": 0.30}.get(signal.edge_quality, 0.0)
        dd_mult = max(0.0, 1.0 - current_dd_pct / 20.0)

        vol_ratio = current_vol / (baseline_vol + 1e-10)
        vol_mult = float(np.clip(1.0 / vol_ratio, 0.3, 1.5))

        regime_mult = 1.2 if signal.regime == MarketRegime.TRENDING else 1.0
        conf_mult = float(np.clip(signal.confidence / 0.5, 0.5, 1.5))

        raw = half_kelly * grade_mult * dd_mult * vol_mult * regime_mult * conf_mult
        return float(np.clip(raw, 0.0, self.max_size_pct))
