"""
V3StrategyEngine — the single, shared strategy brain used by both
the BacktestEngineV3 and the async production bot.

Two entry points:
  generate_signal_from_arrays(...)  — batch mode (backtest)
  generate_signal()                 — streaming mode (reads from MarketDataBuffer)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from config.config import StrategyConfig
from src.core.data_buffer import MarketDataBuffer
from src.core.utils import compute_atr, compute_adx
from src.core.volatility import VolatilityClassifier, VolatilityPhase
from src.strategy.exits import MarketStructureSL, chandelier_update
from src.strategy.signals import FundingVelocitySignal, LiquidationMapper, SignalEngineV3


# =============================================================================
# Signal DTO
# =============================================================================


@dataclass
class V3Signal:
    """Fully-described trading signal returned by V3StrategyEngine."""

    direction: Literal["LONG", "SHORT", "NEUTRAL"]
    confidence: float
    grade: str                              # A+, A, B, C, D
    entry_price: float
    limit_entry_price: Optional[float]      # Preferred maker limit price
    stop_loss: float                        # Market-structure SL
    take_profit: float
    initial_risk: float                     # |entry - SL|

    # Component scores (for logging / debugging)
    trend_score: float = 0.0
    momentum_score: float = 0.0
    orderflow_score: float = 0.0
    funding_score: float = 0.0
    composite: float = 0.0

    # Context
    vol_phase: str = "normal"
    squeeze_fired: bool = False
    liq_alignment: float = 0.0
    regime: str = "unknown"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def is_tradeable(self) -> bool:
        return (
            self.direction != "NEUTRAL"
            and self.grade in ("A+", "A", "B")
            and self.confidence > 0.15
        )


# =============================================================================
# Shared Strategy Engine
# =============================================================================


class V3StrategyEngine:
    """
    Single strategy implementation consumed by both backtest and production.

    The three pillars:
      1. Signal quality  — vol-phase gate + 4-signal composite + liq alignment
      2. Trade management — market-structure SL + 4-phase Chandelier exit
      3. Perpetual edge  — limit-order entries + funding carry awareness
    """

    def __init__(self, cfg: Optional[StrategyConfig] = None) -> None:
        self._cfg = cfg or StrategyConfig()

        self._vol_clf = VolatilityClassifier(
            bb_period=self._cfg.bb_period,
            bb_std=self._cfg.bb_std,
            kc_period=self._cfg.kc_period,
            kc_mult=self._cfg.kc_mult,
            lookback=self._cfg.vol_lookback,
        )
        self._sig_engine = SignalEngineV3(
            ema_fast=self._cfg.ema_fast,
            ema_mid=self._cfg.ema_mid,
            ema_slow=self._cfg.ema_slow,
        )
        self._funding_sig = FundingVelocitySignal(
            velocity_period=self._cfg.funding_vel_period,
            acceleration_period=self._cfg.funding_accel_period,
        )
        self._liq_mapper = LiquidationMapper()
        self._structure_sl = MarketStructureSL()

        # Production-mode buffer (None in pure backtest usage)
        self._buffer: Optional[MarketDataBuffer] = None

        # Open-position registry for exit management
        self._positions: Dict[str, dict] = {}

    # ── Production-mode buffer attachment ────────────────────────────────────

    def attach_buffer(self, buf: MarketDataBuffer) -> None:
        self._buffer = buf

    # ── Core signal generation (arrays) ──────────────────────────────────────

    def generate_signal_from_arrays(
        self,
        c: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        o: np.ndarray,
        v: np.ndarray,
        fr: np.ndarray,
        oi: np.ndarray,
        regime_type: str = "normal",
        regime_confidence: float = 0.6,
    ) -> V3Signal:
        """
        Main signal computation path. Called by backtest directly and by
        generate_signal() after pulling from MarketDataBuffer.
        """
        neutral = self._neutral(float(c[-1]) if len(c) > 0 else 0.0, regime_type)
        if len(c) < 120:
            return neutral

        price = float(c[-1])

        # Step 1: Volatility phase gate
        vol_phase, bbw_pctile, squeeze_fired = self._vol_clf.classify(c, h, l)
        if vol_phase == VolatilityPhase.COMPRESSION:
            neutral.vol_phase = "compression"
            return neutral

        # Step 2: Four orthogonal signals
        trend = self._sig_engine.trend_signal(c, h, l, v)
        momentum = self._sig_engine.momentum_signal(c, h, l, regime_type)
        orderflow = self._sig_engine.orderflow_signal(c, o, v)

        fr_signal, _, _ = self._funding_sig.compute(fr) if len(fr) >= 12 else (0.0, 0.0, 0.0)

        # Step 3: Regime-weighted composite
        w = self._sig_engine.get_regime_weights(regime_type)
        composite = (
            w["trend"] * trend
            + w["momentum"] * momentum
            + w["orderflow"] * orderflow
            + w["funding"] * fr_signal
        )

        thr = self._cfg.composite_threshold
        if composite > thr:
            direction = 1
        elif composite < -thr:
            direction = -1
        else:
            neutral.composite = composite
            return neutral

        # Step 4: Liquidation cluster alignment
        oi_safe = oi if len(oi) > 0 else np.ones(len(c))
        clusters = self._liq_mapper.find_clusters(h, l, c, oi_safe)
        liq_score = self._liq_mapper.entry_score(direction, price, clusters)

        # Step 5: Confidence & grading
        confidence = abs(composite) * regime_confidence
        if squeeze_fired:
            confidence *= 1.5
        if liq_score > 0.5:
            confidence *= 1.3

        if confidence > 0.50:
            grade = "A+"
        elif confidence > 0.35:
            grade = "A"
        elif confidence > 0.20:
            grade = "B"
        else:
            grade = "C"

        if grade in ("C", "D"):
            neutral.composite = composite
            return neutral

        # Step 6: Liquidity filter
        vol_ma = float(np.mean(v[-min(self._cfg.volume_ma_bars, len(v)):]))
        if float(np.mean(v[-6:])) < vol_ma * self._cfg.volume_min_ratio:
            return neutral

        # Step 7: Confluence — at least N signals must agree
        signs = [np.sign(trend), np.sign(momentum), np.sign(orderflow), np.sign(fr_signal)]
        if sum(s == direction for s in signs if s != 0) < self._cfg.min_signal_agreement:
            return neutral

        # Step 8: Momentum confirmation (>=1 of last 3 bars in direction)
        if len(c) >= 4:
            last3 = c[-3:]
            ups = sum(1 for j in range(1, 3) if last3[j] > last3[j - 1])
            dns = sum(1 for j in range(1, 3) if last3[j] < last3[j - 1])
            if direction > 0 and ups < 1:
                return neutral
            if direction < 0 and dns < 1:
                return neutral

        # Step 9: Market-structure stop loss
        current_atr = float(np.mean(h[-14:] - l[-14:]))
        sl = self._structure_sl.compute_sl(direction, h, l, c, current_atr)
        initial_risk = abs(price - sl)

        tp_mult = (
            self._cfg.tp_mult_trending if "trending" in regime_type else self._cfg.tp_mult_normal
        )
        tp = price + direction * tp_mult * initial_risk

        # Step 10: Limit order entry price
        if vol_phase == VolatilityPhase.EXPANSION:
            offset_mult = 0.15
        elif vol_phase == VolatilityPhase.COMPRESSION:
            offset_mult = 0.40
        else:
            offset_mult = 0.25
        limit_price = price - direction * current_atr * offset_mult

        dir_str: Literal["LONG", "SHORT"] = "LONG" if direction > 0 else "SHORT"

        return V3Signal(
            direction=dir_str,
            confidence=confidence,
            grade=grade,
            entry_price=price,
            limit_entry_price=limit_price,
            stop_loss=sl,
            take_profit=tp,
            initial_risk=initial_risk,
            trend_score=trend,
            momentum_score=momentum,
            orderflow_score=orderflow,
            funding_score=fr_signal,
            composite=composite,
            vol_phase=vol_phase.value,
            squeeze_fired=squeeze_fired,
            liq_alignment=liq_score,
            regime=regime_type,
        )

    # ── Production streaming path ─────────────────────────────────────────────

    def generate_signal(self) -> V3Signal:
        """Pull from attached MarketDataBuffer and generate signal."""
        if self._buffer is None or not self._buffer.ready:
            return self._neutral(0.0)

        c = self._buffer.close.as_array()
        h = self._buffer.high.as_array()
        l = self._buffer.low.as_array()
        o = self._buffer.open.as_array()
        v = self._buffer.volume.as_array()
        fr = self._buffer.funding_rate.as_array()
        oi = self._buffer.open_interest.as_array()

        # Lightweight regime classification from ADX
        adx, plus_di, minus_di = compute_adx(h, l, c)
        adx_now = float(adx[-1])
        if adx_now > 25:
            regime = "trending_bull" if plus_di[-1] > minus_di[-1] else "trending_bear"
            reg_conf = min(0.9, adx_now / 50)
        elif adx_now < 20:
            regime = "mean_reverting"
            reg_conf = min(0.85, (25 - adx_now) / 25)
        else:
            regime = "normal"
            reg_conf = 0.6

        return self.generate_signal_from_arrays(c, h, l, o, v, fr, oi, regime, reg_conf)

    # ── Exit management ───────────────────────────────────────────────────────

    def register_position(
        self,
        trade_id: str,
        direction: int,
        entry_price: float,
        initial_risk: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        self._positions[trade_id] = {
            "direction": direction,
            "entry_price": entry_price,
            "initial_risk": initial_risk,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "highest": entry_price,
            "lowest": entry_price,
            "trail_phase": 0,
        }

    def update_exit(
        self, trade_id: str, current_high: float, current_low: float
    ) -> Optional[dict]:
        """Update chandelier stop and return exit dict if triggered, else None."""
        pos = self._positions.get(trade_id)
        if pos is None:
            return None

        pos["highest"] = max(pos["highest"], current_high)
        pos["lowest"] = min(pos["lowest"], current_low)

        current_atr = 0.0
        if self._buffer is not None and len(self._buffer.high) >= 14:
            h = self._buffer.high.as_array()
            l = self._buffer.low.as_array()
            current_atr = float(np.mean(h[-14:] - l[-14:]))

        new_stop, phase = chandelier_update(
            pos["direction"], pos["entry_price"], pos["initial_risk"],
            current_high, current_low, current_atr,
            pos["stop_loss"], pos["highest"], pos["lowest"],
        )
        pos["stop_loss"] = new_stop
        pos["trail_phase"] = phase

        hit_sl = (
            (pos["direction"] > 0 and current_low <= pos["stop_loss"])
            or (pos["direction"] < 0 and current_high >= pos["stop_loss"])
        )
        hit_tp = (
            (pos["direction"] > 0 and current_high >= pos["take_profit"])
            or (pos["direction"] < 0 and current_low <= pos["take_profit"])
        )

        if hit_sl or hit_tp:
            result = {
                "trade_id": trade_id,
                "reason": (
                    f"trail_phase_{phase}" if hit_sl and phase > 0
                    else "stop_loss" if hit_sl
                    else "take_profit"
                ),
                "exit_price": pos["stop_loss"] if hit_sl else pos["take_profit"],
                "phase": phase,
                "use_maker": not hit_sl,
            }
            del self._positions[trade_id]
            return result
        return None

    def get_current_stop(self, trade_id: str) -> Optional[float]:
        pos = self._positions.get(trade_id)
        return float(pos["stop_loss"]) if pos else None

    @staticmethod
    def _neutral(price: float, regime: str = "unknown") -> V3Signal:
        return V3Signal(
            direction="NEUTRAL", confidence=0.0, grade="D",
            entry_price=price, limit_entry_price=None,
            stop_loss=0.0, take_profit=0.0, initial_risk=0.0,
            regime=regime,
        )
