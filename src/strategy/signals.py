"""
Three of the four orthogonal signals plus the liquidation cluster detector.

Signal roster:
  1. Trend        — EMA stack + market structure + VWAP
  2. Momentum     — RSI + MACD histogram acceleration + price velocity
  3. Orderflow    — CVD divergence + buy/sell volume imbalance
  4. Funding Vel  — rate of change of funding rate (crowding indicator)

Each signal returns a float on [-1, +1].
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.core.utils import ema, compute_rsi
from src.core.volatility import VolatilityPhase


# =============================================================================
# PILLAR 1B — Funding Rate Velocity
# =============================================================================


class FundingVelocitySignal:
    """
    Funding rate velocity is orthogonal to price-based signals because it is
    derived from the basis (perp vs spot), not from price action directly.

    Counter-trend logic:
      - FR rising fast + positive  → longs crowded → bearish signal
      - FR falling fast + negative → shorts crowded → bullish signal
      - FR velocity decelerating   → crowding easing → trend may resume
    """

    def __init__(self, velocity_period: int = 8, acceleration_period: int = 4) -> None:
        self.velocity_period = velocity_period
        self.acceleration_period = acceleration_period

    def compute(self, funding_rates: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (signal in [-1,1], velocity, acceleration).
        Positive signal = bullish; negative = bearish.
        """
        n = len(funding_rates)
        min_len = self.velocity_period + self.acceleration_period + 2
        if n < min_len:
            return 0.0, 0.0, 0.0

        fr_now = funding_rates[-1]
        velocity = (fr_now - funding_rates[-(self.velocity_period + 1)]) / self.velocity_period

        fr_mid = funding_rates[-(self.acceleration_period + 1)]
        fr_early = funding_rates[-(self.velocity_period + self.acceleration_period + 1)]
        vel_prior = (fr_mid - fr_early) / self.velocity_period
        acceleration = (velocity - vel_prior) / self.acceleration_period

        vel_history = np.diff(funding_rates[-min(100, n):])
        vel_std = float(np.std(vel_history)) + 1e-10
        vel_norm = velocity / vel_std

        if fr_now > 0.001 and vel_norm > 1.0:
            signal = -float(np.clip(vel_norm * 0.3, 0, 1))
        elif fr_now < -0.001 and vel_norm < -1.0:
            signal = float(np.clip(abs(vel_norm) * 0.3, 0, 1))
        elif abs(fr_now) > 0.002 and abs(vel_norm) < 0.5:
            signal = 0.2 if fr_now > 0 else -0.2
        else:
            signal = float(np.clip(vel_norm * 0.1, -0.3, 0.3))

        return signal, velocity, acceleration


# =============================================================================
# PILLAR 1C — Liquidation Cluster Detection
# =============================================================================


class LiquidationMapper:
    """
    Estimates liquidation cluster locations from swing points + OI changes.

    Entry alignment logic:
      LONG  → want a short-liq cluster above (cascade will pull price up)
      SHORT → want a long-liq cluster below (cascade will push price down)
    """

    def find_clusters(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        oi: np.ndarray,
        lookback: int = 50,
    ) -> Dict[str, object]:
        n = len(close)
        price = float(close[-1])

        if n < lookback:
            return {
                "long_liq": [],
                "short_liq": [],
                "nearest_long": price * 0.95,
                "nearest_short": price * 1.05,
                "long_density": 0.0,
                "short_density": 0.0,
            }

        rh = high[-lookback:]
        rl = low[-lookback:]

        swing_highs: list[float] = []
        swing_lows: list[float] = []
        for i in range(2, len(rh) - 2):
            if rh[i] > rh[i-1] and rh[i] > rh[i-2] and rh[i] > rh[i+1] and rh[i] > rh[i+2]:
                swing_highs.append(float(rh[i]))
            if rl[i] < rl[i-1] and rl[i] < rl[i-2] and rl[i] < rl[i+1] and rl[i] < rl[i+2]:
                swing_lows.append(float(rl[i]))

        long_liqs = [s for s in swing_lows if s < price] + [s * 0.98 for s in swing_lows if s < price]
        short_liqs = [s for s in swing_highs if s > price] + [s * 1.02 for s in swing_highs if s > price]

        nearest_long = float(max(long_liqs)) if long_liqs else price * 0.95
        nearest_short = float(min(short_liqs)) if short_liqs else price * 1.05

        oi_window = oi[-min(lookback, len(oi)):] if len(oi) > 1 else oi
        oi_chg = np.diff(oi_window) if len(oi_window) > 1 else np.array([0.0])
        density = float(np.sum(np.abs(oi_chg))) * 0.5

        return {
            "long_liq": long_liqs,
            "short_liq": short_liqs,
            "nearest_long": nearest_long,
            "nearest_short": nearest_short,
            "long_density": density,
            "short_density": density,
        }

    def entry_score(self, direction: int, price: float, clusters: Dict[str, object]) -> float:
        """Score in [0, 1] — how well the entry aligns with a nearby liq cluster."""
        if direction > 0:
            dist = (float(clusters["nearest_short"]) - price) / price  # type: ignore[arg-type]
        else:
            dist = (price - float(clusters["nearest_long"])) / price  # type: ignore[arg-type]

        if dist < 0.01:
            return 0.8
        elif dist < 0.03:
            return 0.5
        elif dist < 0.05:
            return 0.2
        return 0.0


# =============================================================================
# v3 Signal Engine (Trend, Momentum, Orderflow)
# =============================================================================


class SignalEngineV3:
    """
    Computes the three price-based signals plus regime-specific weight lookup.
    The fourth signal (FundingVelocity) is computed by FundingVelocitySignal.
    """

    def __init__(
        self,
        ema_fast: int = 8,
        ema_mid: int = 21,
        ema_slow: int = 55,
        rsi_period: int = 14,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period

    def trend_signal(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> float:
        """EMA stack (40%) + market structure (35%) + VWAP (25%)."""
        e8 = ema(close, self.ema_fast)
        e21 = ema(close, self.ema_mid)
        e55 = ema(close, self.ema_slow)

        if e8[-1] > e21[-1] > e55[-1]:
            ema_score = 1.0
        elif e8[-1] < e21[-1] < e55[-1]:
            ema_score = -1.0
        elif e8[-1] > e21[-1]:
            ema_score = 0.3
        elif e8[-1] < e21[-1]:
            ema_score = -0.3
        else:
            ema_score = 0.0

        ms_score = 0.0
        lb = 20
        n = len(high)
        if n >= lb * 2:
            rh, ph = high[-lb:].max(), high[-lb * 2:-lb].max()
            rl, pl = low[-lb:].min(), low[-lb * 2:-lb].min()
            if rh > ph and rl > pl:
                ms_score = 1.0
            elif rh < ph and rl < pl:
                ms_score = -1.0

        vwap = np.cumsum(close * volume) / (np.cumsum(volume) + 1e-10)
        vwap_score = 1.0 if close[-1] > vwap[-1] else -1.0

        return float(np.clip(0.40 * ema_score + 0.35 * ms_score + 0.25 * vwap_score, -1, 1))

    def momentum_signal(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        regime_type: str,
    ) -> float:
        """RSI (35%) + MACD histogram acceleration (35%) + price velocity (30%)."""
        rsi = compute_rsi(close, self.rsi_period)
        macd_line = ema(close, 12) - ema(close, 26)
        macd_sig = ema(macd_line, 9)
        hist = macd_line - macd_sig

        if "bull" in regime_type:
            ob, os_ = 80, 40
        elif "bear" in regime_type:
            ob, os_ = 60, 20
        else:
            ob, os_ = 70, 30

        rv = float(rsi[-1])
        if rv > ob:
            rsi_score = -1.0
        elif rv < os_:
            rsi_score = 1.0
        else:
            rsi_score = 1.0 - 2.0 * (rv - os_) / (ob - os_)

        hist_accel = 0.0
        if len(hist) >= 3:
            hist_accel = float(np.clip(
                (hist[-1] - hist[-2]) / (np.std(hist[-50:]) + 1e-10), -1, 1
            ))

        velocity = 0.0
        if len(close) > 10:
            roc = (close[-1] - close[-10]) / close[-10]
            atr_val = float(np.mean(high[-14:] - low[-14:]))
            velocity = float(np.clip(roc / (atr_val / close[-1] + 1e-10) / 5, -1, 1))

        return float(np.clip(0.35 * rsi_score + 0.35 * hist_accel + 0.30 * velocity, -1, 1))

    def orderflow_signal(
        self,
        close: np.ndarray,
        open_prices: np.ndarray,
        volume: np.ndarray,
    ) -> float:
        """CVD divergence (60%) + buy/sell volume imbalance (40%)."""
        buy_vol = np.where(close > open_prices, volume, volume * 0.4)
        sell_vol = np.where(close <= open_prices, volume, volume * 0.4)
        cvd = np.cumsum(buy_vol - sell_vol)

        cvd_score = 0.0
        if len(close) >= 20:
            price_trend = (close[-1] - close[-20]) / close[-20]
            cvd_trend = (cvd[-1] - cvd[-20]) / (abs(float(cvd[-20])) + 1e-10)
            if price_trend > 0 and cvd_trend < -0.1:
                cvd_score = -0.7
            elif price_trend < 0 and cvd_trend > 0.1:
                cvd_score = 0.7
            else:
                cvd_score = float(np.clip(cvd_trend, -1, 1)) * 0.5

        rb = float(np.sum(buy_vol[-10:]))
        rs = float(np.sum(sell_vol[-10:]))
        imb = float(np.clip((rb - rs) / (rb + rs + 1e-10) * 2, -1, 1))

        return float(np.clip(0.6 * cvd_score + 0.4 * imb, -1, 1))

    def get_regime_weights(self, regime_type: str) -> Dict[str, float]:
        """Four-signal weight lookup keyed by regime name substring."""
        if "trending" in regime_type:
            return {"trend": 0.40, "momentum": 0.25, "orderflow": 0.15, "funding": 0.20}
        elif "mean" in regime_type:
            return {"trend": 0.10, "momentum": 0.35, "orderflow": 0.30, "funding": 0.25}
        elif "high_vol" in regime_type or "volatility" in regime_type:
            return {"trend": 0.20, "momentum": 0.20, "orderflow": 0.35, "funding": 0.25}
        else:
            return {"trend": 0.25, "momentum": 0.25, "orderflow": 0.30, "funding": 0.20}
