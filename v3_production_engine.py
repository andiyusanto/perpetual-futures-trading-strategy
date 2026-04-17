"""
===============================================================================
APFTS v3 — PRODUCTION STRATEGY ENGINE
===============================================================================
Bridges the v3 research engine (v3_deep_alpha.py) into the production async
bot framework. This replaces the boilerplate's placeholder StrategyEngine.

Architecture:
  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
  │  WebSocket   │────>│  DataBuffer  │────>│  V3 Strategy │
  │  Trades/OB   │     │  Ring Buffers │     │  Engine      │
  └─────────────┘     └──────────────┘     └──────┬───────┘
                                                   │
                      ┌──────────────┐     ┌──────▼───────┐
                      │  Limit Order │<────│  Risk Gate   │
                      │  Manager     │     │  + Kill Sw.  │
                      └──────┬───────┘     └──────────────┘
                             │
                      ┌──────▼───────┐     ┌──────────────┐
                      │  CCXT Exec   │────>│  Chandelier  │
                      │  Engine      │     │  Exit Mgr    │
                      └──────────────┘     └──────────────┘

Key differences from boilerplate:
  1. Ring buffers (not deques) for O(1) indicator access
  2. 4-signal composite with funding velocity
  3. Volatility phase gate (skip compression)
  4. Market-structure SL (not fixed ATR)
  5. Chandelier 4-phase exit management
  6. Limit order preference with fill simulation
  7. Liquidation cluster entry alignment
===============================================================================
"""

import asyncio
import logging
import time
import json
import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# DATA BUFFER — High-performance ring buffer for indicator computation
# =============================================================================

class RingBuffer:
    """Fixed-size numpy ring buffer for O(1) append and O(1) slice access."""
    
    def __init__(self, capacity: int, dtype=np.float64):
        self._buffer = np.zeros(capacity, dtype=dtype)
        self._capacity = capacity
        self._size = 0
        self._index = 0
    
    def append(self, value):
        self._buffer[self._index] = value
        self._index = (self._index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def as_array(self) -> np.ndarray:
        if self._size < self._capacity:
            return self._buffer[:self._size].copy()
        return np.roll(self._buffer, -self._index)[:self._size].copy()
    
    @property
    def last(self) -> float:
        if self._size == 0:
            return 0.0
        return self._buffer[(self._index - 1) % self._capacity]
    
    def __len__(self):
        return self._size


class MarketDataBuffer:
    """
    Maintains synchronized ring buffers for all OHLCV + perp-specific data.
    
    Feeds from the WebSocket trade stream via the OHLCVBuilder.
    The strategy engine reads from these buffers for indicator computation.
    """
    
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.open = RingBuffer(capacity)
        self.high = RingBuffer(capacity)
        self.low = RingBuffer(capacity)
        self.close = RingBuffer(capacity)
        self.volume = RingBuffer(capacity)
        self.funding_rate = RingBuffer(capacity)
        self.open_interest = RingBuffer(capacity)
        
        # Tick-level data for orderflow
        self.tick_prices = deque(maxlen=10000)
        self.tick_volumes = deque(maxlen=10000)
        self.tick_sides = deque(maxlen=10000)   # 1=buy, -1=sell
        self.tick_timestamps = deque(maxlen=10000)
        
        self.last_candle_time = 0
        self.bar_count = 0
    
    def update_candle(self, candle: dict):
        """Called when a new 1-minute candle closes."""
        ts = candle.get('timestamp', 0)
        if ts <= self.last_candle_time:
            return  # Duplicate
        
        self.open.append(candle['open'])
        self.high.append(candle['high'])
        self.low.append(candle['low'])
        self.close.append(candle['close'])
        self.volume.append(candle['volume'])
        self.last_candle_time = ts
        self.bar_count += 1
    
    def update_funding(self, rate: float):
        """Called every 8 hours when funding settles, or from REST poll."""
        self.funding_rate.append(rate)
    
    def update_oi(self, oi: float):
        """Called from periodic REST poll (every 5 min)."""
        self.open_interest.append(oi)
    
    def update_tick(self, price: float, qty: float, is_buyer_maker: bool, ts: int):
        """Called for every trade from the WebSocket stream."""
        self.tick_prices.append(price)
        self.tick_volumes.append(qty)
        self.tick_sides.append(-1 if is_buyer_maker else 1)  # Taker side
        self.tick_timestamps.append(ts)
    
    @property
    def ready(self) -> bool:
        return len(self.close) >= 120  # Minimum bars for indicators


# =============================================================================
# VOLATILITY PHASE CLASSIFIER
# =============================================================================

class VolatilityPhase(Enum):
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    NORMAL = "normal"
    EXTREME = "extreme"

def classify_volatility(close: np.ndarray, high: np.ndarray,
                        low: np.ndarray, bb_period=20, bb_std=2.0,
                        kc_period=20, kc_mult=1.5,
                        lookback=120) -> Tuple[VolatilityPhase, float, bool]:
    """
    TTM Squeeze concept: BB inside KC = compression.
    BB exits KC after squeeze = expansion (fire).
    
    Returns (phase, bbw_percentile, squeeze_just_fired)
    """
    n = len(close)
    if n < lookback:
        return VolatilityPhase.NORMAL, 50.0, False
    
    import pandas as pd
    sma = pd.Series(close).rolling(bb_period).mean().values
    std = pd.Series(close).rolling(bb_period).std().values
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std
    bbw = (bb_upper - bb_lower) / (sma + 1e-10)
    
    valid_bbw = bbw[-lookback:]
    valid_bbw = valid_bbw[~np.isnan(valid_bbw)]
    if len(valid_bbw) < 20:
        return VolatilityPhase.NORMAL, 50.0, False
    
    bbw_pctile = np.sum(valid_bbw < bbw[-1]) / len(valid_bbw) * 100
    
    # Keltner channels
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False).mean().values
    kc_upper = sma + kc_mult * atr
    kc_lower = sma - kc_mult * atr
    
    squeeze = np.zeros(n, dtype=bool)
    for j in range(bb_period, n):
        if not np.isnan(bb_upper[j]) and not np.isnan(kc_upper[j]):
            squeeze[j] = (bb_upper[j] < kc_upper[j]) and (bb_lower[j] > kc_lower[j])
    
    squeeze_fired = False
    if n >= 5:
        squeeze_fired = any(squeeze[-5:-1]) and not squeeze[-1]
    
    if bbw_pctile > 90:
        phase = VolatilityPhase.EXTREME
    elif squeeze[-1] or bbw_pctile < 20:
        phase = VolatilityPhase.COMPRESSION
    elif squeeze_fired or (50 < bbw_pctile < 90):
        phase = VolatilityPhase.EXPANSION
    else:
        phase = VolatilityPhase.NORMAL
    
    return phase, bbw_pctile, squeeze_fired


# =============================================================================
# FUNDING RATE VELOCITY SIGNAL
# =============================================================================

def compute_funding_velocity(funding_rates: np.ndarray,
                             vel_period=8, accel_period=4) -> Tuple[float, float]:
    """
    Compute funding rate velocity signal.
    Counter-trend to crowding: high positive FR accelerating → bearish signal.
    """
    n = len(funding_rates)
    if n < vel_period + accel_period + 1:
        return 0.0, 0.0
    
    fr_current = funding_rates[-1]
    velocity = (fr_current - funding_rates[-(vel_period + 1)]) / vel_period
    
    vel_history = np.diff(funding_rates[-min(100, n):])
    vel_std = np.std(vel_history) + 1e-10
    vel_norm = velocity / vel_std
    
    signal = 0.0
    if fr_current > 0.001 and vel_norm > 1.0:
        signal = -np.clip(vel_norm * 0.3, 0, 1)
    elif fr_current < -0.001 and vel_norm < -1.0:
        signal = np.clip(abs(vel_norm) * 0.3, 0, 1)
    elif abs(fr_current) > 0.002 and abs(vel_norm) < 0.5:
        signal = 0.2 if fr_current > 0 else -0.2
    else:
        signal = np.clip(vel_norm * 0.1, -0.3, 0.3)
    
    return signal, velocity


# =============================================================================
# MARKET-STRUCTURE STOP LOSS
# =============================================================================

def compute_structure_sl(direction: int, high: np.ndarray, low: np.ndarray,
                         close: np.ndarray, atr: float,
                         lookback: int = 30) -> float:
    """
    Place SL behind nearest swing point + 0.3×ATR buffer.
    Falls back to 1.2×ATR if no structure found.
    """
    n = len(close)
    price = close[-1]
    
    if n < lookback:
        return price - direction * 1.2 * atr
    
    if direction > 0:
        recent_low = low[-lookback:]
        swings = []
        for i in range(2, len(recent_low) - 1):
            if (recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i-2]
                    and recent_low[i] <= recent_low[i+1]):
                swings.append(recent_low[i])
        if swings:
            sl = max(swings) - 0.3 * atr
            return np.clip(sl, price - 3.0 * atr, price - 0.5 * atr)
        return price - 1.2 * atr
    else:
        recent_high = high[-lookback:]
        swings = []
        for i in range(2, len(recent_high) - 1):
            if (recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i-2]
                    and recent_high[i] >= recent_high[i+1]):
                swings.append(recent_high[i])
        if swings:
            sl = min(swings) + 0.3 * atr
            return np.clip(sl, price + 0.5 * atr, price + 3.0 * atr)
        return price + 1.2 * atr


# =============================================================================
# LIQUIDATION CLUSTER MAPPER
# =============================================================================

def find_liquidation_clusters(high: np.ndarray, low: np.ndarray,
                              close: np.ndarray, oi: np.ndarray,
                              lookback: int = 50) -> Dict:
    """Find estimated liquidation cluster levels from swing points + OI."""
    n = len(close)
    price = close[-1]
    
    if n < lookback:
        return {'nearest_long': price * 0.95, 'nearest_short': price * 1.05,
                'long_density': 0, 'short_density': 0}
    
    rh, rl = high[-lookback:], low[-lookback:]
    
    swing_highs, swing_lows = [], []
    for i in range(2, len(rh) - 2):
        if rh[i] > rh[i-1] and rh[i] > rh[i-2] and rh[i] > rh[i+1] and rh[i] > rh[i+2]:
            swing_highs.append(rh[i])
        if rl[i] < rl[i-1] and rl[i] < rl[i-2] and rl[i] < rl[i+1] and rl[i] < rl[i+2]:
            swing_lows.append(rl[i])
    
    long_liqs = [sl for sl in swing_lows if sl < price]
    short_liqs = [sh for sh in swing_highs if sh > price]
    
    nearest_long = max(long_liqs) if long_liqs else price * 0.95
    nearest_short = min(short_liqs) if short_liqs else price * 1.05
    
    oi_change = np.diff(oi[-min(lookback, len(oi)):]) if len(oi) > 1 else np.array([0])
    density = np.sum(np.abs(oi_change)) * 0.5
    
    return {'nearest_long': nearest_long, 'nearest_short': nearest_short,
            'long_density': density, 'short_density': density}


def liquidation_entry_score(direction: int, price: float, clusters: Dict) -> float:
    """Score how well entry aligns with nearby liquidation clusters."""
    if direction > 0:
        dist = (clusters['nearest_short'] - price) / price
    else:
        dist = (price - clusters['nearest_long']) / price
    
    if dist < 0.01:
        return 0.8
    elif dist < 0.03:
        return 0.5
    elif dist < 0.05:
        return 0.2
    return 0.0


# =============================================================================
# CHANDELIER EXIT — 4-Phase Non-Linear Trailing Stop
# =============================================================================

CHANDELIER_PHASES = [
    {'threshold_r': 1.0, 'action': 'breakeven', 'trail_atr_mult': None},
    {'threshold_r': 1.5, 'action': 'chandelier', 'trail_atr_mult': 2.5},
    {'threshold_r': 2.5, 'action': 'chandelier', 'trail_atr_mult': 1.5},
    {'threshold_r': 4.0, 'action': 'chandelier', 'trail_atr_mult': 1.0},
]

def chandelier_update(direction: int, entry_price: float, initial_risk: float,
                      current_high: float, current_low: float, current_atr: float,
                      current_stop: float, highest: float, lowest: float) -> Tuple[float, int]:
    """Compute chandelier exit level. Returns (new_stop, phase_index)."""
    if direction > 0:
        unrealized_r = (highest - entry_price) / (initial_risk + 1e-10)
    else:
        unrealized_r = (entry_price - lowest) / (initial_risk + 1e-10)
    
    active_phase = -1
    for i, phase in enumerate(CHANDELIER_PHASES):
        if unrealized_r >= phase['threshold_r']:
            active_phase = i
    
    if active_phase < 0:
        return current_stop, 0
    
    phase = CHANDELIER_PHASES[active_phase]
    
    if direction > 0:
        if phase['action'] == 'breakeven':
            new_sl = entry_price + 0.2 * initial_risk
        else:
            new_sl = highest - phase['trail_atr_mult'] * current_atr
        return max(current_stop, new_sl), active_phase + 1
    else:
        if phase['action'] == 'breakeven':
            new_sl = entry_price - 0.2 * initial_risk
        else:
            new_sl = lowest + phase['trail_atr_mult'] * current_atr
        return min(current_stop, new_sl), active_phase + 1


# =============================================================================
# CORE INDICATOR FUNCTIONS
# =============================================================================

def ema(data: np.ndarray, period: int) -> np.ndarray:
    import pandas as pd
    return pd.Series(data).ewm(span=period, adjust=False).mean().values

def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    import pandas as pd
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    import pandas as pd
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values

def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd
    n = len(high)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = compute_atr(high, low, close, period)
    atr_safe = np.where(atr == 0, 1e-10, atr)
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr_safe
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr_safe
    
    dx_denom = np.where(np.abs(plus_di + minus_di) == 0, 1e-10, np.abs(plus_di + minus_di))
    dx = 100 * np.abs(plus_di - minus_di) / dx_denom
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    return adx, plus_di, minus_di


# =============================================================================
# v3 SIGNAL MODEL
# =============================================================================

@dataclass
class V3Signal:
    direction: Literal["LONG", "SHORT", "NEUTRAL"]
    confidence: float
    grade: str                          # A+, A, B, C, D
    entry_price: float
    limit_entry_price: Optional[float]  # Maker limit order price
    stop_loss: float                    # Market-structure SL
    take_profit: float                  # Risk-multiple TP
    initial_risk: float                 # |entry - SL|
    
    # Signal components (for logging/debugging)
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
    timestamp: int = 0
    
    def is_tradeable(self) -> bool:
        return (self.direction != "NEUTRAL" 
                and self.grade in ["A+", "A", "B"]
                and self.confidence > 0.15)


# =============================================================================
# v3 PRODUCTION STRATEGY ENGINE
# =============================================================================

class V3StrategyEngine:
    """
    Production-ready v3 strategy engine.
    
    Call flow:
      1. Feed data via update_candle() / update_tick() / update_funding()
      2. Call generate_signal() on each new candle close
      3. If signal is tradeable, pass to risk manager
      4. For open positions, call update_exit() on every tick
    
    Thread safety: all methods are synchronous (called from async event loop
    via run_in_executor if CPU-bound computation takes >1ms).
    """
    
    def __init__(self, config: dict):
        self.symbol = config.get('trading', {}).get('symbol', 'BTC/USDT')
        self.buffer = MarketDataBuffer(capacity=500)
        
        # EMA periods
        self.ema_fast = 8
        self.ema_mid = 21
        self.ema_slow = 55
        
        # Signal threshold
        self.composite_threshold = 0.25
        
        # Position tracking for exit management
        self._open_positions: Dict[str, dict] = {}
    
    # ── Data Ingestion ───────────────────────────────────────────────
    
    def update_candle(self, candle: dict):
        """Feed closed 1-minute candle."""
        self.buffer.update_candle(candle)
    
    def update_tick(self, price: float, qty: float, 
                    is_buyer_maker: bool, ts: int):
        """Feed raw trade tick."""
        self.buffer.update_tick(price, qty, is_buyer_maker, ts)
    
    def update_funding(self, rate: float):
        """Feed funding rate (every 8h or from REST)."""
        self.buffer.update_funding(rate)
    
    def update_oi(self, oi: float):
        """Feed open interest (from REST poll)."""
        self.buffer.update_oi(oi)
    
    # ── Signal Generation ────────────────────────────────────────────
    
    def generate_signal(self) -> V3Signal:
        """
        Core signal generation. Call after each candle close.
        Returns a V3Signal with direction, confidence, SL/TP, and grade.
        """
        neutral = V3Signal(
            direction="NEUTRAL", confidence=0.0, grade="D",
            entry_price=0, limit_entry_price=None,
            stop_loss=0, take_profit=0, initial_risk=0,
            timestamp=int(time.time() * 1000)
        )
        
        if not self.buffer.ready:
            return neutral
        
        # Get arrays
        c = self.buffer.close.as_array()
        h = self.buffer.high.as_array()
        l = self.buffer.low.as_array()
        o = self.buffer.open.as_array()
        v = self.buffer.volume.as_array()
        fr = self.buffer.funding_rate.as_array()
        oi = self.buffer.open_interest.as_array()
        
        price = c[-1]
        
        # ── STEP 1: Regime Classification ────────────────────────────
        adx_vals, plus_di, minus_di = compute_adx(h, l, c)
        current_adx = adx_vals[-1]
        
        atr_arr = compute_atr(h, l, c)
        current_atr = atr_arr[-1]
        
        if current_adx > 25:
            regime = "trending_bull" if plus_di[-1] > minus_di[-1] else "trending_bear"
            regime_confidence = min(0.9, current_adx / 50)
        elif current_adx < 20:
            regime = "mean_reverting"
            regime_confidence = min(0.85, (25 - current_adx) / 25)
        else:
            regime = "normal"
            regime_confidence = 0.6
        
        # ── STEP 2: Volatility Phase Filter (PILLAR 1) ──────────────
        vol_phase, bbw_pctile, squeeze_fired = classify_volatility(c, h, l)
        
        if vol_phase == VolatilityPhase.COMPRESSION:
            neutral.vol_phase = "compression"
            neutral.regime = regime
            return neutral  # SKIP — death by 1000 cuts
        
        # ── STEP 3: Compute 4 Orthogonal Signals ────────────────────
        trend = self._trend_signal(c, h, l, v)
        momentum = self._momentum_signal(c, h, l, regime)
        orderflow = self._orderflow_signal(c, o, v)
        
        fr_signal = 0.0
        if len(fr) >= 12:
            fr_signal, _ = compute_funding_velocity(fr)
        
        # ── STEP 4: Regime-Weighted Composite ────────────────────────
        weights = self._get_weights(regime)
        composite = (weights['trend'] * trend + weights['momentum'] * momentum +
                     weights['orderflow'] * orderflow + weights['funding'] * fr_signal)
        
        # Direction gate
        if composite > self.composite_threshold:
            direction = 1
        elif composite < -self.composite_threshold:
            direction = -1
        else:
            neutral.composite = composite
            neutral.regime = regime
            return neutral
        
        dir_str = "LONG" if direction > 0 else "SHORT"
        
        # ── STEP 5: Liquidation Cluster Alignment (PILLAR 1C) ───────
        clusters = find_liquidation_clusters(h, l, c, oi if len(oi) > 0 else np.ones(len(c)))
        liq_score = liquidation_entry_score(direction, price, clusters)
        
        # ── STEP 6: Confidence & Grading ─────────────────────────────
        confidence = abs(composite) * regime_confidence
        if squeeze_fired:
            confidence *= 1.5
        if liq_score > 0.5:
            confidence *= 1.3
        
        if confidence > 0.5:
            grade = "A+"
        elif confidence > 0.35:
            grade = "A"
        elif confidence > 0.20:
            grade = "B"
        else:
            grade = "C"
        
        if grade in ["C", "D"]:
            neutral.composite = composite
            neutral.regime = regime
            return neutral
        
        # ── STEP 7: Edge Filters ─────────────────────────────────────
        vol_ma = np.mean(v[-min(720, len(v)):])
        if np.mean(v[-6:]) < vol_ma * 0.3:
            neutral.regime = regime
            return neutral  # Low liquidity
        
        # Confluence check
        signs = [np.sign(trend), np.sign(momentum), np.sign(orderflow), np.sign(fr_signal)]
        if sum(s == direction for s in signs if s != 0) < 2:
            neutral.regime = regime
            return neutral
        
        # Momentum confirmation
        if len(c) >= 4:
            last3 = c[-3:]
            if direction > 0 and sum(1 for j in range(1, 3) if last3[j] > last3[j-1]) < 1:
                neutral.regime = regime
                return neutral
            if direction < 0 and sum(1 for j in range(1, 3) if last3[j] < last3[j-1]) < 1:
                neutral.regime = regime
                return neutral
        
        # ── STEP 8: Market-Structure SL (PILLAR 2) ──────────────────
        sl = compute_structure_sl(direction, h, l, c, current_atr)
        initial_risk = abs(price - sl)
        
        # TP: 5× risk for trending, 3× for non-trending
        tp_mult = 5.0 if 'trending' in regime else 3.0
        tp = price + direction * tp_mult * initial_risk
        
        # ── STEP 9: Limit Order Entry Price (PILLAR 3) ───────────────
        if vol_phase == VolatilityPhase.EXPANSION:
            offset_mult = 0.15
        elif vol_phase == VolatilityPhase.COMPRESSION:
            offset_mult = 0.4
        else:
            offset_mult = 0.25
        
        limit_price = price - direction * current_atr * offset_mult
        
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
            regime=regime,
            timestamp=int(time.time() * 1000)
        )
    
    # ── Exit Management ──────────────────────────────────────────────
    
    def register_position(self, trade_id: str, direction: int, 
                          entry_price: float, initial_risk: float,
                          stop_loss: float, take_profit: float):
        """Register a filled position for exit tracking."""
        self._open_positions[trade_id] = {
            'direction': direction,
            'entry_price': entry_price,
            'initial_risk': initial_risk,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'highest': entry_price,
            'lowest': entry_price,
            'trail_phase': 0,
        }
    
    def update_exit(self, trade_id: str, current_high: float,
                    current_low: float) -> Optional[dict]:
        """
        Update chandelier exit for an open position.
        Call on every new candle or tick.
        
        Returns None if no exit triggered, or dict with exit details.
        """
        pos = self._open_positions.get(trade_id)
        if pos is None:
            return None
        
        pos['highest'] = max(pos['highest'], current_high)
        pos['lowest'] = min(pos['lowest'], current_low)
        
        # Current ATR
        c = self.buffer.close.as_array()
        h = self.buffer.high.as_array()
        l = self.buffer.low.as_array()
        current_atr = np.mean(h[-14:] - l[-14:]) if len(h) >= 14 else 0
        
        new_stop, phase = chandelier_update(
            pos['direction'], pos['entry_price'], pos['initial_risk'],
            current_high, current_low, current_atr,
            pos['stop_loss'], pos['highest'], pos['lowest']
        )
        pos['stop_loss'] = new_stop
        pos['trail_phase'] = phase
        
        # Check exit triggers
        hit_sl = (pos['direction'] > 0 and current_low <= pos['stop_loss']) or \
                 (pos['direction'] < 0 and current_high >= pos['stop_loss'])
        hit_tp = (pos['direction'] > 0 and current_high >= pos['take_profit']) or \
                 (pos['direction'] < 0 and current_low <= pos['take_profit'])
        
        if hit_sl or hit_tp:
            exit_info = {
                'trade_id': trade_id,
                'reason': f"trail_phase_{phase}" if hit_sl and phase > 0 else 
                          "stop_loss" if hit_sl else "take_profit",
                'exit_price': pos['stop_loss'] if hit_sl else pos['take_profit'],
                'phase': phase,
                'use_maker': not hit_sl,  # TP = limit order, SL = market
            }
            del self._open_positions[trade_id]
            return exit_info
        
        return None
    
    def get_current_stop(self, trade_id: str) -> Optional[float]:
        """Get current chandelier stop level for position."""
        pos = self._open_positions.get(trade_id)
        return pos['stop_loss'] if pos else None
    
    # ── Internal Signal Components ───────────────────────────────────
    
    def _trend_signal(self, close, high, low, volume) -> float:
        ema8 = ema(close, self.ema_fast)
        ema21 = ema(close, self.ema_mid)
        ema55 = ema(close, self.ema_slow)
        
        s = 0.0
        if ema8[-1] > ema21[-1] > ema55[-1]:
            s = 1.0
        elif ema8[-1] < ema21[-1] < ema55[-1]:
            s = -1.0
        elif ema8[-1] > ema21[-1]:
            s = 0.3
        elif ema8[-1] < ema21[-1]:
            s = -0.3
        
        # Market structure
        ms = 0.0
        lb = 20
        n = len(high)
        if n >= lb * 2:
            if high[-lb:].max() > high[-lb*2:-lb].max() and low[-lb:].min() > low[-lb*2:-lb].min():
                ms = 1.0
            elif high[-lb:].max() < high[-lb*2:-lb].max() and low[-lb:].min() < low[-lb*2:-lb].min():
                ms = -1.0
        
        vwap = np.cumsum(close * volume) / (np.cumsum(volume) + 1e-10)
        vwap_s = 1.0 if close[-1] > vwap[-1] else -1.0
        
        return np.clip(0.4 * s + 0.35 * ms + 0.25 * vwap_s, -1, 1)
    
    def _momentum_signal(self, close, high, low, regime: str) -> float:
        rsi = compute_rsi(close)
        macd_line = ema(close, 12) - ema(close, 26)
        macd_signal = ema(macd_line, 9)
        hist = macd_line - macd_signal
        
        if 'bull' in regime:
            ob, os_ = 80, 40
        elif 'bear' in regime:
            ob, os_ = 60, 20
        else:
            ob, os_ = 70, 30
        
        rv = rsi[-1]
        rs = -1.0 if rv > ob else (1.0 if rv < os_ else 1.0 - 2.0 * (rv - os_) / (ob - os_))
        
        ha = 0.0
        if len(hist) >= 3:
            ha = np.clip((hist[-1] - hist[-2]) / (np.std(hist[-50:]) + 1e-10), -1, 1)
        
        vel = 0.0
        if len(close) > 10:
            roc = (close[-1] - close[-10]) / close[-10]
            atr = np.mean(high[-14:] - low[-14:])
            vel = np.clip(roc / (atr / close[-1] + 1e-10) / 5, -1, 1)
        
        return np.clip(0.35 * rs + 0.35 * ha + 0.30 * vel, -1, 1)
    
    def _orderflow_signal(self, close, open_p, volume) -> float:
        buy = np.where(close > open_p, volume, volume * 0.4)
        sell = np.where(close <= open_p, volume, volume * 0.4)
        cvd = np.cumsum(buy - sell)
        
        cs = 0.0
        if len(close) >= 20:
            pt = (close[-1] - close[-20]) / close[-20]
            ct = (cvd[-1] - cvd[-20]) / (np.abs(cvd[-20]) + 1e-10)
            if pt > 0 and ct < -0.1:
                cs = -0.7
            elif pt < 0 and ct > 0.1:
                cs = 0.7
            else:
                cs = np.clip(ct, -1, 1) * 0.5
        
        rb = np.sum(buy[-10:])
        rs = np.sum(sell[-10:])
        imb = np.clip((rb - rs) / (rb + rs + 1e-10) * 2, -1, 1)
        
        return np.clip(0.6 * cs + 0.4 * imb, -1, 1)
    
    def _get_weights(self, regime: str) -> dict:
        if 'trending' in regime:
            return {'trend': 0.40, 'momentum': 0.25, 'orderflow': 0.15, 'funding': 0.20}
        elif 'mean' in regime:
            return {'trend': 0.10, 'momentum': 0.35, 'orderflow': 0.30, 'funding': 0.25}
        elif 'vol' in regime:
            return {'trend': 0.20, 'momentum': 0.20, 'orderflow': 0.35, 'funding': 0.25}
        return {'trend': 0.25, 'momentum': 0.25, 'orderflow': 0.30, 'funding': 0.20}


# =============================================================================
# INTEGRATION ADAPTER — Bridges V3Engine into the async bot framework
# =============================================================================

class V3BotAdapter:
    """
    Async wrapper that plugs V3StrategyEngine into the production bot.
    
    Usage in main.py:
    
        adapter = V3BotAdapter(config)
        
        # In WebSocket message handler:
        async def on_trade(data):
            adapter.feed_trade(data)
            
            if new_candle_closed:
                signal = await adapter.evaluate()
                if signal and signal.is_tradeable():
                    # Send to risk manager → execution
                    ...
        
        # In position monitoring loop:
        async def monitor_positions():
            for trade_id in open_positions:
                exit_info = adapter.check_exit(trade_id, current_high, current_low)
                if exit_info:
                    await execute_exit(exit_info)
    """
    
    def __init__(self, config: dict):
        self.engine = V3StrategyEngine(config)
        self._lock = asyncio.Lock()
    
    def feed_trade(self, trade_data: dict):
        """Feed raw exchange trade. Call from WebSocket handler."""
        price = float(trade_data.get('p', 0))
        qty = float(trade_data.get('q', 0))
        is_buyer_maker = trade_data.get('m', False)
        ts = trade_data.get('E', int(time.time() * 1000))
        self.engine.update_tick(price, qty, is_buyer_maker, ts)
    
    def feed_candle(self, candle: dict):
        """Feed closed candle from OHLCVBuilder."""
        self.engine.update_candle(candle)
    
    def feed_funding(self, rate: float):
        """Feed funding rate."""
        self.engine.update_funding(rate)
    
    def feed_oi(self, oi: float):
        """Feed open interest."""
        self.engine.update_oi(oi)
    
    async def evaluate(self) -> V3Signal:
        """
        Generate signal. Uses lock to prevent concurrent computation.
        Run in executor if computation exceeds 1ms.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            signal = await loop.run_in_executor(None, self.engine.generate_signal)
            return signal
    
    def register_fill(self, trade_id: str, direction: int, entry_price: float,
                      initial_risk: float, stop_loss: float, take_profit: float):
        """Register a filled order for exit management."""
        self.engine.register_position(
            trade_id, direction, entry_price, initial_risk, stop_loss, take_profit
        )
    
    def check_exit(self, trade_id: str, current_high: float,
                   current_low: float) -> Optional[dict]:
        """Check if chandelier exit triggered for position."""
        return self.engine.update_exit(trade_id, current_high, current_low)
    
    def get_stop(self, trade_id: str) -> Optional[float]:
        """Get current trailing stop for a position."""
        return self.engine.get_current_stop(trade_id)
