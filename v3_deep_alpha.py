"""
===============================================================================
APFTS v3 — DEEP ALPHA REFINEMENT
===============================================================================
Builds on v2 (PF 1.14, Exp +0.041%) targeting PF >1.75, Exp >0.15%

Three Pillars:
  PILLAR 1: Signal Quality & Filtering (boost winrate)
    - Volatility compression/expansion classifier
    - Funding rate velocity as non-correlated 4th signal
    - Liquidation cluster detection for entry precision

  PILLAR 2: Trade Management (boost R:R)
    - Chandelier exit (non-linear trailing stop)
    - Market-structure SL (swing-based, not fixed ATR)
    - Multi-phase exit (partial TP1, trail remainder)

  PILLAR 3: Perpetual-Specific Edge (reduce cost drag)
    - Limit-order entry preference (maker rebates)
    - Funding rate carry optimization
    - Liquidation magnet targeting

v2 Diagnostics (what we're fixing):
  - SL hit rate 58% → too many trades stopped out by noise
  - Fixed ATR stops ignore actual market structure
  - Trailing activates at 1.8R → misses many 1.0-1.7R runners
  - Entry always uses market orders → 5.5bp wasted per trade
  - No volatility compression filter → death by 1000 cuts in chop
  - Orderflow signal is an OHLCV proxy → needs funding velocity layer
===============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PILLAR 1A: VOLATILITY REGIME — COMPRESSION vs EXPANSION
# =============================================================================

class VolatilityPhase(Enum):
    COMPRESSION = "compression"     # Bollinger squeeze → DON'T trade
    EXPANSION = "expansion"         # Breakout from squeeze → trade aggressively
    NORMAL = "normal"               # Standard volatility
    EXTREME = "extreme"             # Vol spike (news) → widen stops or skip

class VolatilityClassifier:
    """
    Differentiates vol compression from expansion using Bollinger Band Width
    and Keltner Channel relationship (TTM Squeeze concept).
    
    Math:
      BBW = (upper_BB - lower_BB) / middle_BB
      BBW_percentile = rank(BBW_current, BBW_lookback)
      
      Squeeze = BB_inside_KC → compression phase
      Fire    = BB_outside_KC after squeeze → expansion
    
    The key insight: volatility is mean-reverting. Low vol compresses, then
    explodes. We want to AVOID the compression (chop/whipsaw) and ENTER
    on the expansion (directional move just starting).
    """
    
    def __init__(self, bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5,
                 lookback=120):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.lookback = lookback  # BBW percentile lookback
    
    def classify(self, close: np.ndarray, high: np.ndarray, 
                 low: np.ndarray) -> Tuple[VolatilityPhase, float, bool]:
        """
        Returns (phase, bbw_percentile, squeeze_just_fired)
        
        squeeze_just_fired = True when we transition from squeeze → expansion
        within the last 3 bars. This is the highest-probability entry window.
        """
        n = len(close)
        if n < self.lookback:
            return VolatilityPhase.NORMAL, 50.0, False
        
        # --- Bollinger Bands ---
        sma = pd.Series(close).rolling(self.bb_period).mean().values
        std = pd.Series(close).rolling(self.bb_period).std().values
        bb_upper = sma + self.bb_std * std
        bb_lower = sma - self.bb_std * std
        
        # Bollinger Band Width
        bbw = (bb_upper - bb_lower) / (sma + 1e-10)
        
        # BBW percentile over lookback
        valid_bbw = bbw[-self.lookback:]
        valid_bbw = valid_bbw[~np.isnan(valid_bbw)]
        if len(valid_bbw) < 20:
            return VolatilityPhase.NORMAL, 50.0, False
        
        bbw_current = bbw[-1]
        bbw_percentile = np.sum(valid_bbw < bbw_current) / len(valid_bbw) * 100
        
        # --- Keltner Channels ---
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - np.roll(close, 1)),
                       np.abs(low - np.roll(close, 1)))
        )
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=self.kc_period, adjust=False).mean().values
        kc_upper = sma + self.kc_mult * atr
        kc_lower = sma - self.kc_mult * atr
        
        # --- Squeeze Detection ---
        # Squeeze = BB fully inside KC (bands contracted)
        squeeze = np.zeros(n, dtype=bool)
        for j in range(self.bb_period, n):
            if not np.isnan(bb_upper[j]) and not np.isnan(kc_upper[j]):
                squeeze[j] = (bb_upper[j] < kc_upper[j]) and (bb_lower[j] > kc_lower[j])
        
        # Squeeze fire = was in squeeze, now exiting
        squeeze_just_fired = False
        if n >= 5:
            was_squeezed = any(squeeze[-5:-1])  # Squeezed in recent bars
            not_squeezed_now = not squeeze[-1]   # No longer squeezed
            squeeze_just_fired = was_squeezed and not_squeezed_now
        
        # --- Phase Classification ---
        if bbw_percentile > 90:
            phase = VolatilityPhase.EXTREME
        elif squeeze[-1] or bbw_percentile < 20:
            phase = VolatilityPhase.COMPRESSION
        elif squeeze_just_fired or (bbw_percentile > 50 and bbw_percentile < 90):
            phase = VolatilityPhase.EXPANSION
        else:
            phase = VolatilityPhase.NORMAL
        
        return phase, bbw_percentile, squeeze_just_fired


# =============================================================================
# PILLAR 1B: FUNDING RATE VELOCITY — NON-CORRELATED 4TH SIGNAL
# =============================================================================

class FundingVelocitySignal:
    """
    Funding rate alone is a crowding indicator. But the RATE OF CHANGE of
    funding (velocity) is a leading indicator of position unwinds.
    
    Math:
      FR_velocity = ∂FR/∂t = (FR_t - FR_{t-n}) / n
      FR_accel    = ∂²FR/∂t² = (FR_vel_t - FR_vel_{t-n}) / n
    
    Signal logic:
      - FR rising rapidly + positive → longs getting crowded, expect reversal DOWN
      - FR falling rapidly + negative → shorts getting crowded, expect reversal UP
      - FR velocity decelerating → crowding easing, trend may resume
      - FR velocity near zero → no crowding signal (neutral)
    
    This is orthogonal to price-based signals because funding is derived from
    the basis between perp and spot, not from price action directly.
    """
    
    def __init__(self, velocity_period=8, acceleration_period=4):
        self.velocity_period = velocity_period    # ~1 day (8 funding periods)
        self.acceleration_period = acceleration_period
    
    def compute(self, funding_rates: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (signal: -1 to 1, velocity, acceleration)
        
        Positive signal = bullish (shorts crowded, expect squeeze up)
        Negative signal = bearish (longs crowded, expect squeeze down)
        """
        n = len(funding_rates)
        if n < self.velocity_period + self.acceleration_period:
            return 0.0, 0.0, 0.0
        
        # Current funding rate
        fr_current = funding_rates[-1]
        
        # Velocity: rate of change of funding
        fr_past = funding_rates[-(self.velocity_period + 1)]
        velocity = (fr_current - fr_past) / self.velocity_period
        
        # Acceleration: rate of change of velocity
        vel_recent = velocity
        fr_earlier = funding_rates[-(self.velocity_period + self.acceleration_period + 1)]
        fr_mid = funding_rates[-(self.acceleration_period + 1)]
        vel_prior = (fr_mid - fr_earlier) / self.velocity_period
        acceleration = (vel_recent - vel_prior) / self.acceleration_period
        
        # Normalize velocity by historical std
        vel_history = np.diff(funding_rates[-min(100, n):])
        vel_std = np.std(vel_history) + 1e-10
        vel_normalized = velocity / vel_std
        
        # Signal: counter-trend to extreme funding
        # High positive FR + rising velocity → BEARISH (longs will get squeezed)
        # High negative FR + falling velocity → BULLISH (shorts will get squeezed)
        signal = 0.0
        
        if fr_current > 0.001 and vel_normalized > 1.0:
            # Extreme long crowding accelerating → bearish
            signal = -np.clip(vel_normalized * 0.3, 0, 1)
        elif fr_current < -0.001 and vel_normalized < -1.0:
            # Extreme short crowding accelerating → bullish
            signal = np.clip(abs(vel_normalized) * 0.3, 0, 1)
        elif abs(fr_current) > 0.002 and abs(vel_normalized) < 0.5:
            # Extreme funding but velocity slowing → crowding easing
            # This means the trend might resume (pro-trend)
            if fr_current > 0:
                signal = 0.2  # Slight bullish (crowding eased)
            else:
                signal = -0.2  # Slight bearish
        else:
            # Normal funding — no signal
            signal = np.clip(vel_normalized * 0.1, -0.3, 0.3)
        
        return signal, velocity, acceleration


# =============================================================================
# PILLAR 1C: LIQUIDATION CLUSTER DETECTION
# =============================================================================

class LiquidationMapper:
    """
    Estimates where liquidation clusters sit based on recent price action,
    open interest changes, and typical leverage patterns.
    
    In real implementation: use exchange liquidation feeds.
    Here: estimate based on statistical model.
    
    Key insight: price acts as a "magnet" toward liquidation clusters because
    large liquidations create forced buying/selling that amplifies moves.
    
    Model:
      - Recent swing highs/lows = likely stop-loss cluster locations
      - OI increase during move = new positions that will be liquidated on reversal
      - Typical leverage 5-20x → liquidation at 5-20% adverse move
      - Round numbers attract clusters (psychological levels)
    
    Entry logic: enter when price is moving TOWARD a liquidation cluster
    (liquidation cascade will amplify your trade direction)
    """
    
    def find_clusters(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, oi: np.ndarray,
                      lookback: int = 50) -> Dict:
        """
        Returns dict with:
          - long_liq_levels: prices where long liquidations cluster (below current)
          - short_liq_levels: prices where short liquidations cluster (above current)
          - nearest_long_liq: closest long liquidation level
          - nearest_short_liq: closest short liquidation level
          - long_liq_density: estimated volume at nearest long liq level
          - short_liq_density: estimated volume at nearest short liq level
        """
        n = len(close)
        if n < lookback:
            return {'long_liq': [], 'short_liq': [], 'nearest_long': 0, 
                    'nearest_short': 0, 'long_density': 0, 'short_density': 0}
        
        current_price = close[-1]
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        recent_oi = oi[-lookback:] if len(oi) >= lookback else oi
        
        # --- Find swing highs/lows (liquidation magnets) ---
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_high) - 2):
            if (recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i-2] and
                recent_high[i] > recent_high[i+1] and recent_high[i] > recent_high[i+2]):
                swing_highs.append(recent_high[i])
            if (recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i-2] and
                recent_low[i] < recent_low[i+1] and recent_low[i] < recent_low[i+2]):
                swing_lows.append(recent_low[i])
        
        # --- Estimate liquidation levels ---
        # Longs get liquidated BELOW entry → swing lows are likely SL zones
        # Shorts get liquidated ABOVE entry → swing highs are likely SL zones
        # With 10x leverage, liquidation at ~10% below entry
        # With 20x leverage, liquidation at ~5% below entry
        
        long_liq_levels = []
        for sl in swing_lows:
            if sl < current_price:
                # Also add levels 2-5% below swing lows (high leverage liq)
                long_liq_levels.append(sl)
                long_liq_levels.append(sl * 0.98)  # 2% below for 50x leverage
        
        short_liq_levels = []
        for sh in swing_highs:
            if sh > current_price:
                short_liq_levels.append(sh)
                short_liq_levels.append(sh * 1.02)
        
        # --- Find nearest clusters ---
        nearest_long = max(long_liq_levels) if long_liq_levels else current_price * 0.95
        nearest_short = min(short_liq_levels) if short_liq_levels else current_price * 1.05
        
        # --- Estimate density (based on OI changes near those levels) ---
        # More OI built during the price range = denser liquidation cluster
        oi_change = np.diff(recent_oi) if len(recent_oi) > 1 else np.array([0])
        oi_buildup = np.sum(np.abs(oi_change))
        long_density = oi_buildup * 0.5  # Rough estimate: half are longs
        short_density = oi_buildup * 0.5
        
        return {
            'long_liq': long_liq_levels,
            'short_liq': short_liq_levels,
            'nearest_long': nearest_long,
            'nearest_short': nearest_short,
            'long_density': long_density,
            'short_density': short_density
        }
    
    def entry_score(self, direction: int, current_price: float, 
                    clusters: Dict) -> float:
        """
        Score how well the entry aligns with liquidation clusters.
        
        +1.0 = price moving toward dense cluster (cascade will help us)
        -1.0 = price moving away from clusters (no cascade to help)
         0.0 = neutral
        
        For LONG: we want a short-liq cluster ABOVE (price will be pulled up)
        For SHORT: we want a long-liq cluster BELOW (price will be pulled down)
        """
        if direction > 0:  # Long
            # Short liquidations above us = bullish catalyst
            target_liq = clusters['nearest_short']
            distance_pct = (target_liq - current_price) / current_price
            
            if distance_pct < 0.01:  # Very close → cascade imminent
                return 0.8
            elif distance_pct < 0.03:  # Reachable
                return 0.5
            elif distance_pct < 0.05:
                return 0.2
            else:
                return 0.0
        
        elif direction < 0:  # Short
            target_liq = clusters['nearest_long']
            distance_pct = (current_price - target_liq) / current_price
            
            if distance_pct < 0.01:
                return 0.8
            elif distance_pct < 0.03:
                return 0.5
            elif distance_pct < 0.05:
                return 0.2
            else:
                return 0.0
        
        return 0.0


# =============================================================================
# PILLAR 2A: MARKET-STRUCTURE STOP LOSS
# =============================================================================

class MarketStructureSL:
    """
    Instead of fixed N×ATR stop loss, place SL behind the nearest
    significant market structure (swing high/low).
    
    For LONG: SL just below the most recent higher low
    For SHORT: SL just above the most recent lower high
    
    Add a small ATR buffer (0.3×ATR) to avoid exact-level stops.
    
    This typically gives tighter stops than fixed ATR (better R:R)
    while being more structurally sound (only invalidated when
    market structure actually breaks).
    """
    
    def compute_sl(self, direction: int, high: np.ndarray, low: np.ndarray,
                   close: np.ndarray, atr: float, 
                   lookback: int = 30) -> float:
        """
        Returns the market-structure stop loss price.
        Falls back to 1.2×ATR if no clear structure found.
        """
        n = len(close)
        current_price = close[-1]
        
        if n < lookback:
            # Fallback
            if direction > 0:
                return current_price - 1.2 * atr
            else:
                return current_price + 1.2 * atr
        
        if direction > 0:  # Long → SL below recent swing low
            # Find the most recent higher low
            swing_lows = []
            recent_low = low[-lookback:]
            for i in range(2, len(recent_low) - 1):
                if (recent_low[i] < recent_low[i-1] and 
                    recent_low[i] < recent_low[i-2] and
                    recent_low[i] <= recent_low[i+1]):
                    swing_lows.append(recent_low[i])
            
            if swing_lows:
                # Use the highest swing low (most recent support)
                structure_level = max(swing_lows)
                # Add buffer below
                sl = structure_level - 0.3 * atr
                
                # Sanity check: SL shouldn't be more than 3×ATR away
                max_sl = current_price - 3.0 * atr
                min_sl = current_price - 0.5 * atr  # Minimum distance
                sl = np.clip(sl, max_sl, min_sl)
                return sl
            else:
                return current_price - 1.2 * atr
        
        else:  # Short → SL above recent swing high
            swing_highs = []
            recent_high = high[-lookback:]
            for i in range(2, len(recent_high) - 1):
                if (recent_high[i] > recent_high[i-1] and 
                    recent_high[i] > recent_high[i-2] and
                    recent_high[i] >= recent_high[i+1]):
                    swing_highs.append(recent_high[i])
            
            if swing_highs:
                structure_level = min(swing_highs)
                sl = structure_level + 0.3 * atr
                max_sl = current_price + 3.0 * atr
                min_sl = current_price + 0.5 * atr
                sl = np.clip(sl, min_sl, max_sl)
                return sl
            else:
                return current_price + 1.2 * atr


# =============================================================================
# PILLAR 2B: CHANDELIER EXIT (NON-LINEAR TRAILING STOP)
# =============================================================================

class ChandelierExit:
    """
    The Chandelier Exit trails the highest high (for longs) minus N×ATR.
    Unlike a fixed trailing stop, it automatically adjusts for volatility
    and "hangs" from the extreme of the move.
    
    v3 Enhancement: Multi-phase trailing
      Phase 0 (initial): Fixed SL from market structure
      Phase 1 (after +1.0R): Move to breakeven + 0.2R
      Phase 2 (after +1.5R): Chandelier with 2.5×ATR
      Phase 3 (after +2.5R): Tighten chandelier to 1.5×ATR
      Phase 4 (after +4.0R): Very tight 1.0×ATR trail
    
    This captures maximum trend while giving room during early movement.
    """
    
    def __init__(self):
        self.phases = [
            {'threshold_r': 1.0, 'action': 'breakeven', 'trail_atr_mult': None},
            {'threshold_r': 1.5, 'action': 'chandelier', 'trail_atr_mult': 2.5},
            {'threshold_r': 2.5, 'action': 'chandelier', 'trail_atr_mult': 1.5},
            {'threshold_r': 4.0, 'action': 'chandelier', 'trail_atr_mult': 1.0},
        ]
    
    def update_stop(self, direction: int, entry_price: float,
                    initial_risk: float, current_high: float, 
                    current_low: float, current_atr: float,
                    current_stop: float, highest_since_entry: float,
                    lowest_since_entry: float) -> Tuple[float, int]:
        """
        Returns (new_stop_loss, current_phase).
        """
        if direction > 0:
            unrealized_r = (highest_since_entry - entry_price) / initial_risk
            
            # Find current phase
            active_phase = -1
            for i, phase in enumerate(self.phases):
                if unrealized_r >= phase['threshold_r']:
                    active_phase = i
            
            if active_phase < 0:
                return current_stop, 0  # Phase 0: hold initial SL
            
            phase = self.phases[active_phase]
            
            if phase['action'] == 'breakeven':
                new_sl = entry_price + 0.2 * initial_risk
            else:  # chandelier
                new_sl = highest_since_entry - phase['trail_atr_mult'] * current_atr
            
            # SL can only move UP for longs
            return max(current_stop, new_sl), active_phase + 1
        
        else:  # Short
            unrealized_r = (entry_price - lowest_since_entry) / initial_risk
            
            active_phase = -1
            for i, phase in enumerate(self.phases):
                if unrealized_r >= phase['threshold_r']:
                    active_phase = i
            
            if active_phase < 0:
                return current_stop, 0
            
            phase = self.phases[active_phase]
            
            if phase['action'] == 'breakeven':
                new_sl = entry_price - 0.2 * initial_risk
            else:
                new_sl = lowest_since_entry + phase['trail_atr_mult'] * current_atr
            
            return min(current_stop, new_sl), active_phase + 1


# =============================================================================
# PILLAR 3: LIMIT ORDER ENTRY SIMULATION
# =============================================================================

class LimitOrderManager:
    """
    Instead of always entering at market (5.5bp taker fee), 
    place limit orders at favorable levels.
    
    Strategy:
      - Place limit at bid/ask + offset toward our direction
      - Offset = 0.2–0.5× ATR from current price
      - If price pulls back to our limit within N bars → filled at maker rate (2bp)
      - If price runs away without filling → opportunity cost (missed trade)
    
    Expected impact:
      - 70% fill rate (price usually retests within 3 bars in crypto)
      - Save 3.5bp per filled trade (5.5bp taker → 2.0bp maker)
      - Plus 0.2×ATR better entry price on average
    
    Net: we trade less frequently but each trade has lower cost and
    slightly better entry, compounding into higher expectancy.
    """
    
    def compute_limit_price(self, direction: int, current_price: float,
                            atr: float, vol_phase: VolatilityPhase) -> float:
        """Compute limit order price with regime-adaptive offset."""
        # In expansion: smaller offset (don't want to miss the move)
        # In normal/compression: larger offset (more likely to retrace)
        if vol_phase == VolatilityPhase.EXPANSION:
            offset_mult = 0.15
        elif vol_phase == VolatilityPhase.COMPRESSION:
            offset_mult = 0.4
        else:
            offset_mult = 0.25
        
        offset = atr * offset_mult
        
        if direction > 0:  # Long → bid below current
            return current_price - offset
        else:  # Short → ask above current
            return current_price + offset
    
    def simulate_fill(self, limit_price: float, direction: int,
                      future_lows: np.ndarray, future_highs: np.ndarray,
                      max_wait_bars: int = 3) -> Tuple[bool, int]:
        """
        Check if limit order would have filled within max_wait_bars.
        Returns (filled: bool, bar_filled: int).
        """
        for i in range(min(max_wait_bars, len(future_lows))):
            if direction > 0 and future_lows[i] <= limit_price:
                return True, i
            elif direction < 0 and future_highs[i] >= limit_price:
                return True, i
        return False, -1


# =============================================================================
# v3 INTEGRATED SIGNAL ENGINE
# =============================================================================

class SignalEngineV3:
    """
    Enhanced signal engine with 4 orthogonal signals + vol phase filter.
    
    Signals (each output -1 to 1):
      1. Trend (EMA alignment + market structure)        — same as v2
      2. Momentum (RSI + MACD + velocity)                — same as v2
      3. Orderflow (CVD + volume imbalance)              — same as v2
      4. Funding Velocity (NEW — rate of change of FR)   — orthogonal
    
    Regime weights now include 4th signal:
      Trending:  T:40 M:25 O:15 F:20
      MR:        T:10 M:35 O:30 F:25
      High Vol:  T:20 M:20 O:35 F:25
      Low Liq:   T:25 M:25 O:30 F:20
    
    Entry gating:
      - Vol phase COMPRESSION → SKIP (death by 1000 cuts)
      - Vol phase EXPANSION + squeeze fire → BOOST confidence 1.5×
      - Liquidation alignment → BOOST confidence 1.3×
    """
    
    def __init__(self):
        # v2 components
        self.ema_fast = 8
        self.ema_mid = 21
        self.ema_slow = 55
        self.rsi_period = 14
        
        # v3 new components
        self.vol_classifier = VolatilityClassifier()
        self.funding_signal = FundingVelocitySignal()
        self.liq_mapper = LiquidationMapper()
        self.structure_sl = MarketStructureSL()
        self.chandelier = ChandelierExit()
        self.limit_mgr = LimitOrderManager()
    
    def ema(self, data, period):
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    def compute_rsi(self, close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def compute_macd(self, close):
        fast = self.ema(close, 12)
        slow = self.ema(close, 26)
        macd_line = fast - slow
        signal_line = self.ema(macd_line, 9)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def trend_signal(self, close, high, low, volume) -> float:
        ema8 = self.ema(close, self.ema_fast)
        ema21 = self.ema(close, self.ema_mid)
        ema55 = self.ema(close, self.ema_slow)
        
        ema_score = 0.0
        if ema8[-1] > ema21[-1] > ema55[-1]:
            ema_score = 1.0
        elif ema8[-1] < ema21[-1] < ema55[-1]:
            ema_score = -1.0
        elif ema8[-1] > ema21[-1]:
            ema_score = 0.3
        elif ema8[-1] < ema21[-1]:
            ema_score = -0.3
        
        # Market structure
        n = len(high)
        lookback = 20
        ms_score = 0.0
        if n >= lookback * 2:
            rh = high[-lookback:].max()
            ph = high[-lookback*2:-lookback].max()
            rl = low[-lookback:].min()
            pl = low[-lookback*2:-lookback].min()
            if rh > ph and rl > pl:
                ms_score = 1.0
            elif rh < ph and rl < pl:
                ms_score = -1.0
        
        vwap = np.cumsum(close * volume) / (np.cumsum(volume) + 1e-10)
        vwap_score = 1.0 if close[-1] > vwap[-1] else -1.0
        
        return np.clip(0.4 * ema_score + 0.35 * ms_score + 0.25 * vwap_score, -1, 1)
    
    def momentum_signal(self, close, high, low, regime_type: str) -> float:
        rsi = self.compute_rsi(close)
        _, _, macd_hist = self.compute_macd(close)
        
        if 'bull' in regime_type:
            rsi_ob, rsi_os = 80, 40
        elif 'bear' in regime_type:
            rsi_ob, rsi_os = 60, 20
        else:
            rsi_ob, rsi_os = 70, 30
        
        rsi_val = rsi[-1]
        if rsi_val > rsi_ob:
            rsi_score = -1.0
        elif rsi_val < rsi_os:
            rsi_score = 1.0
        else:
            rsi_score = 1.0 - 2.0 * (rsi_val - rsi_os) / (rsi_ob - rsi_os)
        
        if len(macd_hist) >= 3:
            hist_accel = macd_hist[-1] - macd_hist[-2]
            hist_accel_norm = np.clip(hist_accel / (np.std(macd_hist[-50:]) + 1e-10), -1, 1)
        else:
            hist_accel_norm = 0.0
        
        roc_period = 10
        if len(close) > roc_period:
            roc = (close[-1] - close[-roc_period]) / close[-roc_period]
            atr = np.mean(high[-14:] - low[-14:])
            velocity = np.clip(roc / (atr / close[-1] + 1e-10) / 5, -1, 1)
        else:
            velocity = 0.0
        
        return np.clip(0.35 * rsi_score + 0.35 * hist_accel_norm + 0.30 * velocity, -1, 1)
    
    def orderflow_signal(self, close, open_prices, volume) -> float:
        buy_vol = np.where(close > open_prices, volume, volume * 0.4)
        sell_vol = np.where(close <= open_prices, volume, volume * 0.4)
        cvd = np.cumsum(buy_vol - sell_vol)
        
        if len(close) >= 20:
            price_trend = (close[-1] - close[-20]) / close[-20]
            cvd_trend = (cvd[-1] - cvd[-20]) / (np.abs(cvd[-20]) + 1e-10)
            
            if price_trend > 0 and cvd_trend < -0.1:
                cvd_score = -0.7
            elif price_trend < 0 and cvd_trend > 0.1:
                cvd_score = 0.7
            else:
                cvd_score = np.clip(cvd_trend, -1, 1) * 0.5
        else:
            cvd_score = 0.0
        
        recent_buy = np.sum(buy_vol[-10:])
        recent_sell = np.sum(sell_vol[-10:])
        imbalance = (recent_buy - recent_sell) / (recent_buy + recent_sell + 1e-10)
        imbalance_score = np.clip(imbalance * 2, -1, 1)
        
        return np.clip(0.6 * cvd_score + 0.4 * imbalance_score, -1, 1)
    
    def get_regime_weights_v3(self, regime_type: str) -> dict:
        """4-signal weights (Trend, Momentum, Orderflow, Funding Velocity)."""
        if 'trending' in regime_type:
            return {'trend': 0.40, 'momentum': 0.25, 'orderflow': 0.15, 'funding': 0.20}
        elif 'mean' in regime_type:
            return {'trend': 0.10, 'momentum': 0.35, 'orderflow': 0.30, 'funding': 0.25}
        elif 'high_vol' in regime_type or 'volatility' in regime_type:
            return {'trend': 0.20, 'momentum': 0.20, 'orderflow': 0.35, 'funding': 0.25}
        else:
            return {'trend': 0.25, 'momentum': 0.25, 'orderflow': 0.30, 'funding': 0.20}


# =============================================================================
# v3 SYNTHETIC DATA — WITH LIQUIDATION CASCADES
# =============================================================================

def generate_v3_market_data(n_bars=10000, seed=42):
    """
    v3 data generator with realistic microstructure PLUS:
    - Liquidation cascade events (price accelerates through liq clusters)
    - Funding rate with realistic velocity dynamics
    - Volatility compression/expansion cycles
    """
    np.random.seed(seed)
    
    log_price = np.log(50000.0)
    prev_return = 0.0
    opens, highs, lows, closes, volumes = [], [], [], [], []
    funding_rates, open_interests = [], []
    
    regime_duration = 0
    current_regime = 'trending'
    trend_dir = 1
    vol_state = 0.002
    funding_state = 0.0  # Continuous funding state
    
    # Track swing points for liquidation events
    recent_highs_for_liq = []
    recent_lows_for_liq = []
    
    for i in range(n_bars):
        regime_duration += 1
        if regime_duration > np.random.randint(120, 500):
            regime_duration = 0
            r = np.random.random()
            if r < 0.3:
                current_regime = 'trending'
                trend_dir = np.random.choice([-1, 1])
            elif r < 0.7:
                current_regime = 'mean_reverting'
            elif r < 0.9:
                current_regime = 'high_vol'
            else:
                current_regime = 'low_liq'
        
        if current_regime == 'trending':
            drift = trend_dir * 0.0003
            ar_coeff = 0.25
            base_vol = 0.0020
            volume_base = 1200
        elif current_regime == 'mean_reverting':
            drift = (np.log(50000) - log_price) * 0.00012
            ar_coeff = -0.15
            base_vol = 0.0015
            volume_base = 800
        elif current_regime == 'high_vol':
            drift = 0
            ar_coeff = 0.12
            base_vol = 0.0045
            volume_base = 2500
        else:
            drift = 0
            ar_coeff = -0.05
            base_vol = 0.001
            volume_base = 200
        
        # GARCH volatility
        vol_innovation = abs(np.random.randn()) * 0.0005
        vol_state = 0.85 * vol_state + 0.10 * base_vol + 0.05 * vol_innovation
        vol_state = np.clip(vol_state, 0.0005, 0.015)
        
        # --- LIQUIDATION CASCADE EVENTS ---
        # If price approaches a swing point cluster, add extra drift (cascade)
        liq_cascade = 0.0
        current_price = np.exp(log_price)
        
        if len(recent_lows_for_liq) > 3 and current_regime == 'trending' and trend_dir < 0:
            nearest_low = max([l for l in recent_lows_for_liq if l < current_price], default=0)
            if nearest_low > 0:
                dist = (current_price - nearest_low) / current_price
                if dist < 0.01:  # Within 1% of liq cluster
                    liq_cascade = -0.001 * (1 - dist / 0.01)  # Accelerate downward
        
        if len(recent_highs_for_liq) > 3 and current_regime == 'trending' and trend_dir > 0:
            nearest_high = min([h for h in recent_highs_for_liq if h > current_price], default=1e10)
            if nearest_high < 1e10:
                dist = (nearest_high - current_price) / current_price
                if dist < 0.01:
                    liq_cascade = 0.001 * (1 - dist / 0.01)
        
        # AR(1) return with cascade
        innovation = np.random.randn() * vol_state
        log_ret = drift + ar_coeff * prev_return + innovation + liq_cascade
        prev_return = log_ret
        
        open_p = np.exp(log_price)
        log_price += log_ret
        close_p = np.exp(log_price)
        
        wick_up = abs(np.random.randn()) * vol_state * 0.35 * open_p
        wick_dn = abs(np.random.randn()) * vol_state * 0.35 * open_p
        high_p = max(open_p, close_p) + wick_up
        low_p = min(open_p, close_p) - wick_dn
        
        # Track swings for liquidation mapping
        recent_highs_for_liq.append(high_p)
        recent_lows_for_liq.append(low_p)
        if len(recent_highs_for_liq) > 200:
            recent_highs_for_liq.pop(0)
            recent_lows_for_liq.pop(0)
        
        # Volume with cascade spikes
        return_magnitude = abs(log_ret) / vol_state
        vol_multiplier = 1.0 + 0.8 * return_magnitude
        if abs(liq_cascade) > 0.0005:
            vol_multiplier *= 2.5  # Liquidation events = massive volume spike
        volume = max(10, volume_base * vol_multiplier * (1 + 0.2 * np.random.randn()))
        
        # --- FUNDING RATE WITH VELOCITY DYNAMICS ---
        # Funding drifts toward 0 but builds up during trends
        funding_mean_revert = -funding_state * 0.05  # Mean revert toward 0
        funding_trend = 0
        if current_regime == 'trending':
            funding_trend = trend_dir * 0.00005 * min(regime_duration / 50, 1.0)
        funding_noise = np.random.normal(0, 0.0003)
        funding_state += funding_mean_revert + funding_trend + funding_noise
        funding_state = np.clip(funding_state, -0.01, 0.01)
        
        oi = 500000000 + np.random.randn() * 30000000
        if current_regime == 'trending':
            oi += regime_duration * 800000
        elif current_regime == 'high_vol':
            oi -= regime_duration * 500000
        
        opens.append(open_p)
        highs.append(high_p)
        lows.append(low_p)
        closes.append(close_p)
        volumes.append(volume)
        funding_rates.append(funding_state)
        open_interests.append(max(1e8, oi))
        
        log_price = np.clip(log_price, np.log(10000), np.log(200000))
    
    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'volume': volumes, 'funding_rate': funding_rates,
        'open_interest': open_interests
    })


# =============================================================================
# v3 BACKTEST ENGINE
# =============================================================================

from perp_trading_system import (
    MarketRegime, RegimeState, RegimeClassifier, SignalType, TradeSignal,
    PositionSizer, ExecutionSimulator, TradeRecord, BacktestMetrics, RiskManager
)

class BacktestEngineV3:
    """
    v3 backtester with all three pillars integrated.
    
    Key changes from v2:
    1. Volatility phase filter (skip compression)
    2. 4th signal: funding velocity
    3. Market-structure SL instead of fixed ATR
    4. Chandelier exit with 4-phase trailing
    5. Limit order entries (maker fee preference)
    6. Liquidation cluster alignment scoring
    """
    
    def __init__(self, initial_capital=100000, max_hold_bars=96):
        self.initial_capital = initial_capital
        self.max_hold_bars = max_hold_bars
        
        # v2 components
        self.regime_classifier = RegimeClassifier()
        self.position_sizer = PositionSizer()
        self.exec_sim = ExecutionSimulator()
        
        # v3 new components
        self.signal_engine = SignalEngineV3()
        self.chandelier = ChandelierExit()
        self.structure_sl = MarketStructureSL()
        self.liq_mapper = LiquidationMapper()
        self.limit_mgr = LimitOrderManager()
    
    def run(self, df: pd.DataFrame) -> BacktestMetrics:
        n = len(df)
        capital = self.initial_capital
        peak_capital = capital
        max_dd = 0
        
        trades = []
        equity_curve = [capital]
        
        in_position = False
        position_dir = 0
        entry_price = 0
        stop_loss = 0
        initial_risk = 0
        take_profit = 0
        pos_size_pct = 0
        entry_bar = 0
        entry_regime = ""
        entry_quality = ""
        exec_cost = 0
        highest_since_entry = 0
        lowest_since_entry = 1e18
        trail_phase = 0
        
        # Limit order state
        pending_limit = False
        limit_price = 0
        limit_direction = 0
        limit_signal = None
        limit_regime = None
        limit_bar = 0
        limit_take_profit = 0
        limit_stop_loss = 0
        limit_initial_risk = 0
        limit_quality = ""
        limit_pos_size = 0
        
        lookback = 120
        baseline_vol = np.std(df['close'].values[:min(720, n)] / 
                             np.roll(df['close'].values[:min(720, n)], 1))
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_p = df['open'].values
        volume = df['volume'].values
        funding = df['funding_rate'].values
        oi = df['open_interest'].values
        
        for i in range(lookback, n):
            current_close = close[i]
            current_high = high[i]
            current_low = low[i]
            
            # --- CHECK PENDING LIMIT ORDERS ---
            if pending_limit and not in_position:
                bars_waiting = i - limit_bar
                if bars_waiting > 3:  # Expired
                    pending_limit = False
                elif limit_direction > 0 and current_low <= limit_price:
                    # Long limit filled
                    in_position = True
                    position_dir = limit_direction
                    entry_price = limit_price
                    entry_bar = i
                    stop_loss = limit_stop_loss
                    initial_risk = limit_initial_risk
                    take_profit = limit_take_profit
                    pos_size_pct = limit_pos_size
                    entry_regime = limit_regime
                    entry_quality = limit_quality
                    exec_cost = self.exec_sim.maker_fee_bps + 1.0  # Maker + minimal slip
                    highest_since_entry = current_high
                    lowest_since_entry = current_low
                    trail_phase = 0
                    pending_limit = False
                elif limit_direction < 0 and current_high >= limit_price:
                    in_position = True
                    position_dir = limit_direction
                    entry_price = limit_price
                    entry_bar = i
                    stop_loss = limit_stop_loss
                    initial_risk = limit_initial_risk
                    take_profit = limit_take_profit
                    pos_size_pct = limit_pos_size
                    entry_regime = limit_regime
                    entry_quality = limit_quality
                    exec_cost = self.exec_sim.maker_fee_bps + 1.0
                    highest_since_entry = current_high
                    lowest_since_entry = current_low
                    trail_phase = 0
                    pending_limit = False
            
            if in_position:
                # Track extremes
                highest_since_entry = max(highest_since_entry, current_high)
                lowest_since_entry = min(lowest_since_entry, current_low)
                
                # --- CHANDELIER EXIT ---
                current_atr = np.mean(high[max(0,i-14):i+1] - low[max(0,i-14):i+1])
                new_stop, trail_phase = self.chandelier.update_stop(
                    position_dir, entry_price, initial_risk,
                    current_high, current_low, current_atr,
                    stop_loss, highest_since_entry, lowest_since_entry
                )
                stop_loss = new_stop
                
                # Check exits
                hit_sl = (position_dir > 0 and current_low <= stop_loss) or \
                         (position_dir < 0 and current_high >= stop_loss)
                hit_tp = (position_dir > 0 and current_high >= take_profit) or \
                         (position_dir < 0 and current_low <= take_profit)
                bars_held = i - entry_bar
                time_exit = bars_held >= self.max_hold_bars
                
                if hit_sl or hit_tp or time_exit:
                    if hit_sl and trail_phase > 0:
                        exit_price = stop_loss
                        exit_reason = f"trail_phase_{trail_phase}"
                    elif hit_sl:
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                    elif hit_tp:
                        exit_price = take_profit
                        exit_reason = "take_profit"
                    else:
                        exit_price = current_close
                        exit_reason = "time_exit"
                    
                    if position_dir > 0:
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                    
                    # Exit fees (maker for TP, taker for SL)
                    is_market = exit_reason != "take_profit"
                    exit_fee = self.exec_sim.taker_fee_bps if is_market else self.exec_sim.maker_fee_bps
                    total_exec = exec_cost + exit_fee + 1.5  # + minimal slippage
                    
                    net_pnl = pnl_pct - total_exec / 100
                    risk_pct = initial_risk / entry_price * 100
                    pnl_r = net_pnl / risk_pct if risk_pct > 0 else 0
                    
                    capital *= (1 + net_pnl / 100 * pos_size_pct)
                    peak_capital = max(peak_capital, capital)
                    dd = (peak_capital - capital) / peak_capital * 100
                    max_dd = max(max_dd, dd)
                    
                    trades.append(TradeRecord(
                        entry_time=entry_bar, exit_time=i,
                        direction=position_dir,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        position_size_pct=pos_size_pct,
                        pnl_pct=net_pnl, pnl_r=pnl_r,
                        exit_reason=exit_reason, regime=entry_regime,
                        edge_quality=entry_quality,
                        execution_cost_bps=total_exec
                    ))
                    
                    in_position = False
                    position_dir = 0
            
            elif not pending_limit:
                # --- ENTRY LOGIC ---
                if i % 3 != 0:
                    equity_curve.append(capital)
                    continue
                
                # Slice window
                ws = max(0, i - 500)
                c_w = close[ws:i+1]
                h_w = high[ws:i+1]
                l_w = low[ws:i+1]
                o_w = open_p[ws:i+1]
                v_w = volume[ws:i+1]
                f_w = funding[ws:i+1]
                oi_w = oi[ws:i+1]
                
                if len(c_w) < lookback:
                    equity_curve.append(capital)
                    continue
                
                # 1. Regime classification (from v2)
                window_df = df.iloc[ws:i+1]
                regime_state = self.regime_classifier.classify(window_df)
                regime_type = regime_state.regime.value
                
                # 2. VOLATILITY PHASE FILTER (v3 new)
                vol_phase, bbw_pctile, squeeze_fired = \
                    self.signal_engine.vol_classifier.classify(c_w, h_w, l_w)
                
                if vol_phase == VolatilityPhase.COMPRESSION:
                    equity_curve.append(capital)
                    continue  # SKIP — death by 1000 cuts
                
                # 3. Compute 4 signals
                trend = self.signal_engine.trend_signal(c_w, h_w, l_w, v_w)
                momentum = self.signal_engine.momentum_signal(c_w, h_w, l_w, regime_type)
                orderflow = self.signal_engine.orderflow_signal(c_w, o_w, v_w)
                fr_signal, fr_vel, fr_accel = \
                    self.signal_engine.funding_signal.compute(f_w)
                
                # 4. Weighted composite
                weights = self.signal_engine.get_regime_weights_v3(regime_type)
                composite = (
                    weights['trend'] * trend +
                    weights['momentum'] * momentum +
                    weights['orderflow'] * orderflow +
                    weights['funding'] * fr_signal
                )
                
                # 5. Direction
                if composite > 0.25:
                    direction = 1
                    strong = composite > 0.55
                elif composite < -0.25:
                    direction = -1
                    strong = composite < -0.55
                else:
                    equity_curve.append(capital)
                    continue
                
                # 6. LIQUIDATION CLUSTER ALIGNMENT (v3 new)
                clusters = self.liq_mapper.find_clusters(h_w, l_w, c_w, oi_w)
                liq_score = self.liq_mapper.entry_score(direction, current_close, clusters)
                
                # 7. Confidence scoring
                confidence = abs(composite) * regime_state.confidence
                if squeeze_fired:
                    confidence *= 1.5  # Boost for vol expansion
                if liq_score > 0.5:
                    confidence *= 1.3  # Boost for liq alignment
                
                # Grade
                if confidence > 0.5:
                    grade = "A+"
                elif confidence > 0.35:
                    grade = "A"
                elif confidence > 0.20:
                    grade = "B"
                else:
                    grade = "C"
                
                # Only trade B+ in v3
                if grade in ["C", "D"]:
                    equity_curve.append(capital)
                    continue
                
                # 8. Edge filters (simplified from v2)
                vol_ma = np.mean(v_w[-720:]) if len(v_w) >= 720 else np.mean(v_w)
                recent_vol = np.mean(v_w[-6:])
                if recent_vol < vol_ma * 0.3:
                    equity_curve.append(capital)
                    continue
                
                # Confluence: at least 2/4 signals agree
                signs = [np.sign(trend), np.sign(momentum), 
                         np.sign(orderflow), np.sign(fr_signal)]
                agreement = sum(s == direction for s in signs if s != 0)
                if agreement < 2:
                    equity_curve.append(capital)
                    continue
                
                # Momentum confirmation
                if len(c_w) >= 4:
                    last3 = c_w[-3:]
                    if direction > 0:
                        if sum(1 for j in range(1, len(last3)) if last3[j] > last3[j-1]) < 1:
                            equity_curve.append(capital)
                            continue
                    else:
                        if sum(1 for j in range(1, len(last3)) if last3[j] < last3[j-1]) < 1:
                            equity_curve.append(capital)
                            continue
                
                # 9. MARKET-STRUCTURE SL (v3 new — Pillar 2)
                current_atr = np.mean(h_w[-14:] - l_w[-14:])
                ms_sl = self.structure_sl.compute_sl(
                    direction, h_w, l_w, c_w, current_atr
                )
                ms_initial_risk = abs(current_close - ms_sl)
                
                # TP at 3× risk for non-trending, 5× for trending
                if 'trending' in regime_type:
                    tp_mult = 5.0
                else:
                    tp_mult = 3.0
                
                if direction > 0:
                    ms_tp = current_close + tp_mult * ms_initial_risk
                else:
                    ms_tp = current_close - tp_mult * ms_initial_risk
                
                # 10. Position sizing
                current_dd = (peak_capital - capital) / peak_capital * 100
                current_vol_metric = np.std(c_w[-24:] / np.roll(c_w[-24:], 1))
                
                # Create a minimal signal object for position sizer
                mock_signal = TradeSignal(
                    direction=SignalType.STRONG_LONG if direction > 0 else SignalType.STRONG_SHORT,
                    confidence=confidence,
                    trend_score=trend, momentum_score=momentum,
                    orderflow_score=orderflow,
                    regime=regime_state.regime,
                    entry_price=current_close,
                    stop_loss=ms_sl, take_profit_1=ms_tp, take_profit_2=ms_tp,
                    position_size_pct=0, timeframe_alignment=0.5,
                    edge_quality=grade, filters_passed=True
                )
                
                pos_size = self.position_sizer.compute_size(
                    mock_signal, capital, current_dd, 0.55, 1.8,
                    baseline_vol, current_vol_metric
                )
                
                if pos_size < 0.001:
                    equity_curve.append(capital)
                    continue
                
                # 11. LIMIT ORDER ENTRY (v3 new — Pillar 3)
                lim_price = self.limit_mgr.compute_limit_price(
                    direction, current_close, current_atr, vol_phase
                )
                
                # Place limit order
                pending_limit = True
                limit_price = lim_price
                limit_direction = direction
                limit_bar = i
                limit_stop_loss = ms_sl
                limit_initial_risk = ms_initial_risk
                limit_take_profit = ms_tp
                limit_quality = grade
                limit_regime = regime_type
                limit_pos_size = pos_size
            
            equity_curve.append(capital)
        
        return self._compute_metrics(trades, equity_curve, max_dd)
    
    def _compute_metrics(self, trades, equity_curve, max_dd) -> BacktestMetrics:
        if not trades:
            return BacktestMetrics(
                total_trades=0, win_rate=0, avg_rr=0, expectancy=0,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown_pct=0,
                profit_factor=0, avg_trade_pnl_pct=0, total_return_pct=0,
                calmar_ratio=0, avg_execution_cost_bps=0
            )
        
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        returns = returns[returns != 0]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(8760) if len(returns) > 0 else 0
        
        downside = returns[returns < 0]
        sortino = np.mean(returns) / (np.std(downside) + 1e-10) * np.sqrt(8760) if len(downside) > 0 else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        calmar = total_return / max_dd if max_dd > 0 else 0
        avg_exec = np.mean([t.execution_cost_bps for t in trades])
        
        return BacktestMetrics(
            total_trades=len(trades), win_rate=win_rate, avg_rr=avg_rr,
            expectancy=expectancy, sharpe_ratio=sharpe, sortino_ratio=sortino,
            max_drawdown_pct=max_dd, profit_factor=profit_factor,
            avg_trade_pnl_pct=np.mean(pnls), total_return_pct=total_return,
            calmar_ratio=calmar, avg_execution_cost_bps=avg_exec,
            trades=trades
        )


# =============================================================================
# RUN v3 BACKTEST
# =============================================================================

def run_v3():
    print("=" * 70)
    print("  APFTS v3 — DEEP ALPHA BACKTEST")
    print("=" * 70)
    
    results_by_seed = []
    
    for seed in [42, 123, 456, 789, 1337]:
        df = generate_v3_market_data(n_bars=10000, seed=seed)
        engine = BacktestEngineV3(initial_capital=100000)
        r = engine.run(df)
        
        a_trades = [t for t in r.trades if t.edge_quality in ['A+', 'A']]
        a_wr = sum(1 for t in a_trades if t.pnl_pct > 0) / len(a_trades) * 100 if a_trades else 0
        a_exp = np.mean([t.pnl_pct for t in a_trades]) if a_trades else 0
        
        results_by_seed.append({
            'seed': seed, 'result': r,
            'a_wr': a_wr, 'a_exp': a_exp, 'a_count': len(a_trades)
        })
        
        print(f"\n  Seed {seed}: {r.total_trades} trades | WR={r.win_rate*100:.1f}% | "
              f"R:R={r.avg_rr:.2f}x | Exp={r.expectancy:.4f}% | "
              f"PF={r.profit_factor:.2f} | DD={r.max_drawdown_pct:.2f}% | "
              f"Ret={r.total_return_pct:.2f}% | Exec={r.avg_execution_cost_bps:.1f}bp")
        print(f"    Grade A: {len(a_trades)} trades, WR={a_wr:.1f}%, Exp={a_exp:.4f}%")
        
        # Exit distribution
        if r.trades:
            exit_counts = {}
            for t in r.trades:
                exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
            exits_str = " | ".join(f"{k}:{v}" for k, v in sorted(exit_counts.items()))
            print(f"    Exits: {exits_str}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  CROSS-SEED SUMMARY")
    print("=" * 70)
    
    all_wr = [r['result'].win_rate * 100 for r in results_by_seed]
    all_rr = [r['result'].avg_rr for r in results_by_seed]
    all_exp = [r['result'].expectancy for r in results_by_seed]
    all_pf = [r['result'].profit_factor for r in results_by_seed]
    all_dd = [r['result'].max_drawdown_pct for r in results_by_seed]
    all_ret = [r['result'].total_return_pct for r in results_by_seed]
    all_exec = [r['result'].avg_execution_cost_bps for r in results_by_seed]
    
    all_a_wr = [r['a_wr'] for r in results_by_seed]
    all_a_exp = [r['a_exp'] for r in results_by_seed]
    
    print(f"\n  {'Metric':<25} {'v2 Avg':>12} {'v3 Avg':>12} {'v3 Std':>12} {'Delta':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Win Rate':<25} {'45.0%':>12} {np.mean(all_wr):>11.1f}% {np.std(all_wr):>11.1f}%")
    print(f"  {'R:R':<25} {'1.37x':>12} {np.mean(all_rr):>11.2f}x {np.std(all_rr):>11.2f}")
    print(f"  {'Expectancy':<25} {'+0.041%':>12} {np.mean(all_exp):>+11.4f}% {np.std(all_exp):>11.4f}%")
    print(f"  {'Profit Factor':<25} {'1.14':>12} {np.mean(all_pf):>11.2f} {np.std(all_pf):>11.2f}")
    print(f"  {'Max Drawdown':<25} {'0.19%':>12} {np.mean(all_dd):>11.2f}% {np.std(all_dd):>11.2f}%")
    print(f"  {'Total Return':<25} {'+0.25%':>12} {np.mean(all_ret):>+11.2f}% {np.std(all_ret):>11.2f}%")
    print(f"  {'Exec Cost':<25} {'13.8bp':>12} {np.mean(all_exec):>10.1f}bp {np.std(all_exec):>11.1f}")
    print(f"\n  {'Grade A WR':<25} {'56.6%':>12} {np.mean(all_a_wr):>11.1f}% {np.std(all_a_wr):>11.1f}%")
    print(f"  {'Grade A Exp':<25} {'+0.228%':>12} {np.mean(all_a_exp):>+11.4f}% {np.std(all_a_exp):>11.4f}%")
    print(f"  {'Positive Seeds':<25} {'3/5':>12} {sum(1 for e in all_exp if e > 0)}/5")
    
    # Risk of ruin
    ror = RiskManager.risk_of_ruin(
        max(np.mean(all_wr)/100, 0.01),
        max(np.mean(all_rr), 0.01),
        0.02, 0.5
    )
    print(f"  {'Risk of Ruin':<25} {'100%':>12} {ror*100:>11.2f}%")
    
    return results_by_seed


if __name__ == "__main__":
    results = run_v3()
