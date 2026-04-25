"""
High-performance ring buffers for streaming market data.
Used exclusively by the production bot; the backtest feeds arrays directly.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.execution.liquidation_feed import LiquidationEvent


class RingBuffer:
    """Fixed-capacity numpy ring buffer with O(1) append and O(n) ordered read."""

    def __init__(self, capacity: int, dtype: type = np.float64) -> None:
        self._buf = np.zeros(capacity, dtype=dtype)
        self._capacity = capacity
        self._size = 0
        self._idx = 0

    def append(self, value: float) -> None:
        self._buf[self._idx] = value
        self._idx = (self._idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def as_array(self) -> np.ndarray:
        """Return data in chronological order."""
        if self._size < self._capacity:
            return self._buf[: self._size].copy()
        return np.roll(self._buf, -self._idx)[: self._size].copy()

    @property
    def last(self) -> float:
        if self._size == 0:
            return 0.0
        return float(self._buf[(self._idx - 1) % self._capacity])

    def __len__(self) -> int:
        return self._size


class MarketDataBuffer:
    """
    Synchronized ring buffers for all OHLCV + perpetual-specific fields.
    The V3StrategyEngine reads from these on each candle close.
    """

    def __init__(self, capacity: int = 500) -> None:
        self.capacity = capacity
        self.open = RingBuffer(capacity)
        self.high = RingBuffer(capacity)
        self.low = RingBuffer(capacity)
        self.close = RingBuffer(capacity)
        self.volume = RingBuffer(capacity)
        self.funding_rate = RingBuffer(capacity)
        self.open_interest = RingBuffer(capacity)

        # Tick-level tape for intrabar orderflow
        self.tick_prices: deque[float] = deque(maxlen=10_000)
        self.tick_volumes: deque[float] = deque(maxlen=10_000)
        self.tick_sides: deque[int] = deque(maxlen=10_000)       # +1 buy / -1 sell
        self.tick_timestamps: deque[int] = deque(maxlen=10_000)

        # Real liquidation events (populated by LiquidationFeed when live_liq_feed=True)
        self.liquidation_events: deque[LiquidationEvent] = deque(maxlen=2_000)

        # L2 order book snapshot (populated by _orderbook_loop every ~5 s)
        self.ob_bids: list[list[float]] = []   # [[price, qty], ...] best-bid first
        self.ob_asks: list[list[float]] = []   # [[price, qty], ...] best-ask first
        self.ob_timestamp: int = 0             # ms epoch of last snapshot

        # Next funding timestamp in ms (updated by _funding_oi_loop)
        self.next_funding_ts_ms: int = 0

        self._last_candle_ts: int = 0
        self.bar_count: int = 0

    def update_candle(self, candle: dict) -> None:
        """Accept a closed OHLCV candle dict (keys: timestamp, open, high, low, close, volume)."""
        ts: int = candle.get("timestamp", 0)
        if ts <= self._last_candle_ts:
            return
        self.open.append(float(candle["open"]))
        self.high.append(float(candle["high"]))
        self.low.append(float(candle["low"]))
        self.close.append(float(candle["close"]))
        self.volume.append(float(candle["volume"]))
        self._last_candle_ts = ts
        self.bar_count += 1

    def update_funding(self, rate: float) -> None:
        """Append a funding rate observation (every ~8 h or from REST poll)."""
        self.funding_rate.append(rate)

    def update_oi(self, oi: float) -> None:
        """Append an open-interest observation (from periodic REST poll)."""
        self.open_interest.append(oi)

    def update_tick(self, price: float, qty: float, is_buyer_maker: bool, ts: int) -> None:
        """Append a raw exchange trade tick."""
        self.tick_prices.append(price)
        self.tick_volumes.append(qty)
        self.tick_sides.append(-1 if is_buyer_maker else 1)
        self.tick_timestamps.append(ts)

    def update_liquidation(self, event: "LiquidationEvent") -> None:
        """Append a real liquidation event from LiquidationFeed."""
        self.liquidation_events.append(event)

    @property
    def ready(self) -> bool:
        """True once sufficient bars exist for all indicators."""
        return len(self.close) >= 120
