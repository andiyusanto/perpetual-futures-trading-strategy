"""
Live liquidation event feed via Binance USDM WebSocket.

Stream:  wss://fstream.binance.com/ws/!forceOrder@arr
Each message:
  {
    "e": "forceOrder",
    "o": {
      "s": "BTCUSDT",   # symbol
      "S": "SELL",      # SELL = long liquidated, BUY = short liquidated
      "q": "0.001",     # qty
      "p": "50000",     # price
      "T": 1234567890   # time ms
    }
  }

REST fallback: CCXTClient.fetch_force_orders() polls /fapi/v1/forceOrders every N seconds
when websocket is unavailable or exchange != Binance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Set

import websockets

log = logging.getLogger(__name__)

BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws/!forceOrder@arr"


@dataclass
class LiquidationEvent:
    symbol: str          # e.g. "BTCUSDT"
    side: str            # "LONG_LIQ" or "SHORT_LIQ"
    price: float
    quantity: float
    timestamp: int       # ms epoch

    @property
    def usd_value(self) -> float:
        return self.price * self.quantity


def _ccxt_symbol_to_ws(symbol: str) -> str:
    """Convert 'BTC/USDT:USDT' → 'BTCUSDT'."""
    return symbol.replace("/", "").replace(":USDT", "").replace(":USD", "").upper()


class LiquidationFeed:
    """
    Subscribes to Binance !forceOrder@arr to receive real forced-liquidation
    events and exposes them as:
      - A deque of recent LiquidationEvent objects
      - A get_clusters() method for LiquidationMapper
      - A register_callback() API for per-event processing
    """

    def __init__(
        self,
        symbols: Optional[Set[str]] = None,
        max_events: int = 2_000,
        reconnect_delay: float = 5.0,
    ) -> None:
        self._filter: Optional[Set[str]] = (
            {_ccxt_symbol_to_ws(s) for s in symbols} if symbols else None
        )
        self._events: Deque[LiquidationEvent] = deque(maxlen=max_events)
        self._callbacks: List[Callable[[LiquidationEvent], None]] = []
        self._reconnect_delay = reconnect_delay
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_callback(self, cb: Callable[[LiquidationEvent], None]) -> None:
        self._callbacks.append(cb)

    @property
    def recent_events(self) -> Deque[LiquidationEvent]:
        return self._events

    def ingest(self, event: LiquidationEvent) -> None:
        """Manually push a liquidation event (used by REST fallback)."""
        if self._filter and event.symbol not in self._filter:
            return
        self._events.append(event)
        for cb in self._callbacks:
            cb(event)

    def get_clusters(
        self, symbol: str, lookback_ms: int = 300_000
    ) -> dict:
        """
        Aggregate recent events for *symbol* into price → USD-value buckets.

        Returns:
          {
            "long_liqs":  [(price, usd_value), ...],   # long positions liquidated
            "short_liqs": [(price, usd_value), ...],   # short positions liquidated
          }
        """
        sym = _ccxt_symbol_to_ws(symbol)
        cutoff = int(time.time() * 1000) - lookback_ms

        long_liqs: List[tuple[float, float]] = []
        short_liqs: List[tuple[float, float]] = []

        for ev in self._events:
            if ev.symbol != sym or ev.timestamp < cutoff:
                continue
            bucket = long_liqs if ev.side == "LONG_LIQ" else short_liqs
            bucket.append((ev.price, ev.usd_value))

        return {"long_liqs": long_liqs, "short_liqs": short_liqs}

    # ── WebSocket lifecycle ────────────────────────────────────────────────────

    async def run(self) -> None:
        """Blocking coroutine — run via asyncio.create_task()."""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    BINANCE_FUTURES_WS, ping_interval=20, ping_timeout=30
                ) as ws:
                    log.info("liq_feed_connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        self._handle(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("liq_feed_reconnecting", delay=self._reconnect_delay, error=str(exc))
                await asyncio.sleep(self._reconnect_delay)

    def stop(self) -> None:
        self._running = False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _handle(self, raw: str) -> None:
        try:
            data = json.loads(raw)
            order = data.get("o", {})
            symbol: str = order.get("s", "")

            if self._filter and symbol not in self._filter:
                return

            side_raw = order.get("S", "")
            # Binance convention: SELL = exchange sold (long position liquidated)
            #                     BUY  = exchange bought (short position liquidated)
            side = "LONG_LIQ" if side_raw == "SELL" else "SHORT_LIQ"

            event = LiquidationEvent(
                symbol=symbol,
                side=side,
                price=float(order.get("p", 0)),
                quantity=float(order.get("q", 0)),
                timestamp=int(order.get("T", int(time.time() * 1000))),
            )
            self._events.append(event)
            for cb in self._callbacks:
                try:
                    cb(event)
                except Exception:
                    pass
        except Exception as exc:
            log.debug("liq_feed_parse_error", error=str(exc))
