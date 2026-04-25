"""
Binance USDM WebSocket candle + aggTrade stream.

Replaces the 30-second REST polling loop in ProductionBot with true server-push
kline events.  Latency drops from ~30 s to ~50 ms for candle closes.

Streams used:
  <sym>@kline_<tf>  — closed-candle events  (x=True flag)
  <sym>@aggTrade    — aggregated trade ticks (for orderflow CVD)

WS URL: wss://fstream.binance.com/stream?streams=btcusdt@kline_1m/btcusdt@aggTrade
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

log = logging.getLogger(__name__)

_WS_BASE = "wss://fstream.binance.com/stream"


def _ws_symbol(ccxt_symbol: str) -> str:
    """'BTC/USDT:USDT' → 'btcusdt'"""
    return ccxt_symbol.replace("/", "").replace(":USDT", "").replace(":USD", "").lower()


class BinanceCandleStream:
    """
    Async WebSocket subscriber for closed klines and aggTrade ticks.

    Callbacks fire on the calling event-loop thread — keep them non-blocking.
    Feed the outputs straight into MarketDataBuffer / ProductionBot handlers:

        stream = BinanceCandleStream(
            symbol="BTC/USDT:USDT",
            timeframe="1m",
            on_candle=bot._on_ws_candle,
            on_tick=adapter.feed_trade,
        )
        asyncio.create_task(stream.run())
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1m",
        on_candle: Optional[Callable[[dict], None]] = None,
        on_tick: Optional[Callable[[dict], None]] = None,
        reconnect_delay: float = 3.0,
    ) -> None:
        self._sym = _ws_symbol(symbol)
        self._tf = timeframe
        self._on_candle = on_candle
        self._on_tick = on_tick
        self._reconnect_delay = reconnect_delay
        self._running = False

    @property
    def _url(self) -> str:
        streams = f"{self._sym}@kline_{self._tf}/{self._sym}@aggTrade"
        return f"{_WS_BASE}?streams={streams}"

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Blocking coroutine — launch via asyncio.create_task()."""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    self._url, ping_interval=20, ping_timeout=30
                ) as ws:
                    log.info("ws_stream_connected", symbol=self._sym, tf=self._tf)
                    async for raw in ws:
                        if not self._running:
                            break
                        self._dispatch(json.loads(raw))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("ws_stream_reconnecting",
                            symbol=self._sym, delay=self._reconnect_delay, error=str(exc))
                await asyncio.sleep(self._reconnect_delay)
        log.info("ws_stream_stopped", symbol=self._sym)

    def stop(self) -> None:
        self._running = False

    # ── Message dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, msg: dict) -> None:
        stream: str = msg.get("stream", "")
        data: dict = msg.get("data", msg)

        if "@kline_" in stream:
            self._handle_kline(data)
        elif "@aggTrade" in stream:
            self._handle_agg_trade(data)

    def _handle_kline(self, data: dict) -> None:
        k = data.get("k", {})
        if not k.get("x", False):   # x=True only when the candle is *closed*
            return
        if self._on_candle:
            candle = {
                "timestamp": int(k["t"]),
                "open":   float(k["o"]),
                "high":   float(k["h"]),
                "low":    float(k["l"]),
                "close":  float(k["c"]),
                "volume": float(k["v"]),
            }
            try:
                self._on_candle(candle)
            except Exception as exc:
                log.error("ws_on_candle_error", error=str(exc))

    def _handle_agg_trade(self, data: dict) -> None:
        if self._on_tick:
            tick = {
                "p": data.get("p"),   # price string
                "q": data.get("q"),   # qty string
                "m": data.get("m"),   # is_buyer_maker bool
                "E": data.get("E"),   # event time ms
            }
            try:
                self._on_tick(tick)
            except Exception as exc:
                log.error("ws_on_tick_error", error=str(exc))
