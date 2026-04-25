"""
ShadowBot — runs the full APFTS v3 strategy pipeline against live market data
without sending any orders to the exchange.

Inherits from ProductionBot and overrides only the execution layer:
  - _open_position  → simulate limit/market fill with configurable slippage
  - _execute_exit   → simulate exit with slippage
  - start()         → skip set_leverage; use configured starting equity
  - _daily_report_loop → send shadow-specific report

Pending limit orders are tracked per-symbol and checked on every candle.
If the limit price is not touched within fill_timeout_bars, the order
is simulated as a market fill at the current close.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog

from config.config import AppConfig
from src.persistence.trade_store import ShadowTradeStore, TradeRecord
from src.production.bot import ProductionBot, V3BotAdapter
from src.strategy.engine import V3Signal

log = structlog.get_logger(__name__)


# =============================================================================
# Fill simulation helpers
# =============================================================================


def _apply_slippage(price: float, direction: int, slippage_bps: float) -> float:
    """Slippage is always adverse: longs pay more, shorts receive less."""
    factor = slippage_bps / 10_000
    return price * (1 + factor) if direction > 0 else price * (1 - factor)


# =============================================================================
# Shadow Bot
# =============================================================================


class ShadowBot(ProductionBot):
    """
    Drop-in replacement for ProductionBot that intercepts all order
    submission and routes it through a simulated fill engine.

    Zero CCXT private API calls:  no create_order, no cancel_order,
    no set_leverage, no fetch_balance.  Only public market-data endpoints
    are used (fetch_ohlcv, fetch_funding_rate, fetch_open_interest,
    fetch_order_book, fetch_ticker).
    """

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg)

        self._shadow_cfg = cfg.shadow
        self._shadow_store = ShadowTradeStore(cfg.database)

        # Pending limit orders waiting for fill confirmation
        # key = trade_id, value = {signal, direction, amount, limit_price, bars_left}
        self._pending_fills: Dict[str, dict] = {}

        # Trade IDs confirmed on the current candle — excluded from exit checks
        # until the next candle closes, matching real exchange behaviour where a
        # trailing stop cannot fire in the same bar the order fills.
        self._new_fills: set = set()

        # Separate structlog logger tagged with "shadow" prefix
        self._slog = structlog.get_logger("shadow")

        # Override capital with configured starting equity
        self._capital = cfg.shadow.starting_equity
        self._start_capital = cfg.shadow.starting_equity
        self._peak_capital = cfg.shadow.starting_equity

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect market-data client and start loops — skip trading setup."""
        # Connect CCXTClient for public market data only (no API key needed)
        await self._client.connect()

        # Skip: set_leverage (requires auth)
        # Skip: fetch_balance (use shadow starting equity instead)
        self._kill.initialise(self._capital)

        await self._notifier.start()
        await self._store.connect()
        await self._shadow_store.connect()

        log.info(
            "shadow_bot_started",
            symbol=self._cfg.trading.symbol,
            equity=self._capital,
            slippage_bps=self._shadow_cfg.slippage_bps,
            fill_timeout_bars=self._shadow_cfg.fill_timeout_bars,
        )

        self._running = True

        tasks: List[asyncio.Task] = []

        if self._cfg.trading.use_websocket:
            from src.execution.ws_stream import BinanceCandleStream
            self._ws_stream = BinanceCandleStream(
                symbol=self._cfg.trading.symbol,
                timeframe=self._cfg.trading.timeframe,
                on_candle=self._on_ws_candle,
                on_tick=self._adapter.feed_trade,
            )
            tasks.append(asyncio.create_task(self._ws_stream.run()))
            log.info("shadow_ws_stream_enabled")
        else:
            tasks.append(asyncio.create_task(self._candle_loop()))

        tasks += [
            asyncio.create_task(self._funding_oi_loop()),
            asyncio.create_task(self._position_monitor_loop()),
            asyncio.create_task(self._daily_report_loop()),
            asyncio.create_task(self._orderbook_loop()),
        ]

        if self._cfg.trading.live_liq_feed:
            from src.execution.liquidation_feed import LiquidationFeed
            self._liq_feed = LiquidationFeed(symbols={self._cfg.trading.symbol})
            self._liq_feed.register_callback(self._adapter._buf.update_liquidation)
            tasks.append(asyncio.create_task(self._liq_feed.run()))

        await asyncio.gather(*tasks)

    async def _shutdown(self) -> None:
        log.warning("shadow_shutdown")
        self._running = False
        if self._ws_stream:
            self._ws_stream.stop()
        if self._liq_feed:
            self._liq_feed.stop()
        # No real positions to close — just log pending
        for trade_id in list(self._pending_fills.keys()):
            self._slog.info("shadow_pending_abandoned", trade_id=trade_id)
        await self._notifier.close()
        await self._shadow_store.close()
        await self._store.close()
        await self._client.close()

    # ── Candle hook — check pending fills before signal eval ─────────────────

    async def _on_candle_close(self, candle: dict) -> None:
        # Trades from the previous candle can now be monitored for exits.
        self._new_fills.clear()
        await self._process_pending_fills(candle)
        # Count pending fills as occupied slots so a second signal can't fire
        # while we're waiting for a limit order to be confirmed.
        if len(self._pending_fills) + len(self._open_trades) >= self._cfg.risk.max_open_trades:
            return
        await super()._on_candle_close(candle)

    async def _process_pending_fills(self, candle: dict) -> None:
        """Advance each pending limit order by one bar; simulate fill if hit."""
        high = float(candle.get("high", candle.get("close", 0)))
        low  = float(candle.get("low",  candle.get("close", 0)))
        close = float(candle.get("close", 0))

        for trade_id in list(self._pending_fills.keys()):
            pending = self._pending_fills[trade_id]
            direction   = pending["direction"]
            limit_price = pending["limit_price"]
            amount      = pending["amount"]
            signal: V3Signal = pending["signal"]

            # Check if limit price was touched this bar
            if direction > 0 and low <= limit_price:
                fill_price = limit_price
            elif direction < 0 and high >= limit_price:
                fill_price = limit_price
            else:
                pending["bars_left"] -= 1
                if pending["bars_left"] <= 0:
                    # Timeout: market-fill at current close
                    fill_price = close
                    self._slog.info(
                        "shadow_limit_expired_market_fill",
                        trade_id=trade_id, limit_price=limit_price, fill_price=fill_price,
                    )
                else:
                    continue  # still waiting

            size_pct = pending["size_pct"]
            del self._pending_fills[trade_id]
            await self._confirm_shadow_fill(
                trade_id, signal, direction, amount, fill_price, size_pct
            )

    # ── Shadow entry (no exchange calls) ─────────────────────────────────────

    async def _open_position(self, signal: V3Signal) -> None:
        symbol    = self._cfg.trading.symbol
        direction = 1 if signal.direction == "LONG" else -1

        from src.risk.position_sizer import PositionSizer, SignalType, TradeSignal
        from src.risk.risk_manager import MarketRegime

        cur_dd = (self._peak_capital - self._capital) / (self._peak_capital + 1e-10) * 100
        mock_regime = MarketRegime.TRENDING if "trending" in signal.regime else MarketRegime.MEAN_REVERT
        mock_signal = TradeSignal(
            direction=SignalType.STRONG_LONG if direction > 0 else SignalType.STRONG_SHORT,
            confidence=signal.confidence,
            trend_score=signal.trend_score,
            momentum_score=signal.momentum_score,
            orderflow_score=signal.orderflow_score,
            regime=mock_regime,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit,
            take_profit_2=signal.take_profit,
            position_size_pct=0.0,
            timeframe_alignment=0.5,
            edge_quality=signal.grade,
            filters_passed=True,
        )
        size_pct = self._sizer.compute_size(
            mock_signal, self._capital, cur_dd, 0.55, 1.8, 0.002, 0.002
        )
        amount = self._capital * size_pct / signal.entry_price

        if amount < 0.001:
            self._slog.warning("shadow_position_too_small", amount=amount)
            return

        limit_price = signal.limit_entry_price or signal.entry_price
        trade_id    = str(uuid.uuid4())[:8]

        self._slog.info(
            "SHADOW_ORDER",
            side="buy" if direction > 0 else "sell",
            limit_price=limit_price,
            amount=amount,
            grade=signal.grade,
            confidence=f"{signal.confidence:.3f}",
            trade_id=trade_id,
        )

        # Queue for fill simulation on subsequent candles
        self._pending_fills[trade_id] = {
            "signal":      signal,
            "direction":   direction,
            "amount":      amount,
            "size_pct":    size_pct,
            "limit_price": limit_price,
            "bars_left":   self._shadow_cfg.fill_timeout_bars,
        }

    async def _confirm_shadow_fill(
        self,
        trade_id: str,
        signal: V3Signal,
        direction: int,
        amount: float,
        raw_fill_price: float,
        size_pct: float,
    ) -> None:
        """Apply slippage, register position, and persist."""
        actual_entry = _apply_slippage(raw_fill_price, direction, self._shadow_cfg.slippage_bps)
        symbol       = self._cfg.trading.symbol

        self._adapter.register_fill(
            trade_id, direction, actual_entry,
            signal.initial_risk, signal.stop_loss, signal.take_profit,
        )

        self._open_trades[trade_id] = {
            "direction":   direction,
            "entry_price": actual_entry,
            "amount":      amount,
            "stop_loss":   signal.stop_loss,
            "take_profit": signal.take_profit,
        }
        # Guard: do not evaluate exits until the next candle.
        self._new_fills.add(trade_id)

        self._slog.info(
            "shadow_fill_confirmed",
            trade_id=trade_id,
            direction=signal.direction,
            entry=actual_entry,
            sl=signal.stop_loss,
            tp=signal.take_profit,
            slippage_bps=self._shadow_cfg.slippage_bps,
        )

        rec = TradeRecord(
            trade_id=trade_id, symbol=symbol, direction=direction,
            entry_price=actual_entry, stop_loss=signal.stop_loss,
            take_profit=signal.take_profit, amount=amount,
            size_pct=size_pct, grade=signal.grade,
            confidence=signal.confidence,
        )
        await self._shadow_store.record_open(rec, slippage_bps=self._shadow_cfg.slippage_bps)

        await self._notifier.send_trade_opened(
            trade_id=trade_id, symbol=symbol, direction=signal.direction,
            entry=actual_entry, stop_loss=signal.stop_loss,
            take_profit=signal.take_profit, size_pct=size_pct,
            grade=signal.grade,
        )

    # ── Shadow position monitor ───────────────────────────────────────────────

    async def _position_monitor_loop(self) -> None:
        """
        Same cadence as ProductionBot (every 10 s) but with two corrections:
          1. Skip trades filled on the current candle (_new_fills guard) so a
             trailing stop cannot fire in the same bar the order was confirmed.
          2. Use ticker["last"] for both high and low instead of the 24-hour
             high/low, which would give the exit logic a misleadingly wide range.
        """
        while self._running and not self._kill.triggered:
            try:
                for trade_id in list(self._open_trades.keys()):
                    if trade_id in self._new_fills:
                        continue
                    ticker = await self._client.fetch_ticker(self._cfg.trading.symbol)
                    price = float(ticker["last"])
                    exit_info = self._adapter.check_exit(trade_id, price, price)
                    if exit_info:
                        await self._execute_exit(trade_id, exit_info)
            except Exception as exc:
                log.error("shadow_position_monitor_error", error=str(exc))
            await asyncio.sleep(10)

    # ── Shadow exit (no exchange calls) ──────────────────────────────────────

    async def _execute_exit(self, trade_id: str, exit_info: dict) -> None:
        if trade_id not in self._open_trades:
            return

        trade      = self._open_trades.pop(trade_id)
        symbol     = self._cfg.trading.symbol
        direction  = trade["direction"]
        raw_exit   = float(exit_info["exit_price"])
        reason     = exit_info["reason"]

        # For stop-loss exits slippage is adverse (fill beyond the stop)
        is_sl     = "stop_loss" in reason or "trail_phase" in reason
        slip_dir  = -direction if is_sl else direction
        actual_exit = _apply_slippage(raw_exit, slip_dir, self._shadow_cfg.slippage_bps)

        if direction > 0:
            pnl_pct = (actual_exit - trade["entry_price"]) / trade["entry_price"] * 100
        else:
            pnl_pct = (trade["entry_price"] - actual_exit) / trade["entry_price"] * 100

        self._capital *= (1 + pnl_pct / 100)
        self._peak_capital = max(self._peak_capital, self._capital)
        self._kill.update_capital(self._capital)

        self._slog.info(
            "SHADOW_CLOSE",
            trade_id=trade_id,
            reason=reason,
            exit=actual_exit,
            pnl_pct=f"{pnl_pct:+.3f}%",
            equity=f"{self._capital:.2f}",
        )

        await self._shadow_store.record_close(
            trade_id=trade_id,
            exit_price=actual_exit,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            capital_after=self._capital,
        )
        direction_str = "LONG" if direction > 0 else "SHORT"
        await self._notifier.send_trade_closed(
            trade_id=trade_id, symbol=symbol, direction=direction_str,
            entry=trade["entry_price"], exit_price=actual_exit,
            pnl_pct=pnl_pct, reason=reason, capital=self._capital,
        )

    # ── Shadow daily report ───────────────────────────────────────────────────

    async def _daily_report_loop(self) -> None:
        report_hour = self._cfg.notifications.daily_report_hour_utc

        while self._running and not self._kill.triggered:
            now = datetime.now(timezone.utc)
            next_hour = now.replace(minute=0, second=0, microsecond=0)
            if now.hour >= report_hour:
                from datetime import timedelta
                next_hour = next_hour.replace(hour=report_hour) + timedelta(days=1)
            else:
                next_hour = next_hour.replace(hour=report_hour)
            await asyncio.sleep((next_hour - now).total_seconds())

            try:
                summary = await self._shadow_store.daily_summary()
                total_pnl = (self._capital - self._start_capital) / (self._start_capital + 1e-10) * 100
                await self._notifier.send_shadow_report(
                    symbol=self._cfg.trading.symbol,
                    starting_equity=self._start_capital,
                    current_equity=self._capital,
                    trades_today=summary.get("trades", 0),
                    win_rate=summary.get("win_rate", 0.0),
                    total_pnl_pct=total_pnl,
                    avg_slippage_bps=summary.get("avg_slip_bps", 0.0),
                )
                self._start_capital = self._capital
            except Exception as exc:
                log.error("shadow_daily_report_error", error=str(exc))
