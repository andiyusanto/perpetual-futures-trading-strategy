"""
APFTS v3 — Async Production Bot

Run:
    apfts-bot
    python -m src.production.bot
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

import structlog

from config.config import AppConfig
from src.core.data_buffer import MarketDataBuffer
from src.execution.ccxt_client import CCXTClient
from src.execution.executor import OrderExecutor
from src.execution.liquidation_feed import LiquidationFeed
from src.execution.ws_stream import BinanceCandleStream
from src.notifications.notifier import AlertNotifier
from src.persistence.trade_store import TradeRecord, TradeStore
from src.risk.kill_switch import KillSwitch
from src.risk.position_sizer import PositionSizer, SignalType, TradeSignal
from src.risk.risk_manager import MarketRegime, RegimeClassifier
from src.strategy.engine import V3Signal, V3StrategyEngine

log = structlog.get_logger(__name__)


# =============================================================================
# Async Adapter (thin async wrapper over the synchronous engine)
# =============================================================================


class V3BotAdapter:
    """
    Plugs V3StrategyEngine into the async WebSocket loop.

    Feed order each candle:
      1. feed_candle()  — closed OHLCV bar
      2. evaluate()     — generate signal (runs engine in executor)
      3. check_exit()   — chandelier update for open positions
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._buf = MarketDataBuffer(capacity=500)
        self._engine = V3StrategyEngine(cfg.strategy)
        self._engine.attach_buffer(self._buf)
        self._lock = asyncio.Lock()

    def feed_trade(self, data: dict) -> None:
        price = float(data.get("p", 0))
        qty   = float(data.get("q", 0))
        maker = bool(data.get("m", False))
        ts    = int(data.get("E", time.time() * 1000))
        self._buf.update_tick(price, qty, maker, ts)

    def feed_candle(self, candle: dict) -> None:
        self._buf.update_candle(candle)

    def feed_funding(self, rate: float) -> None:
        self._buf.update_funding(rate)

    def feed_oi(self, oi: float) -> None:
        self._buf.update_oi(oi)

    async def evaluate(self) -> V3Signal:
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._engine.generate_signal)

    def register_fill(
        self, trade_id: str, direction: int, entry_price: float,
        initial_risk: float, stop_loss: float, take_profit: float,
    ) -> None:
        self._engine.register_position(
            trade_id, direction, entry_price, initial_risk, stop_loss, take_profit
        )

    def check_exit(self, trade_id: str, high: float, low: float) -> Optional[dict]:
        return self._engine.update_exit(trade_id, high, low)

    def get_stop(self, trade_id: str) -> Optional[float]:
        return self._engine.get_current_stop(trade_id)


# =============================================================================
# Main Bot
# =============================================================================


class ProductionBot:
    """
    Full async production bot:
      - REST polling for closed candles (every 30 s)
      - REST polling for funding rate + OI (every 5 min)
      - Signal evaluation on each new candle
      - Limit-first order execution with market fallback
      - Chandelier exit monitoring (every 10 s)
      - Kill-switch + graceful shutdown
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._adapter  = V3BotAdapter(cfg)
        self._client   = CCXTClient(cfg.exchange)
        self._executor = OrderExecutor(self._client)
        self._sizer    = PositionSizer(
            base_risk_pct=cfg.risk.base_risk_pct,
            max_size_pct=cfg.risk.max_position_size_pct,
        )
        self._kill = KillSwitch(
            kill_drawdown_pct=cfg.risk.kill_switch_drawdown,
            max_daily_loss_pct=cfg.risk.max_daily_loss_pct,
        )
        self._kill.on_shutdown(self._shutdown)

        self._notifier = AlertNotifier(cfg.notifications)
        self._store    = TradeStore(cfg.database)

        self._open_trades: Dict[str, dict] = {}
        self._capital: float = 0.0
        self._start_capital: float = 0.0          # for daily report
        self._peak_capital: float = 0.0
        self._running = False
        self._last_candle_ts: int = 0

        # Optional improvements (activated via config flags)
        self._ws_stream: Optional[BinanceCandleStream] = None
        self._liq_feed: Optional[LiquidationFeed] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._client.connect()
        await self._client.set_leverage(self._cfg.trading.symbol, self._cfg.trading.leverage)

        balance = await self._client.fetch_balance()
        usdt = balance.get("USDT", {})
        self._capital = float(usdt.get("free", 0.0))
        self._start_capital = self._capital
        self._peak_capital = self._capital
        self._kill.initialise(self._capital)

        await self._notifier.start()
        await self._store.connect()

        log.info("bot_started",
                 symbol=self._cfg.trading.symbol,
                 capital=self._capital,
                 leverage=self._cfg.trading.leverage)

        self._running = True

        tasks = [
            self._funding_oi_loop(),
            self._position_monitor_loop(),
            self._daily_report_loop(),
        ]

        if self._cfg.trading.use_websocket:
            # WebSocket stream replaces the 30-second REST candle polling loop.
            # on_candle fires synchronously in the WS receive loop — wrap in
            # asyncio.create_task so heavy signal work doesn't block the stream.
            self._ws_stream = BinanceCandleStream(
                symbol=self._cfg.trading.symbol,
                timeframe=self._cfg.trading.timeframe,
                on_candle=self._on_ws_candle,
                on_tick=self._adapter.feed_trade,
            )
            tasks.append(self._ws_stream.run())
            log.info("ws_stream_enabled", symbol=self._cfg.trading.symbol)
        else:
            tasks.append(self._candle_loop())

        if self._cfg.trading.live_liq_feed:
            self._liq_feed = LiquidationFeed(
                symbols={self._cfg.trading.symbol},
            )
            # Wire events into MarketDataBuffer so V3StrategyEngine can read them
            self._liq_feed.register_callback(self._adapter._buf.update_liquidation)
            tasks.append(self._liq_feed.run())
            log.info("live_liq_feed_enabled")

        tasks.append(self._orderbook_loop())

        await asyncio.gather(*tasks)

    async def _shutdown(self) -> None:
        log.warning("graceful_shutdown_initiated")
        self._running = False

        if self._ws_stream:
            self._ws_stream.stop()
        if self._liq_feed:
            self._liq_feed.stop()

        for trade_id, info in list(self._open_trades.items()):
            side = "sell" if info["direction"] > 0 else "buy"
            await self._executor.submit_market_exit(
                self._cfg.trading.symbol, side, info["amount"]
            )
            log.info("emergency_close", trade_id=trade_id)

        await self._notifier.send_kill_switch("graceful_shutdown", self._capital)
        await self._notifier.close()
        await self._store.close()
        await self._client.close()
        log.info("bot_stopped")

    # ── Data loops ────────────────────────────────────────────────────────────

    async def _candle_loop(self) -> None:
        symbol    = self._cfg.trading.symbol
        timeframe = self._cfg.trading.timeframe

        while self._running and not self._kill.triggered:
            try:
                bars = await self._client.fetch_ohlcv(symbol, timeframe, limit=10)
                for bar in bars[:-1]:  # Last bar is still open
                    ts = int(bar[0])
                    if ts <= self._last_candle_ts:
                        continue
                    candle = {
                        "timestamp": ts,
                        "open": bar[1], "high": bar[2],
                        "low": bar[3],  "close": bar[4],
                        "volume": bar[5],
                    }
                    self._adapter.feed_candle(candle)
                    self._last_candle_ts = ts
                    await self._on_candle_close(candle)
            except Exception as exc:
                log.error("candle_loop_error", error=str(exc))

            await asyncio.sleep(30)

    def _on_ws_candle(self, candle: dict) -> None:
        """Synchronous callback from BinanceCandleStream — schedule async work."""
        self._adapter.feed_candle(candle)
        self._last_candle_ts = candle["timestamp"]
        asyncio.get_event_loop().create_task(self._on_candle_close(candle))

    async def _funding_oi_loop(self) -> None:
        symbol = self._cfg.trading.symbol

        while self._running and not self._kill.triggered:
            try:
                fr = await self._client.fetch_funding_rate(symbol)
                oi = await self._client.fetch_open_interest(symbol)
                next_ts = await self._client.fetch_next_funding_time(symbol)
                self._adapter.feed_funding(fr)
                self._adapter.feed_oi(oi)
                self._adapter._buf.next_funding_ts_ms = next_ts
                log.debug("funding_oi_updated", fr=fr, oi=oi, next_funding_ts=next_ts)
            except Exception as exc:
                log.error("funding_oi_loop_error", error=str(exc))

            await asyncio.sleep(300)

    async def _daily_report_loop(self) -> None:
        """Fire a PnL report every day at the configured UTC hour."""
        report_hour = self._cfg.notifications.daily_report_hour_utc

        while self._running and not self._kill.triggered:
            now = datetime.now(timezone.utc)
            # Seconds until next report window
            next_hour = now.replace(minute=0, second=0, microsecond=0)
            if now.hour >= report_hour:
                from datetime import timedelta
                next_hour = next_hour.replace(hour=report_hour) + timedelta(days=1)
            else:
                next_hour = next_hour.replace(hour=report_hour)
            wait_s = (next_hour - now).total_seconds()
            await asyncio.sleep(wait_s)

            try:
                summary = await self._store.daily_summary()
                total_pnl = (self._capital - self._start_capital) / (self._start_capital + 1e-10) * 100
                max_dd = (self._peak_capital - self._capital) / (self._peak_capital + 1e-10) * 100
                await self._notifier.send_daily_report(
                    symbol=self._cfg.trading.symbol,
                    starting_capital=self._start_capital,
                    current_capital=self._capital,
                    trades_today=summary.get("trades", 0),
                    win_rate=summary.get("win_rate", 0.0),
                    total_pnl_pct=total_pnl,
                    max_drawdown_pct=max_dd,
                )
                self._start_capital = self._capital  # reset baseline for next day
            except Exception as exc:
                log.error("daily_report_error", error=str(exc))

    async def _orderbook_loop(self) -> None:
        """Poll L2 order book every 5 seconds to feed the OB imbalance signal."""
        symbol = self._cfg.trading.symbol

        while self._running and not self._kill.triggered:
            try:
                ob = await self._client.fetch_order_book(symbol, limit=20)
                self._adapter._buf.ob_bids = ob.get("bids", [])
                self._adapter._buf.ob_asks = ob.get("asks", [])
                self._adapter._buf.ob_timestamp = int(time.time() * 1000)
                log.debug("orderbook_updated",
                          best_bid=ob["bids"][0][0] if ob.get("bids") else None,
                          best_ask=ob["asks"][0][0] if ob.get("asks") else None)
            except Exception as exc:
                log.error("orderbook_loop_error", error=str(exc))

            await asyncio.sleep(5)

    async def _position_monitor_loop(self) -> None:
        while self._running and not self._kill.triggered:
            try:
                for trade_id in list(self._open_trades.keys()):
                    ticker = await self._client.fetch_ticker(self._cfg.trading.symbol)
                    high = float(ticker.get("high", ticker["last"]))
                    low  = float(ticker.get("low",  ticker["last"]))
                    exit_info = self._adapter.check_exit(trade_id, high, low)
                    if exit_info:
                        await self._execute_exit(trade_id, exit_info)
            except Exception as exc:
                log.error("position_monitor_error", error=str(exc))

            await asyncio.sleep(10)

    # ── Signal -> execution ───────────────────────────────────────────────────

    async def _on_candle_close(self, candle: dict) -> None:
        if self._kill.triggered:
            return
        if len(self._open_trades) >= self._cfg.risk.max_open_trades:
            return

        signal = await self._adapter.evaluate()

        if not signal.is_tradeable():
            log.debug("signal_skipped",
                      direction=signal.direction, grade=signal.grade,
                      vol_phase=signal.vol_phase)
            return

        log.info("signal_generated",
                 direction=signal.direction, grade=signal.grade,
                 confidence=f"{signal.confidence:.3f}",
                 composite=f"{signal.composite:.3f}",
                 trend=f"{signal.trend_score:.3f}",
                 momentum=f"{signal.momentum_score:.3f}",
                 orderflow=f"{signal.orderflow_score:.3f}",
                 funding=f"{signal.funding_score:.3f}",
                 orderbook=f"{signal.orderbook_score:.3f}",
                 vol_phase=signal.vol_phase, regime=signal.regime,
                 squeeze_fired=signal.squeeze_fired)

        await self._open_position(signal)

    async def _open_position(self, signal: V3Signal) -> None:
        symbol    = self._cfg.trading.symbol
        direction = 1 if signal.direction == "LONG" else -1
        side      = "buy" if direction > 0 else "sell"

        cur_dd  = (self._peak_capital - self._capital) / (self._peak_capital + 1e-10) * 100
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
            log.warning("position_too_small", amount=amount)
            return

        limit_price = signal.limit_entry_price or signal.entry_price
        filled = await self._executor.submit_limit_entry(
            symbol, side, amount, limit_price, timeout_s=90.0
        )

        if filled is None:
            log.info("limit_expired_using_market", limit_price=limit_price)
            filled = await self._executor.submit_market_exit(symbol, side, amount, reduce_only=False)

        if filled is None:
            log.error("entry_order_failed")
            return

        actual_entry = float(filled.get("average", signal.entry_price))
        trade_id = str(uuid.uuid4())[:8]

        self._adapter.register_fill(
            trade_id, direction, actual_entry,
            signal.initial_risk, signal.stop_loss, signal.take_profit,
        )
        self._open_trades[trade_id] = {
            "direction": direction,
            "entry_price": actual_entry,
            "amount": amount,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
        }

        log.info("position_opened",
                 trade_id=trade_id, direction=signal.direction,
                 entry=actual_entry, sl=signal.stop_loss, tp=signal.take_profit,
                 size_pct=f"{size_pct:.1%}")

        rec = TradeRecord(
            trade_id=trade_id, symbol=symbol, direction=direction,
            entry_price=actual_entry, stop_loss=signal.stop_loss,
            take_profit=signal.take_profit, amount=amount,
            size_pct=size_pct, grade=signal.grade,
            confidence=signal.confidence,
        )
        await self._store.record_open(rec)
        await self._notifier.send_trade_opened(
            trade_id=trade_id, symbol=symbol, direction=signal.direction,
            entry=actual_entry, stop_loss=signal.stop_loss,
            take_profit=signal.take_profit, size_pct=size_pct,
            grade=signal.grade,
        )

    async def _execute_exit(self, trade_id: str, exit_info: dict) -> None:
        if trade_id not in self._open_trades:
            return

        trade      = self._open_trades.pop(trade_id)
        symbol     = self._cfg.trading.symbol
        side       = "sell" if trade["direction"] > 0 else "buy"
        amount     = trade["amount"]
        use_maker  = exit_info.get("use_maker", False)
        exit_price = float(exit_info["exit_price"])

        if use_maker:
            order = await self._executor.submit_limit_exit(symbol, side, amount, exit_price)
        else:
            order = await self._executor.submit_market_exit(symbol, side, amount)

        if order:
            actual_exit = float(order.get("average", exit_price))
            if trade["direction"] > 0:
                pnl_pct = (actual_exit - trade["entry_price"]) / trade["entry_price"] * 100
            else:
                pnl_pct = (trade["entry_price"] - actual_exit) / trade["entry_price"] * 100

            self._capital *= (1 + pnl_pct / 100)
            self._peak_capital = max(self._peak_capital, self._capital)
            self._kill.update_capital(self._capital)

            log.info("position_closed",
                     trade_id=trade_id, reason=exit_info["reason"],
                     exit=actual_exit, pnl_pct=f"{pnl_pct:+.3f}%",
                     capital=f"{self._capital:.2f}")

            await self._store.record_close(
                trade_id=trade_id,
                exit_price=actual_exit,
                pnl_pct=pnl_pct,
                exit_reason=exit_info["reason"],
                capital_after=self._capital,
            )
            direction_str = "LONG" if trade["direction"] > 0 else "SHORT"
            await self._notifier.send_trade_closed(
                trade_id=trade_id, symbol=self._cfg.trading.symbol,
                direction=direction_str,
                entry=trade["entry_price"], exit_price=actual_exit,
                pnl_pct=pnl_pct, reason=exit_info["reason"],
                capital=self._capital,
            )


# =============================================================================
# CLI entry point
# =============================================================================


def _configure_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    import os
    os.makedirs(log_dir, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )

    file_handler = logging.FileHandler(f"{log_dir}/bot.log", mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def main() -> None:
    cfg = AppConfig()
    _configure_logging(cfg.log_level, cfg.log_dir)

    if cfg.shadow.enabled:
        from src.shadow.engine import ShadowBot
        _configure_logging(cfg.shadow.log_level, cfg.log_dir)
        bot: ProductionBot = ShadowBot(cfg)
        log.info("running_in_shadow_mode", slippage_bps=cfg.shadow.slippage_bps)
    else:
        bot = ProductionBot(cfg)

    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
