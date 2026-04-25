"""
Multi-symbol orchestrator.

Runs one independent ProductionBot per symbol concurrently.
Each bot has its own strategy engine, data buffer, and position tracker.
All bots share the same exchange credentials and risk config from .env.

Usage:
    apfts-multi-bot                          # reads TRADING_SYMBOLS from .env
    TRADING_SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT apfts-multi-bot

.env example:
    TRADING_SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

import structlog

from config.config import AppConfig, TradingConfig
from src.production.bot import ProductionBot, _configure_logging

log = structlog.get_logger(__name__)


class MultiSymbolBot:
    """
    Instantiates one bot per symbol and runs them concurrently via asyncio.gather().
    Uses ShadowBot when SHADOW_ENABLED=true, ProductionBot otherwise.
    Each bot operates in complete isolation — separate buffers, engines, and trackers.
    """

    def __init__(self, cfg: AppConfig) -> None:
        symbols: List[str] = list(cfg.trading.symbols) if cfg.trading.symbols else [cfg.trading.symbol]

        if not symbols:
            raise ValueError("No symbols configured. Set TRADING_SYMBOLS or TRADING_SYMBOL in .env")

        if cfg.shadow.enabled:
            from src.shadow.engine import ShadowBot
            BotClass = ShadowBot
            log.info("multi_bot_shadow_mode", slippage_bps=cfg.shadow.slippage_bps)
        else:
            BotClass = ProductionBot

        self._bots: List[ProductionBot] = []
        for sym in symbols:
            tc = TradingConfig(
                symbol=sym,
                leverage=cfg.trading.leverage,
                timeframe=cfg.trading.timeframe,
                max_position_pct=cfg.trading.max_position_pct,
                use_websocket=cfg.trading.use_websocket,
                live_liq_feed=cfg.trading.live_liq_feed,
                symbols=cfg.trading.symbols,
            )
            sym_cfg = cfg.model_copy(update={"trading": tc})
            self._bots.append(BotClass(sym_cfg))

        log.info("multi_bot_init", n_symbols=len(self._bots), symbols=symbols)

    async def start(self) -> None:
        log.info("multi_bot_starting", n_symbols=len(self._bots))
        results = await asyncio.gather(
            *[bot.start() for bot in self._bots],
            return_exceptions=True,
        )
        for sym, result in zip([b._cfg.trading.symbol for b in self._bots], results):
            if isinstance(result, BaseException):
                log.error("bot_crashed", symbol=sym, error=str(result), exc_info=result)


def main() -> None:
    cfg = AppConfig()
    _configure_logging(cfg.log_level, cfg.log_dir)
    bot = MultiSymbolBot(cfg)
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
