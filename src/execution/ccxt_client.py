"""
Async CCXT wrapper for perpetual futures exchanges.
Handles connection lifecycle, rate limiting, and testnet switching.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt

from config.config import ExchangeConfig

logger = logging.getLogger(__name__)


class CCXTClient:
    """Thin async wrapper around a CCXT exchange instance."""

    def __init__(self, cfg: ExchangeConfig) -> None:
        self._cfg = cfg
        self._exchange: Optional[ccxt.Exchange] = None

    async def connect(self) -> None:
        exchange_cls = getattr(ccxt, self._cfg.id)
        params: Dict[str, Any] = {
            "apiKey": self._cfg.api_key,
            "secret": self._cfg.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
        if self._cfg.passphrase:
            params["password"] = self._cfg.passphrase

        self._exchange = exchange_cls(params)

        if self._cfg.testnet:
            self._exchange.set_sandbox_mode(True)

        await self._exchange.load_markets()
        logger.info("CCXT connected to %s (testnet=%s)", self._cfg.id, self._cfg.testnet)

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            logger.info("CCXT connection closed")

    @property
    def exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            raise RuntimeError("Call connect() before accessing the exchange")
        return self._exchange

    async def fetch_balance(self) -> Dict[str, Any]:
        return await self.exchange.fetch_balance()

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return await self.exchange.fetch_ticker(symbol)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 500
    ) -> List[list]:
        return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def fetch_funding_rate(self, symbol: str) -> float:
        data = await self.exchange.fetch_funding_rate(symbol)
        return float(data.get("fundingRate", 0.0))

    async def fetch_open_interest(self, symbol: str) -> float:
        data = await self.exchange.fetch_open_interest(symbol)
        return float(data.get("openInterestAmount", 0.0))

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self.exchange.set_leverage(leverage, symbol)
        logger.info("Leverage set to %dx for %s", leverage, symbol)

    async def create_limit_order(
        self, symbol: str, side: str, amount: float, price: float,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return await self.exchange.create_limit_order(symbol, side, amount, price, params or {})

    async def create_market_order(
        self, symbol: str, side: str, amount: float, params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        return await self.exchange.create_market_order(symbol, side, amount, params or {})

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return await self.exchange.cancel_order(order_id, symbol)

    async def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return await self.exchange.fetch_order(order_id, symbol)

    async def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        symbols = [symbol] if symbol else None
        return await self.exchange.fetch_positions(symbols)

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch Level-2 order book snapshot. Returns {'bids': [...], 'asks': [...]}."""
        return await self.exchange.fetch_order_book(symbol, limit)

    async def fetch_next_funding_time(self, symbol: str) -> int:
        """
        Return the next funding settlement timestamp in milliseconds.
        Falls back to the locally-computed Binance schedule if the exchange
        does not expose this field.
        """
        try:
            data = await self.exchange.fetch_funding_rate(symbol)
            ts = data.get("nextFundingDatetime") or data.get("nextFundingTimestamp")
            if ts:
                return int(ts) if isinstance(ts, (int, float)) else int(ts) // 1000
        except Exception:
            pass

        from src.risk.funding_carry import next_funding_timestamp_ms
        return next_funding_timestamp_ms()
