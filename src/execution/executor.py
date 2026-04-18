"""
Async order executor: handles submission, fill confirmation, and retries.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from src.execution.ccxt_client import CCXTClient

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_DELAY_S = 1.0


class OrderExecutor:
    """Submit, monitor, and cancel orders with automatic retry."""

    def __init__(self, client: CCXTClient) -> None:
        self._client = client

    async def submit_limit_entry(
        self,
        symbol: str,
        side: str,
        amount: float,
        limit_price: float,
        timeout_s: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a limit entry order and wait up to timeout_s for fill.
        Returns filled order dict, or None if expired unfilled.
        """
        order = await self._retry(
            self._client.create_limit_order, symbol, side, amount, limit_price,
            {"timeInForce": "GTC", "postOnly": True},
        )
        if order is None:
            return None

        order_id: str = order["id"]
        deadline = asyncio.get_event_loop().time() + timeout_s
        poll_interval = 2.0

        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(poll_interval)
            status = await self._retry(self._client.fetch_order, order_id, symbol)
            if status and status.get("status") == "closed":
                logger.info("Limit order %s filled at %.4f", order_id, limit_price)
                return status

        await self._retry(self._client.cancel_order, order_id, symbol)
        logger.info("Limit order %s cancelled (timeout)", order_id)
        return None

    async def submit_market_exit(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Place a market exit order (stop-loss hit or kill-switch)."""
        params: Dict[str, Any] = {"reduceOnly": reduce_only}
        return await self._retry(self._client.create_market_order, symbol, side, amount, params)

    async def submit_limit_exit(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Place a limit take-profit order (maker fee)."""
        params: Dict[str, Any] = {"reduceOnly": reduce_only, "postOnly": True}
        return await self._retry(self._client.create_limit_order, symbol, side, amount, price, params)

    async def _retry(self, fn, *args, **kwargs) -> Optional[Any]:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Order attempt %d/%d failed: %s", attempt + 1, _RETRY_ATTEMPTS, exc
                )
                if attempt < _RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(_RETRY_DELAY_S * (attempt + 1))
        logger.error("Order failed after %d attempts", _RETRY_ATTEMPTS)
        return None
