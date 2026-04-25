"""
AlertNotifier — send real-time trade alerts and daily PnL reports via
Telegram Bot API and/or Discord webhooks.

Both transports are optional and independently enabled via config.
All HTTP calls use aiohttp (already a project dependency) so the notifier
is fully non-blocking inside the async bot event loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from config.config import NotificationConfig

logger = logging.getLogger(__name__)

_TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"


class AlertNotifier:
    """
    Async alert dispatcher.  Both Telegram and Discord are fire-and-forget:
    network errors are logged but never propagate to the trading loop.
    """

    def __init__(self, cfg: NotificationConfig) -> None:
        self._cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Session lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Public alert methods ──────────────────────────────────────────────────

    async def send_trade_opened(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        size_pct: float,
        grade: str,
    ) -> None:
        if not self._cfg.notify_on_trade:
            return
        risk_r = abs(take_profit - entry) / max(abs(entry - stop_loss), 1e-10)
        text = (
            f"🟢 *POSITION OPENED*\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: `{direction}` | Grade: `{grade}`\n"
            f"Entry: `{entry:.4f}`\n"
            f"Stop:  `{stop_loss:.4f}`\n"
            f"TP:    `{take_profit:.4f}`  ({risk_r:.1f}R)\n"
            f"Size:  `{size_pct:.1%}` of equity\n"
            f"ID:    `{trade_id}`"
        )
        await self._dispatch(text)

    async def send_trade_closed(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry: float,
        exit_price: float,
        pnl_pct: float,
        reason: str,
        capital: float,
    ) -> None:
        if not self._cfg.notify_on_trade:
            return
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        text = (
            f"{emoji} *POSITION CLOSED*\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: `{direction}` | Reason: `{reason}`\n"
            f"Entry:  `{entry:.4f}`\n"
            f"Exit:   `{exit_price:.4f}`\n"
            f"PnL:    `{pnl_pct:+.3f}%`\n"
            f"Equity: `{capital:.2f} USDT`\n"
            f"ID:     `{trade_id}`"
        )
        await self._dispatch(text)

    async def send_daily_report(
        self,
        symbol: str,
        starting_capital: float,
        current_capital: float,
        trades_today: int,
        win_rate: float,
        total_pnl_pct: float,
        max_drawdown_pct: float,
    ) -> None:
        if not self._cfg.notify_on_pnl:
            return
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        emoji = "📈" if total_pnl_pct >= 0 else "📉"
        text = (
            f"{emoji} *DAILY PnL REPORT*  {now_utc}\n"
            f"Symbol:   `{symbol}`\n"
            f"Equity:   `{current_capital:.2f} USDT`  "
            f"(`{total_pnl_pct:+.2f}%` vs start)\n"
            f"Trades:   `{trades_today}` | Win rate: `{win_rate:.0%}`\n"
            f"Max DD:   `{max_drawdown_pct:.2f}%`"
        )
        await self._dispatch(text)

    async def send_shadow_report(
        self,
        symbol: str,
        starting_equity: float,
        current_equity: float,
        trades_today: int,
        win_rate: float,
        total_pnl_pct: float,
        avg_slippage_bps: float,
        real_pnl_pct: Optional[float] = None,
    ) -> None:
        if not self._cfg.notify_on_pnl:
            return
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        emoji = "🔵" if total_pnl_pct >= 0 else "🟠"
        lines = [
            f"{emoji} *SHADOW MODE — DAILY REPORT*  {now_utc}",
            f"Symbol:   `{symbol}`",
            f"Equity:   `{current_equity:.2f} USDT`  (`{total_pnl_pct:+.2f}%`)",
            f"Trades:   `{trades_today}` | Win rate: `{win_rate:.0%}`",
            f"Avg slip: `{avg_slippage_bps:.1f} bps`",
        ]
        if real_pnl_pct is not None:
            delta = total_pnl_pct - real_pnl_pct
            lines.append(f"vs Real:  `{real_pnl_pct:+.2f}%`  (delta `{delta:+.2f}%`)")
        await self._dispatch("\n".join(lines))

    async def send_alert(self, message: str) -> None:
        """Generic system alert (kill switch, errors, reconnects)."""
        await self._dispatch(f"⚠️ *ALERT*\n{message}")

    async def send_kill_switch(self, reason: str, capital: float) -> None:
        text = (
            f"🚨 *KILL SWITCH TRIGGERED*\n"
            f"Reason:  `{reason}`\n"
            f"Capital: `{capital:.2f} USDT`"
        )
        await self._dispatch(text)

    # ── Internal dispatch ─────────────────────────────────────────────────────

    async def _dispatch(self, text: str) -> None:
        """Fan-out to all enabled transports, swallowing errors."""
        await asyncio.gather(
            self._send_telegram(text),
            self._send_discord(text),
            return_exceptions=True,
        )

    async def _send_telegram(self, text: str) -> None:
        if not self._cfg.telegram_token or not self._cfg.telegram_chat_id:
            return
        await self.start()
        url = _TELEGRAM_URL.format(token=self._cfg.telegram_token)
        payload = {
            "chat_id": self._cfg.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            async with self._session.post(url, json=payload) as resp:  # type: ignore[union-attr]
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("telegram_error status=%s body=%s", resp.status, body[:200])
        except Exception as exc:
            logger.warning("telegram_send_failed: %s", exc)

    async def _send_discord(self, text: str) -> None:
        if not self._cfg.discord_webhook_url:
            return
        await self.start()
        # Discord uses "content" field; strip Markdown bold markers for cleaner rendering
        content = text.replace("*", "**")
        try:
            async with self._session.post(  # type: ignore[union-attr]
                self._cfg.discord_webhook_url, json={"content": content}
            ) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.warning("discord_error status=%s body=%s", resp.status, body[:200])
        except Exception as exc:
            logger.warning("discord_send_failed: %s", exc)
