"""
Kill switch + graceful shutdown handler.
Monitors drawdown, daily loss, and manual OS signals.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Callable, List

logger = logging.getLogger(__name__)


class KillSwitch:
    """
    Three activation paths:
      1. Drawdown breach   — equity dropped below kill_switch_drawdown threshold
      2. Daily loss breach — intraday PnL exceeded max_daily_loss_pct
      3. Manual signal     — SIGTERM / SIGINT or explicit trigger() call
    """

    def __init__(
        self,
        kill_drawdown_pct: float = 20.0,
        max_daily_loss_pct: float = 5.0,
    ) -> None:
        self.kill_drawdown_pct = kill_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct

        self._triggered = False
        self._reason = ""
        self._callbacks: List[Callable] = []
        self._peak_capital: float = 0.0
        self._day_start_capital: float = 0.0

    def initialise(self, starting_capital: float) -> None:
        self._peak_capital = starting_capital
        self._day_start_capital = starting_capital
        self._install_os_handlers()

    def on_shutdown(self, callback: Callable) -> None:
        """Register a coroutine or callable to run when kill switch fires."""
        self._callbacks.append(callback)

    def update_capital(self, current_capital: float) -> bool:
        """Call after every trade. Returns True if kill switch was just triggered."""
        if self._triggered:
            return False

        self._peak_capital = max(self._peak_capital, current_capital)
        dd = (self._peak_capital - current_capital) / self._peak_capital * 100
        daily_loss = (self._day_start_capital - current_capital) / self._day_start_capital * 100

        if dd >= self.kill_drawdown_pct:
            return self.trigger(f"Max drawdown breached: {dd:.1f}%")
        if daily_loss >= self.max_daily_loss_pct:
            return self.trigger(f"Daily loss limit breached: {daily_loss:.1f}%")
        return False

    def reset_daily(self, current_capital: float) -> None:
        """Call at the start of each UTC day."""
        self._day_start_capital = current_capital

    @property
    def triggered(self) -> bool:
        return self._triggered

    def trigger(self, reason: str = "manual") -> bool:
        if self._triggered:
            return False
        self._triggered = True
        self._reason = reason
        logger.critical("KILL SWITCH ACTIVATED — %s", reason)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._run_callbacks())
        except RuntimeError:
            pass
        return True

    async def _run_callbacks(self) -> None:
        for cb in self._callbacks:
            try:
                result = cb()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("Kill-switch callback error: %s", exc)

    def _install_os_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda s=sig: self.trigger(f"{s.name} received"))
        except (NotImplementedError, RuntimeError):
            pass
