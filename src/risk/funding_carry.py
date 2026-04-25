"""
Funding carry management for perpetual futures positions.

Two complementary behaviours:

1. HOLD for carry  — when the next funding payment is <N minutes away and the
   position earns that payment (positive carry), suppress chandelier exits and
   position-monitor exits until after funding is collected.

2. EXIT before funding — when the next payment is <15 minutes away and the
   position *pays* funding at a rate exceeding a threshold, trigger an early
   limit exit to avoid the charge.

Binance USDM funding schedule: 00:00, 08:00, 16:00 UTC (every 8 hours).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Tuple


_BINANCE_FUNDING_HOURS = (0, 8, 16)


def next_funding_timestamp_ms() -> int:
    """Return the next Binance USDM funding timestamp in milliseconds UTC."""
    now = datetime.now(tz=timezone.utc)
    for h in _BINANCE_FUNDING_HOURS:
        candidate = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if candidate > now:
            return int(candidate.timestamp() * 1000)
    next_day = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int(next_day.timestamp() * 1000)


def minutes_to_next_funding() -> float:
    """Convenience: how many minutes until the next funding settlement."""
    ms_remaining = next_funding_timestamp_ms() - int(time.time() * 1000)
    return ms_remaining / 60_000


class FundingCarryManager:
    """
    Funding carry decision engine.

    Parameters
    ----------
    threshold : float
        Minimum absolute funding rate to trigger carry logic (default 0.0005 = 0.05%).
        Rates below this are noise-level and ignored.
    hold_window_minutes : int
        How many minutes before funding settlement to activate the hold logic.
        Default 30 min — wide enough to capture the payment even with REST polling lag.
    exit_window_minutes : int
        How many minutes before funding to trigger the early-exit logic.
        Default 15 min — narrow window so we don't exit too soon.
    exit_rate_multiplier : float
        The funding rate must exceed threshold × this multiplier to trigger early exit.
        Prevents exiting for trivial negative carry (default 3×).
    """

    def __init__(
        self,
        threshold: float = 0.0005,
        hold_window_minutes: int = 30,
        exit_window_minutes: int = 15,
        exit_rate_multiplier: float = 3.0,
    ) -> None:
        self.threshold = threshold
        self.hold_window = hold_window_minutes
        self.exit_window = exit_window_minutes
        self.exit_multiplier = exit_rate_multiplier

    # ── Public API ─────────────────────────────────────────────────────────────

    def should_hold_for_carry(
        self,
        direction: int,
        funding_rate: float,
        next_funding_ts_ms: int | None = None,
    ) -> Tuple[bool, str]:
        """
        Returns (hold, reason_string).

        Positive carry situations:
          LONG  + funding_rate < -threshold → shorts pay longs → collect
          SHORT + funding_rate > +threshold → longs pay shorts → collect
        """
        mins = self._minutes_remaining(next_funding_ts_ms)
        if not (0 < mins <= self.hold_window):
            return False, ""

        if direction > 0 and funding_rate < -self.threshold:
            return True, (
                f"hold_for_long_carry: fr={funding_rate:.5f} "
                f"({mins:.1f} min to funding)"
            )
        if direction < 0 and funding_rate > self.threshold:
            return True, (
                f"hold_for_short_carry: fr={funding_rate:.5f} "
                f"({mins:.1f} min to funding)"
            )
        return False, ""

    def should_exit_before_funding(
        self,
        direction: int,
        funding_rate: float,
        next_funding_ts_ms: int | None = None,
    ) -> Tuple[bool, str]:
        """
        Returns (exit_early, reason_string).

        Negative carry situations (avoid paying large funding):
          LONG  + funding_rate > threshold × multiplier → longs pay → exit
          SHORT + funding_rate < -threshold × multiplier → shorts pay → exit
        """
        mins = self._minutes_remaining(next_funding_ts_ms)
        if not (0 < mins <= self.exit_window):
            return False, ""

        pay_threshold = self.threshold * self.exit_multiplier

        if direction > 0 and funding_rate > pay_threshold:
            return True, (
                f"exit_avoid_long_funding_cost: fr={funding_rate:.5f} "
                f"({mins:.1f} min to funding)"
            )
        if direction < 0 and funding_rate < -pay_threshold:
            return True, (
                f"exit_avoid_short_funding_cost: fr={funding_rate:.5f} "
                f"({mins:.1f} min to funding)"
            )
        return False, ""

    def carry_pnl_bps(self, direction: int, funding_rate: float) -> float:
        """
        Estimated carry P&L for the next funding period in basis points.
        Positive = collect, negative = pay.
        """
        return -direction * funding_rate * 10_000

    # ── Internal ───────────────────────────────────────────────────────────────

    def _minutes_remaining(self, next_ts_ms: int | None) -> float:
        ts = next_ts_ms if next_ts_ms is not None else next_funding_timestamp_ms()
        return (ts - int(time.time() * 1000)) / 60_000
