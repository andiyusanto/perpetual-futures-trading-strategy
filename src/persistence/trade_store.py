"""
TradeStore — async SQLite persistence layer for all trade records.

Schema (single table, append-only for opens, update on close):

    trades(
        trade_id TEXT PK,
        symbol TEXT, direction INTEGER,
        entry_price REAL, stop_loss REAL, take_profit REAL,
        amount REAL, size_pct REAL,
        grade TEXT, confidence REAL,
        open_ts INTEGER,          -- ms epoch
        close_ts INTEGER,         -- ms epoch, NULL while open
        exit_price REAL,
        pnl_pct REAL,
        exit_reason TEXT,
        capital_after REAL
    )

For TimescaleDB/PostgreSQL: set DATABASE_URL to a postgres:// connection string
and install asyncpg.  The SQL used here is standard and compatible with both.

SQLite default requires `aiosqlite` (listed in project deps).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiosqlite

from config.config import DatabaseConfig

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id      TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    direction     INTEGER NOT NULL,       -- +1 LONG / -1 SHORT
    entry_price   REAL NOT NULL,
    stop_loss     REAL NOT NULL,
    take_profit   REAL NOT NULL,
    amount        REAL NOT NULL,
    size_pct      REAL NOT NULL,
    grade         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    open_ts       INTEGER NOT NULL,
    close_ts      INTEGER,
    exit_price    REAL,
    pnl_pct       REAL,
    exit_reason   TEXT,
    capital_after REAL
);
"""

_CREATE_IDX = """
CREATE INDEX IF NOT EXISTS idx_trades_open_ts ON trades (open_ts);
"""


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    direction: int
    entry_price: float
    stop_loss: float
    take_profit: float
    amount: float
    size_pct: float
    grade: str
    confidence: float
    open_ts: int = field(default_factory=lambda: int(time.time() * 1000))
    close_ts: Optional[int] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    capital_after: Optional[float] = None


class TradeStore:
    """
    Async SQLite trade journal.  Falls back gracefully if disabled.
    Usage:
        store = TradeStore(cfg)
        await store.connect()
        await store.record_open(trade)
        await store.record_close(trade_id, ...)
        await store.close()
    """

    def __init__(self, cfg: DatabaseConfig) -> None:
        self._cfg = cfg
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        if not self._cfg.enabled:
            return
        import os
        db_path = self._cfg.db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db = await aiosqlite.connect(db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_IDX)
        await self._db.commit()
        logger.info("TradeStore connected: %s", db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Write operations ──────────────────────────────────────────────────────

    async def record_open(self, rec: TradeRecord) -> None:
        if not self._db:
            return
        try:
            await self._db.execute(
                """INSERT INTO trades
                   (trade_id, symbol, direction, entry_price, stop_loss, take_profit,
                    amount, size_pct, grade, confidence, open_ts)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.trade_id, rec.symbol, rec.direction,
                    rec.entry_price, rec.stop_loss, rec.take_profit,
                    rec.amount, rec.size_pct, rec.grade, rec.confidence,
                    rec.open_ts,
                ),
            )
            await self._db.commit()
        except Exception as exc:
            logger.error("trade_store_record_open_error: %s", exc)

    async def record_close(
        self,
        trade_id: str,
        exit_price: float,
        pnl_pct: float,
        exit_reason: str,
        capital_after: float,
        close_ts: Optional[int] = None,
    ) -> None:
        if not self._db:
            return
        ts = close_ts or int(time.time() * 1000)
        try:
            await self._db.execute(
                """UPDATE trades SET
                   close_ts=?, exit_price=?, pnl_pct=?, exit_reason=?, capital_after=?
                   WHERE trade_id=?""",
                (ts, exit_price, pnl_pct, exit_reason, capital_after, trade_id),
            )
            await self._db.commit()
        except Exception as exc:
            logger.error("trade_store_record_close_error: %s", exc)

    # ── Read operations ───────────────────────────────────────────────────────

    async def fetch_trades(self, limit: int = 100) -> List[dict]:
        if not self._db:
            return []
        async with self._db.execute(
            "SELECT * FROM trades ORDER BY open_ts DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def daily_summary(self, day_start_ms: Optional[int] = None) -> dict:
        """Return aggregated stats for closed trades since day_start_ms (defaults to today UTC)."""
        if not self._db:
            return {}
        if day_start_ms is None:
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            day_start_ms = int(today.timestamp() * 1000)

        async with self._db.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                      SUM(pnl_pct) as total_pnl,
                      MIN(pnl_pct) as worst,
                      MAX(pnl_pct) as best
               FROM trades
               WHERE open_ts >= ? AND close_ts IS NOT NULL""",
            (day_start_ms,),
        ) as cur:
            row = await cur.fetchone()

        if row is None or row["total"] == 0:
            return {"trades": 0, "wins": 0, "win_rate": 0.0, "total_pnl": 0.0,
                    "worst": 0.0, "best": 0.0}

        total = row["total"] or 0
        wins  = row["wins"] or 0
        return {
            "trades":    total,
            "wins":      wins,
            "win_rate":  wins / total if total > 0 else 0.0,
            "total_pnl": float(row["total_pnl"] or 0.0),
            "worst":     float(row["worst"] or 0.0),
            "best":      float(row["best"] or 0.0),
        }

    async def open_trade_count(self) -> int:
        if not self._db:
            return 0
        async with self._db.execute(
            "SELECT COUNT(*) FROM trades WHERE close_ts IS NULL"
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0
