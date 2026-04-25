"""
Migration: create `shadow_trades` table in an existing trades.db.

Usage:
    python scripts/migrate_shadow_trades.py [--db-path data/trades.db]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_CREATE_SHADOW_TABLE = """
CREATE TABLE IF NOT EXISTS shadow_trades (
    trade_id               TEXT PRIMARY KEY,
    symbol                 TEXT NOT NULL,
    direction              INTEGER NOT NULL,
    entry_price            REAL NOT NULL,
    stop_loss              REAL NOT NULL,
    take_profit            REAL NOT NULL,
    amount                 REAL NOT NULL,
    size_pct               REAL NOT NULL,
    grade                  TEXT NOT NULL,
    confidence             REAL NOT NULL,
    open_ts                INTEGER NOT NULL,
    close_ts               INTEGER,
    exit_price             REAL,
    pnl_pct                REAL,
    exit_reason            TEXT,
    capital_after          REAL,
    simulated_slippage_bps REAL,
    would_have_executed    INTEGER DEFAULT 1,
    shadow_timestamp       INTEGER NOT NULL
);
"""

_CREATE_SHADOW_IDX = """
CREATE INDEX IF NOT EXISTS idx_shadow_trades_open_ts ON shadow_trades (open_ts);
"""


def migrate(db_path: str) -> None:
    path = Path(db_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_CREATE_SHADOW_TABLE)
        conn.execute(_CREATE_SHADOW_IDX)
        conn.commit()
        print(f"Migration complete: shadow_trades table ready in {db_path}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create shadow_trades table")
    parser.add_argument("--db-path", default="data/trades.db")
    args = parser.parse_args()
    migrate(args.db_path)


if __name__ == "__main__":
    main()
