"""
Shadow report CLI — compare shadow vs. real trade performance.

Usage:
    apfts-shadow-report --days 7
    apfts-shadow-report --days 30 --db-path data/trades.db
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


def _query(db_path: str, table: str, since_ms: int) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            f"""SELECT
                    COUNT(*)                                          AS total,
                    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)    AS wins,
                    SUM(pnl_pct)                                      AS total_pnl,
                    AVG(pnl_pct)                                      AS avg_pnl,
                    MIN(pnl_pct)                                      AS worst,
                    MAX(pnl_pct)                                      AS best
                FROM {table}
                WHERE open_ts >= ? AND close_ts IS NOT NULL""",
            (since_ms,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None or row["total"] == 0:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_pnl": 0.0, "worst": 0.0, "best": 0.0}

    total = row["total"] or 0
    wins  = row["wins"] or 0
    return {
        "trades":    total,
        "wins":      wins,
        "win_rate":  wins / total if total > 0 else 0.0,
        "total_pnl": float(row["total_pnl"] or 0.0),
        "avg_pnl":   float(row["avg_pnl"] or 0.0),
        "worst":     float(row["worst"] or 0.0),
        "best":      float(row["best"] or 0.0),
    }


def _fmt(label: str, shadow: dict, real: dict) -> None:
    def row(name: str, key: str, fmt: str = ".3f") -> None:
        sv = shadow.get(key, 0.0)
        rv = real.get(key, 0.0)
        delta = sv - rv if isinstance(sv, float) else 0
        s_str = f"{sv:{fmt}}" if isinstance(sv, float) else str(sv)
        r_str = f"{rv:{fmt}}" if isinstance(rv, float) else str(rv)
        d_str = f"{delta:+.3f}" if isinstance(sv, float) else ""
        print(f"  {name:<22} shadow={s_str:<12} real={r_str:<12} delta={d_str}")

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    row("Trades",    "trades",    "d")
    row("Win rate",  "win_rate",  ".1%")
    row("Total PnL%","total_pnl", "+.3f")
    row("Avg PnL%",  "avg_pnl",   "+.4f")
    row("Best trade","best",      "+.3f")
    row("Worst trade","worst",    "+.3f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow vs. real PnL comparison")
    parser.add_argument("--days",    type=int,   default=7)
    parser.add_argument("--db-path", type=str,   default="data/trades.db")
    args = parser.parse_args()

    db_path = args.db_path
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return

    since_ms = int((time.time() - args.days * 86400) * 1000)
    label    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    shadow = _query(db_path, "shadow_trades", since_ms)
    real   = _query(db_path, "trades",        since_ms)

    print(f"\nAPFTS v3 — Shadow Report  (last {args.days} days as of {label})")
    _fmt(f"Period: last {args.days} days", shadow, real)
    print()


if __name__ == "__main__":
    main()
