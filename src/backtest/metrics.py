"""
Immutable dataclasses for backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TradeRecord:
    """Single completed trade."""
    entry_time: int
    exit_time: int
    direction: int                  # +1 long / -1 short
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    pnl_pct: float                  # Net PnL after execution costs
    pnl_r: float                    # PnL in units of initial risk
    exit_reason: str                # stop_loss / take_profit / trail_phase_N / time_exit
    regime: str
    edge_quality: str               # A+, A, B, C
    execution_cost_bps: float


@dataclass
class BacktestMetrics:
    """Aggregate performance metrics for one backtest run."""
    total_trades: int
    win_rate: float
    avg_rr: float
    expectancy: float               # Average net PnL % per trade
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    avg_trade_pnl_pct: float
    total_return_pct: float
    calmar_ratio: float
    avg_execution_cost_bps: float
    trades: List[TradeRecord] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Trades={self.total_trades} | WR={self.win_rate*100:.1f}% | "
            f"R:R={self.avg_rr:.2f}x | Exp={self.expectancy:+.4f}% | "
            f"PF={self.profit_factor:.2f} | DD={self.max_drawdown_pct:.2f}% | "
            f"Ret={self.total_return_pct:+.2f}% | Exec={self.avg_execution_cost_bps:.1f}bp"
        )
