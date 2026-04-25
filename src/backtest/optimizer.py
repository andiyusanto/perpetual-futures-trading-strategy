"""
Walk-forward optimisation for StrategyConfig parameters.

Algorithm
---------
1. Generate (or load) historical / synthetic data.
2. Slice into rolling in-sample / out-of-sample windows.
3. Grid-search StrategyConfig parameters on each in-sample window.
4. Evaluate the winning params on the matching out-of-sample window.
5. Aggregate OOS results — this is the realistic expectancy.

Usage
-----
    apfts-optimize
    apfts-optimize --bars 12000 --in-sample 4000 --out-sample 1500 --step 750
    python -m src.backtest.optimizer
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.config import AppConfig, StrategyConfig
from src.backtest.engine import BacktestEngineV3, generate_v3_market_data
from src.backtest.metrics import BacktestMetrics

log = logging.getLogger(__name__)


# =============================================================================
# Default parameter grid
# =============================================================================

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "composite_threshold": [0.20, 0.25, 0.30],
    "strong_threshold":    [0.50, 0.55, 0.60],
    "tp_mult_trending":    [4.0, 5.0, 6.0],
    "tp_mult_normal":      [2.5, 3.0, 3.5],
    "max_hold_bars":       [72, 96, 120],
}


def _zero_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_trades=0, win_rate=0.0, avg_rr=0.0, expectancy=0.0,
        sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown_pct=100.0,
        profit_factor=0.0, avg_trade_pnl_pct=0.0, total_return_pct=0.0,
        calmar_ratio=0.0, avg_execution_cost_bps=0.0,
    )


# =============================================================================
# Result containers
# =============================================================================


@dataclass
class WindowResult:
    window_idx: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    best_params: Dict[str, Any]
    is_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics


@dataclass
class OptimizationReport:
    windows: List[WindowResult] = field(default_factory=list)
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)

    # ── Aggregate statistics ──────────────────────────────────────────────────

    @property
    def oos_sharpe_mean(self) -> float:
        vals = [w.oos_metrics.sharpe_ratio for w in self.windows]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def oos_sharpe_std(self) -> float:
        vals = [w.oos_metrics.sharpe_ratio for w in self.windows]
        return float(np.std(vals)) if vals else 0.0

    @property
    def oos_win_rate_mean(self) -> float:
        vals = [w.oos_metrics.win_rate for w in self.windows]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def oos_return_mean(self) -> float:
        vals = [w.oos_metrics.total_return_pct for w in self.windows]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def oos_max_dd_mean(self) -> float:
        vals = [w.oos_metrics.max_drawdown_pct for w in self.windows]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def recommended_params(self) -> Dict[str, Any]:
        """Params from the OOS window with the highest Sharpe ratio."""
        if not self.windows:
            return {}
        best = max(self.windows, key=lambda w: w.oos_metrics.sharpe_ratio)
        return best.best_params

    # ── Formatting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  WALK-FORWARD OPTIMISATION REPORT",
            "=" * 60,
            f"  Windows completed : {len(self.windows)}",
            f"  OOS Sharpe        : {self.oos_sharpe_mean:+.3f} ± {self.oos_sharpe_std:.3f}",
            f"  OOS Win Rate      : {self.oos_win_rate_mean:.1%}",
            f"  OOS Return (mean) : {self.oos_return_mean:+.2f}%",
            f"  OOS Max Drawdown  : {self.oos_max_dd_mean:.2f}%",
            "",
            "  Recommended params (highest OOS Sharpe window):",
        ]
        for k, v in self.recommended_params.items():
            lines.append(f"    {k}: {v}")

        lines += ["", "  Per-window breakdown:", "  " + "-" * 55]
        header = f"  {'W':>3}  {'IS bars':>8}  {'OOS bars':>8}  {'IS Sharpe':>10}  {'OOS Sharpe':>10}  {'OOS Trades':>10}"
        lines.append(header)
        for w in self.windows:
            is_bars = w.is_end - w.is_start
            oos_bars = w.oos_end - w.oos_start
            lines.append(
                f"  {w.window_idx:>3}  {is_bars:>8}  {oos_bars:>8}"
                f"  {w.is_metrics.sharpe_ratio:>+10.3f}"
                f"  {w.oos_metrics.sharpe_ratio:>+10.3f}"
                f"  {w.oos_metrics.total_trades:>10}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Walk-forward engine
# =============================================================================


class WalkForwardOptimizer:
    """
    Rolling walk-forward optimizer.

    Parameters
    ----------
    base_cfg        : AppConfig  — baseline config; strategy params are overridden
    in_sample_bars  : bars in each in-sample training window
    out_sample_bars : bars in each out-of-sample test window
    step_bars       : bars to advance the window on each iteration
    param_grid      : dict of param_name → list of values to try
    objective       : "sharpe" | "expectancy" | "calmar" — IS fitness metric
    """

    def __init__(
        self,
        base_cfg: AppConfig,
        in_sample_bars: int = 3_000,
        out_sample_bars: int = 1_000,
        step_bars: int = 500,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        objective: str = "sharpe",
    ) -> None:
        self._base_cfg = base_cfg
        self._is_bars = in_sample_bars
        self._oos_bars = out_sample_bars
        self._step = step_bars
        self._grid = param_grid or DEFAULT_PARAM_GRID
        self._objective = objective

        # Pre-compute all combos once
        keys = list(self._grid.keys())
        combos = list(itertools.product(*self._grid.values()))
        self._combos: List[Dict[str, Any]] = [dict(zip(keys, c)) for c in combos]
        log.info("wfo_init",
                 combos=len(self._combos),
                 is_bars=in_sample_bars,
                 oos_bars=out_sample_bars,
                 objective=objective)

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self, data: Optional[pd.DataFrame] = None) -> OptimizationReport:
        """
        Run the full walk-forward sequence.

        Parameters
        ----------
        data : pd.DataFrame (optional)
            Pre-loaded OHLCV + perp data.  If None, generates synthetic data
            with enough bars for all windows.
        """
        if data is None:
            min_bars = self._is_bars + self._oos_bars + self._step * 6
            log.info("generating_synthetic_data", bars=min_bars)
            data = generate_v3_market_data(n_bars=min_bars)

        n = len(data)
        report = OptimizationReport(param_grid=self._grid)
        win_idx = 0
        start = 0

        while start + self._is_bars + self._oos_bars <= n:
            is_end  = start + self._is_bars
            oos_end = is_end + self._oos_bars

            is_data  = data.iloc[start:is_end].reset_index(drop=True)
            oos_data = data.iloc[is_end:oos_end].reset_index(drop=True)

            log.info("wfo_window_start",
                     window=win_idx, is_bars=len(is_data), oos_bars=len(oos_data))

            best_params, is_metrics = self._grid_search(is_data)
            oos_metrics = self._evaluate(oos_data, best_params)

            report.windows.append(WindowResult(
                window_idx=win_idx,
                is_start=start,
                is_end=is_end,
                oos_start=is_end,
                oos_end=oos_end,
                best_params=best_params,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
            ))

            log.info("wfo_window_done",
                     window=win_idx,
                     is_sharpe=f"{is_metrics.sharpe_ratio:+.3f}",
                     oos_sharpe=f"{oos_metrics.sharpe_ratio:+.3f}",
                     oos_trades=oos_metrics.total_trades)

            start += self._step
            win_idx += 1

        return report

    # ── Internal ───────────────────────────────────────────────────────────────

    def _grid_search(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], BacktestMetrics]:
        best_score = float("-inf")
        best_params: Dict[str, Any] = self._combos[0]
        best_metrics: BacktestMetrics = _zero_metrics()

        for params in self._combos:
            metrics = self._evaluate(data, params)
            score = self._score(metrics)
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

        return best_params, best_metrics

    def _evaluate(self, data: pd.DataFrame, params: Dict[str, Any]) -> BacktestMetrics:
        # Merge base strategy config with the override params
        base_dict = self._base_cfg.strategy.model_dump()
        base_dict.update(params)
        try:
            strategy_cfg = StrategyConfig(**base_dict)
        except Exception:
            return _zero_metrics()

        cfg = self._base_cfg.model_copy(update={"strategy": strategy_cfg})
        engine = BacktestEngineV3(cfg)
        try:
            return engine.run(data)
        except Exception as exc:
            log.debug("wfo_eval_error", params=params, error=str(exc))
            return _zero_metrics()

    def _score(self, m: BacktestMetrics) -> float:
        if m.total_trades < 5:           # too few trades — penalise
            return float("-inf")
        if self._objective == "expectancy":
            return m.expectancy
        if self._objective == "calmar":
            return m.calmar_ratio
        return m.sharpe_ratio            # default: Sharpe


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="APFTS walk-forward optimisation")
    parser.add_argument("--bars",        type=int,   default=10_000,
                        help="Total synthetic bars to generate")
    parser.add_argument("--in-sample",   type=int,   default=3_000,
                        help="Bars per in-sample window")
    parser.add_argument("--out-sample",  type=int,   default=1_000,
                        help="Bars per out-of-sample window")
    parser.add_argument("--step",        type=int,   default=500,
                        help="Window advance step in bars")
    parser.add_argument("--objective",   type=str,   default="sharpe",
                        choices=["sharpe", "expectancy", "calmar"],
                        help="In-sample optimisation objective")
    args = parser.parse_args()

    cfg = AppConfig()
    data = generate_v3_market_data(n_bars=args.bars)

    optimizer = WalkForwardOptimizer(
        base_cfg=cfg,
        in_sample_bars=args.in_sample,
        out_sample_bars=args.out_sample,
        step_bars=args.step,
        objective=args.objective,
    )
    report = optimizer.run(data)
    print(report.summary())


if __name__ == "__main__":
    main()
