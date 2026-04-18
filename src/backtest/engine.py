"""
BacktestEngineV3 — bar-by-bar backtester using the shared V3StrategyEngine.

Run:
    apfts-backtest
    python -m src.backtest.engine
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from config.config import AppConfig, StrategyConfig
from src.backtest.metrics import BacktestMetrics, TradeRecord
from src.risk.position_sizer import PositionSizer, SignalType, TradeSignal
from src.risk.risk_manager import MarketRegime, RegimeClassifier, RiskManager
from src.strategy.engine import V3StrategyEngine
from src.strategy.exits import chandelier_update

logger = logging.getLogger(__name__)


# =============================================================================
# Execution cost model
# =============================================================================


class ExecutionSimulator:
    maker_fee_bps: float = 2.0    # Binance USDM maker rebate
    taker_fee_bps: float = 5.5    # Taker fee


# =============================================================================
# Synthetic data generator
# =============================================================================


def generate_v3_market_data(n_bars: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic OHLCV + perpetual data with multi-regime dynamics,
    GARCH volatility clustering, liquidation cascades, and realistic
    funding rate velocity dynamics.
    """
    rng = np.random.default_rng(seed)

    log_price = np.log(50_000.0)
    prev_ret = 0.0
    opens, highs, lows, closes, volumes = [], [], [], [], []
    funding_rates, open_interests = [], []

    regime_dur = 0
    current_regime = "trending"
    trend_dir = 1
    vol_state = 0.002
    funding_state = 0.0

    recent_highs: List[float] = []
    recent_lows: List[float] = []

    for _ in range(n_bars):
        regime_dur += 1
        if regime_dur > int(rng.integers(120, 500)):
            regime_dur = 0
            r = float(rng.random())
            if r < 0.30:
                current_regime = "trending"
                trend_dir = int(rng.choice([-1, 1]))
            elif r < 0.70:
                current_regime = "mean_reverting"
            elif r < 0.90:
                current_regime = "high_vol"
            else:
                current_regime = "low_liq"

        if current_regime == "trending":
            drift, ar, base_vol, vol_base = trend_dir * 0.0003, 0.25, 0.0020, 1200
        elif current_regime == "mean_reverting":
            drift = (np.log(50_000) - log_price) * 0.00012
            ar, base_vol, vol_base = -0.15, 0.0015, 800
        elif current_regime == "high_vol":
            drift, ar, base_vol, vol_base = 0.0, 0.12, 0.0045, 2500
        else:
            drift, ar, base_vol, vol_base = 0.0, -0.05, 0.001, 200

        vol_innovation = float(abs(rng.standard_normal())) * 0.0005
        vol_state = float(np.clip(
            0.85 * vol_state + 0.10 * base_vol + 0.05 * vol_innovation,
            0.0005, 0.015,
        ))

        # Liquidation cascade
        liq = 0.0
        price_now = np.exp(log_price)
        if len(recent_lows) > 3 and current_regime == "trending" and trend_dir < 0:
            candidates = [x for x in recent_lows if x < price_now]
            if candidates:
                d = (price_now - max(candidates)) / price_now
                if d < 0.01:
                    liq = -0.001 * (1 - d / 0.01)
        if len(recent_highs) > 3 and current_regime == "trending" and trend_dir > 0:
            candidates = [x for x in recent_highs if x > price_now]
            if candidates:
                d = (min(candidates) - price_now) / price_now
                if d < 0.01:
                    liq = 0.001 * (1 - d / 0.01)

        log_ret = drift + ar * prev_ret + float(rng.standard_normal()) * vol_state + liq
        prev_ret = log_ret
        open_p = np.exp(log_price)
        log_price += log_ret
        close_p = np.exp(log_price)

        wick_u = float(abs(rng.standard_normal())) * vol_state * 0.35 * open_p
        wick_d = float(abs(rng.standard_normal())) * vol_state * 0.35 * open_p
        high_p = max(open_p, close_p) + wick_u
        low_p  = min(open_p, close_p) - wick_d

        recent_highs.append(high_p)
        recent_lows.append(low_p)
        if len(recent_highs) > 200:
            recent_highs.pop(0)
            recent_lows.pop(0)

        ret_mag = abs(log_ret) / (vol_state + 1e-10)
        vol_mult = 1.0 + 0.8 * ret_mag
        if abs(liq) > 0.0005:
            vol_mult *= 2.5
        volume = float(max(10.0, vol_base * vol_mult * (1 + 0.2 * float(rng.standard_normal()))))

        # Funding rate with realistic velocity dynamics
        funding_state += -funding_state * 0.05
        if current_regime == "trending":
            funding_state += trend_dir * 0.00005 * min(regime_dur / 50, 1.0)
        funding_state += float(rng.normal(0, 0.0003))
        funding_state = float(np.clip(funding_state, -0.01, 0.01))

        oi = 5e8 + float(rng.standard_normal()) * 3e7
        if current_regime == "trending":
            oi += regime_dur * 8e5
        elif current_regime == "high_vol":
            oi -= regime_dur * 5e5

        opens.append(open_p)
        highs.append(high_p)
        lows.append(low_p)
        closes.append(close_p)
        volumes.append(volume)
        funding_rates.append(funding_state)
        open_interests.append(float(max(1e8, oi)))

        log_price = float(np.clip(log_price, np.log(10_000), np.log(200_000)))

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "funding_rate": funding_rates,
        "open_interest": open_interests,
    })


# =============================================================================
# BacktestEngineV3
# =============================================================================


class BacktestEngineV3:
    """
    Bar-by-bar backtester using the exact same V3StrategyEngine as production.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        cfg: Optional[AppConfig] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self._cfg = cfg or AppConfig()
        self._strategy = V3StrategyEngine(self._cfg.strategy)
        self._regime_clf = RegimeClassifier()
        self._sizer = PositionSizer(
            base_risk_pct=self._cfg.risk.base_risk_pct,
            max_size_pct=self._cfg.risk.max_position_size_pct,
        )
        self._exec_sim = ExecutionSimulator()

    def run(self, df: pd.DataFrame) -> BacktestMetrics:
        n = len(df)
        capital = self.initial_capital
        peak_capital = capital
        max_dd = 0.0
        trades: List[TradeRecord] = []
        equity_curve: List[float] = [capital]

        # Position state
        in_position = False
        position_dir = 0
        entry_price = stop_loss = initial_risk = take_profit = 0.0
        pos_size_pct = exec_cost_bps = 0.0
        entry_bar = 0
        entry_regime = entry_quality = ""
        highest_since = 0.0
        lowest_since = 1e18
        trail_phase = 0

        # Limit order state
        pending_limit = False
        lim_price = lim_sl = lim_risk = lim_tp = lim_size = 0.0
        lim_direction = lim_bar = 0
        lim_quality = lim_regime = ""

        lookback = 120
        close_arr = df["close"].values
        high_arr  = df["high"].values
        low_arr   = df["low"].values
        open_arr  = df["open"].values
        vol_arr   = df["volume"].values
        fr_arr    = df["funding_rate"].values
        oi_arr    = df["open_interest"].values

        baseline_vol = float(np.std(np.diff(np.log(close_arr[:min(720, n)] + 1e-10))))
        s_cfg = self._cfg.strategy

        for i in range(lookback, n):
            cur_c = float(close_arr[i])
            cur_h = float(high_arr[i])
            cur_l = float(low_arr[i])

            # Check pending limit order
            if pending_limit and not in_position:
                bars_waiting = i - lim_bar
                if bars_waiting > s_cfg.limit_wait_bars:
                    pending_limit = False
                elif lim_direction > 0 and cur_l <= lim_price:
                    in_position, position_dir = True, lim_direction
                    entry_price, entry_bar = lim_price, i
                    stop_loss, initial_risk = lim_sl, lim_risk
                    take_profit, pos_size_pct = lim_tp, lim_size
                    entry_regime, entry_quality = lim_regime, lim_quality
                    exec_cost_bps = self._exec_sim.maker_fee_bps + 1.0
                    highest_since, lowest_since = cur_h, cur_l
                    trail_phase = 0
                    pending_limit = False
                elif lim_direction < 0 and cur_h >= lim_price:
                    in_position, position_dir = True, lim_direction
                    entry_price, entry_bar = lim_price, i
                    stop_loss, initial_risk = lim_sl, lim_risk
                    take_profit, pos_size_pct = lim_tp, lim_size
                    entry_regime, entry_quality = lim_regime, lim_quality
                    exec_cost_bps = self._exec_sim.maker_fee_bps + 1.0
                    highest_since, lowest_since = cur_h, cur_l
                    trail_phase = 0
                    pending_limit = False

            # Manage open position
            if in_position:
                highest_since = max(highest_since, cur_h)
                lowest_since  = min(lowest_since, cur_l)

                cur_atr = float(np.mean(
                    high_arr[max(0, i-14):i+1] - low_arr[max(0, i-14):i+1]
                ))
                stop_loss, trail_phase = chandelier_update(
                    position_dir, entry_price, initial_risk,
                    cur_h, cur_l, cur_atr,
                    stop_loss, highest_since, lowest_since,
                )

                hit_sl = (position_dir > 0 and cur_l <= stop_loss) or \
                         (position_dir < 0 and cur_h >= stop_loss)
                hit_tp = (position_dir > 0 and cur_h >= take_profit) or \
                         (position_dir < 0 and cur_l <= take_profit)
                time_exit = (i - entry_bar) >= s_cfg.max_hold_bars

                if hit_sl or hit_tp or time_exit:
                    if hit_sl and trail_phase > 0:
                        exit_price, exit_reason = stop_loss, f"trail_phase_{trail_phase}"
                    elif hit_sl:
                        exit_price, exit_reason = stop_loss, "stop_loss"
                    elif hit_tp:
                        exit_price, exit_reason = take_profit, "take_profit"
                    else:
                        exit_price, exit_reason = cur_c, "time_exit"

                    if position_dir > 0:
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100

                    is_taker = exit_reason != "take_profit"
                    exit_fee = self._exec_sim.taker_fee_bps if is_taker else self._exec_sim.maker_fee_bps
                    total_cost = exec_cost_bps + exit_fee + 1.5
                    net_pnl = pnl_pct - total_cost / 100
                    risk_pct = initial_risk / entry_price * 100
                    pnl_r = net_pnl / risk_pct if risk_pct > 0 else 0.0

                    capital *= (1 + net_pnl / 100 * pos_size_pct)
                    peak_capital = max(peak_capital, capital)
                    dd = (peak_capital - capital) / peak_capital * 100
                    max_dd = max(max_dd, dd)

                    trades.append(TradeRecord(
                        entry_time=entry_bar, exit_time=i,
                        direction=position_dir,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        position_size_pct=pos_size_pct,
                        pnl_pct=net_pnl, pnl_r=pnl_r,
                        exit_reason=exit_reason, regime=entry_regime,
                        edge_quality=entry_quality,
                        execution_cost_bps=total_cost,
                    ))
                    in_position = False
                    position_dir = 0

            elif not pending_limit:
                if i % 3 != 0:
                    equity_curve.append(capital)
                    continue

                ws = max(0, i - 500)
                c_w  = close_arr[ws:i+1]
                h_w  = high_arr[ws:i+1]
                l_w  = low_arr[ws:i+1]
                o_w  = open_arr[ws:i+1]
                v_w  = vol_arr[ws:i+1]
                fr_w = fr_arr[ws:i+1]
                oi_w = oi_arr[ws:i+1]

                if len(c_w) < lookback:
                    equity_curve.append(capital)
                    continue

                window_df = df.iloc[ws:i+1]
                regime_state = self._regime_clf.classify(window_df)
                regime_type = regime_state.regime.value

                signal = self._strategy.generate_signal_from_arrays(
                    c_w, h_w, l_w, o_w, v_w, fr_w, oi_w,
                    regime_type=regime_type,
                    regime_confidence=regime_state.confidence,
                )

                if not signal.is_tradeable():
                    equity_curve.append(capital)
                    continue

                direction = 1 if signal.direction == "LONG" else -1
                cur_dd_pct = (peak_capital - capital) / peak_capital * 100
                cur_vol = float(np.std(np.diff(np.log(c_w[-24:] + 1e-10))))

                mock_signal = TradeSignal(
                    direction=SignalType.STRONG_LONG if direction > 0 else SignalType.STRONG_SHORT,
                    confidence=signal.confidence,
                    trend_score=signal.trend_score,
                    momentum_score=signal.momentum_score,
                    orderflow_score=signal.orderflow_score,
                    regime=regime_state.regime,
                    entry_price=cur_c,
                    stop_loss=signal.stop_loss,
                    take_profit_1=signal.take_profit,
                    take_profit_2=signal.take_profit,
                    position_size_pct=0.0,
                    timeframe_alignment=0.5,
                    edge_quality=signal.grade,
                    filters_passed=True,
                )
                pos_size = self._sizer.compute_size(
                    mock_signal, capital, cur_dd_pct, 0.55, 1.8, baseline_vol, cur_vol
                )

                if pos_size < 0.001:
                    equity_curve.append(capital)
                    continue

                pending_limit = True
                lim_price     = signal.limit_entry_price or cur_c
                lim_direction = direction
                lim_bar       = i
                lim_sl        = signal.stop_loss
                lim_risk      = signal.initial_risk
                lim_tp        = signal.take_profit
                lim_quality   = signal.grade
                lim_regime    = regime_type
                lim_size      = pos_size

            equity_curve.append(capital)

        return self._compute_metrics(trades, equity_curve, max_dd)

    def _compute_metrics(
        self,
        trades: List[TradeRecord],
        equity_curve: List[float],
        max_dd: float,
    ) -> BacktestMetrics:
        if not trades:
            return BacktestMetrics(
                total_trades=0, win_rate=0.0, avg_rr=0.0, expectancy=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown_pct=max_dd,
                profit_factor=0.0, avg_trade_pnl_pct=0.0, total_return_pct=0.0,
                calmar_ratio=0.0, avg_execution_cost_bps=0.0,
            )

        pnls   = [t.pnl_pct for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate   = len(wins) / len(pnls)
        avg_win    = float(np.mean(wins))  if wins   else 0.0
        avg_loss   = float(abs(np.mean(losses))) if losses else 0.0
        avg_rr     = avg_win / avg_loss if avg_loss > 0 else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        returns  = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-10)
        returns  = returns[returns != 0]
        ann      = np.sqrt(8_760)
        sharpe   = float(np.mean(returns) / (np.std(returns) + 1e-10) * ann) if len(returns) > 0 else 0.0
        downside = returns[returns < 0]
        sortino  = float(np.mean(returns) / (np.std(downside) + 1e-10) * ann) if len(downside) > 0 else 0.0

        gross_profit  = sum(wins)   if wins   else 0.0
        gross_loss    = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        calmar       = total_return / max_dd if max_dd > 0 else 0.0
        avg_exec     = float(np.mean([t.execution_cost_bps for t in trades]))

        return BacktestMetrics(
            total_trades=len(trades), win_rate=win_rate, avg_rr=avg_rr,
            expectancy=expectancy, sharpe_ratio=sharpe, sortino_ratio=sortino,
            max_drawdown_pct=max_dd, profit_factor=profit_factor,
            avg_trade_pnl_pct=float(np.mean(pnls)), total_return_pct=total_return,
            calmar_ratio=calmar, avg_execution_cost_bps=avg_exec,
            trades=trades,
        )


# =============================================================================
# CLI runner
# =============================================================================


def _setup_logging(level: str = "INFO") -> None:
    import os
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/backtest.log", mode="a"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="APFTS v3 Backtest")
    parser.add_argument("--bars",     type=int,   default=10_000)
    parser.add_argument("--seeds",    type=str,   default="42,123,456,789,1337")
    parser.add_argument("--capital",  type=float, default=100_000.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    cfg   = AppConfig()

    print("=" * 72)
    print("  APFTS v3 — DEEP ALPHA BACKTEST")
    print("=" * 72)

    results = []
    for seed in seeds:
        df = generate_v3_market_data(n_bars=args.bars, seed=seed)
        engine = BacktestEngineV3(initial_capital=args.capital, cfg=cfg)
        r = engine.run(df)

        a_trades = [t for t in r.trades if t.edge_quality in ("A+", "A")]
        a_wr  = sum(1 for t in a_trades if t.pnl_pct > 0) / len(a_trades) * 100 if a_trades else 0.0
        a_exp = float(np.mean([t.pnl_pct for t in a_trades])) if a_trades else 0.0

        results.append({"seed": seed, "r": r, "a_wr": a_wr, "a_exp": a_exp})
        print(f"\n  Seed {seed}: {r.summary()}")
        print(f"    Grade A: {len(a_trades)} trades, WR={a_wr:.1f}%, Exp={a_exp:+.4f}%")

        if r.trades:
            exits: dict[str, int] = {}
            for t in r.trades:
                exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
            print("    Exits: " + " | ".join(f"{k}:{v}" for k, v in sorted(exits.items())))

    print("\n" + "=" * 72)
    print("  CROSS-SEED SUMMARY")
    print("=" * 72)

    def avg(key: str) -> float:
        return float(np.mean([getattr(x["r"], key) for x in results]))

    print(f"\n  Win Rate      : {avg('win_rate')*100:.1f}%")
    print(f"  R:R           : {avg('avg_rr'):.2f}x")
    print(f"  Expectancy    : {avg('expectancy'):+.4f}%")
    print(f"  Profit Factor : {avg('profit_factor'):.2f}")
    print(f"  Max Drawdown  : {avg('max_drawdown_pct'):.2f}%")
    print(f"  Total Return  : {avg('total_return_pct'):+.2f}%")
    print(f"  Exec Cost     : {avg('avg_execution_cost_bps'):.1f} bp")
    print(f"  Grade A WR    : {float(np.mean([x['a_wr'] for x in results])):.1f}%")
    print(f"  Grade A Exp   : {float(np.mean([x['a_exp'] for x in results])):+.4f}%")
    print(f"  Positive Seeds: {sum(1 for x in results if x['r'].expectancy > 0)}/{len(seeds)}")

    ror = RiskManager.risk_of_ruin(
        max(avg("win_rate"), 0.01),
        max(avg("avg_rr"), 0.01),
        cfg.risk.base_risk_pct,
        0.5,
    )
    print(f"  Risk of Ruin  : {ror*100:.2f}%")


if __name__ == "__main__":
    main()
