"""
Unit + smoke tests for the shared strategy engine and core components.
Run: pytest tests/
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.utils import ema, compute_rsi, compute_atr
from src.core.volatility import VolatilityClassifier, VolatilityPhase
from src.strategy.exits import chandelier_update, ChandelierExit
from src.strategy.signals import FundingVelocitySignal, LiquidationMapper
from src.strategy.engine import V3StrategyEngine
from src.backtest.engine import generate_v3_market_data, BacktestEngineV3
from src.backtest.metrics import BacktestMetrics
from src.risk.risk_manager import RegimeClassifier, RiskManager
from config.config import AppConfig


# =============================================================================
# Core indicators
# =============================================================================


def test_ema_convergence():
    data = np.ones(100) * 50.0
    result = ema(data, 20)
    assert abs(result[-1] - 50.0) < 0.01


def test_rsi_bounds():
    rng = np.random.default_rng(0)
    prices = np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100
    rsi = compute_rsi(prices)
    assert np.all(rsi >= 0) and np.all(rsi <= 100)


def test_atr_positive():
    rng = np.random.default_rng(0)
    c = np.cumprod(1 + rng.normal(0, 0.01, 100)) * 100
    h = c * 1.005
    l = c * 0.995
    atr = compute_atr(h, l, c)
    assert np.all(atr[14:] > 0)


# =============================================================================
# Volatility classifier
# =============================================================================


def test_volatility_classifier_returns_phase():
    rng = np.random.default_rng(42)
    c = np.cumprod(1 + rng.normal(0, 0.005, 200)) * 50_000
    h = c * 1.002
    l = c * 0.998
    clf = VolatilityClassifier()
    phase, pctile, _ = clf.classify(c, h, l)
    assert isinstance(phase, VolatilityPhase)
    assert 0 <= pctile <= 100


def test_volatility_insufficient_data():
    c = np.ones(50) * 100.0
    h = c * 1.001
    l = c * 0.999
    phase, _, fired = VolatilityClassifier().classify(c, h, l)
    assert phase == VolatilityPhase.NORMAL
    assert not fired


# =============================================================================
# Chandelier exit
# =============================================================================


def test_chandelier_phase_0_holds_stop():
    stop, phase = chandelier_update(
        1, 100.0, 2.0, 101.0, 100.0, 1.0, 98.0, 101.0, 100.0
    )
    assert phase == 0
    assert stop == 98.0


def test_chandelier_moves_to_breakeven():
    # highest = entry + 2.25R -> triggers phase 1
    stop, phase = chandelier_update(
        1, 100.0, 2.0, 104.5, 103.0, 1.0, 98.0, 104.5, 100.0
    )
    assert phase >= 1
    assert stop >= 100.0


def test_chandelier_stop_never_worsens_long():
    stop1, _ = chandelier_update(1, 100.0, 2.0, 106.0, 104.0, 1.0, 98.0, 106.0, 100.0)
    stop2, _ = chandelier_update(1, 100.0, 2.0, 106.0, 103.0, 1.0, stop1, 106.0, 100.0)
    assert stop2 >= stop1


# =============================================================================
# Funding velocity signal
# =============================================================================


def test_funding_signal_neutral_on_short_history():
    sig, vel, acc = FundingVelocitySignal().compute(np.array([0.001, 0.001, 0.001]))
    assert sig == 0.0


def test_funding_signal_bearish_on_accelerating_positive():
    fr = np.linspace(0.0, 0.005, 30)
    sig, _, _ = FundingVelocitySignal().compute(fr)
    assert sig <= 0.0


# =============================================================================
# Strategy engine
# =============================================================================


def test_engine_neutral_on_short_history():
    engine = V3StrategyEngine()
    c = np.ones(50) * 100.0
    h, l, o, v = c * 1.001, c * 0.999, c, np.ones(50) * 1000.0
    fr, oi = np.zeros(50), np.ones(50) * 1e8
    sig = engine.generate_signal_from_arrays(c, h, l, o, v, fr, oi)
    assert sig.direction == "NEUTRAL"


def test_engine_processes_full_dataset():
    rng = np.random.default_rng(7)
    n = 300
    c = np.cumprod(1 + rng.normal(0.0002, 0.008, n)) * 50_000
    h = c * (1 + rng.uniform(0, 0.003, n))
    l = c * (1 - rng.uniform(0, 0.003, n))
    o = np.roll(c, 1); o[0] = c[0]
    v = np.abs(rng.normal(1000, 200, n))
    fr, oi = rng.normal(0, 0.0005, n), np.ones(n) * 1e8

    engine = V3StrategyEngine()
    sig = engine.generate_signal_from_arrays(c, h, l, o, v, fr, oi, "trending_bull", 0.8)
    assert sig.direction in ("LONG", "SHORT", "NEUTRAL")
    if sig.direction != "NEUTRAL":
        assert sig.stop_loss > 0
        assert sig.take_profit > 0
        assert sig.initial_risk > 0


# =============================================================================
# Risk manager
# =============================================================================


def test_risk_of_ruin_negative_expectancy():
    ror = RiskManager.risk_of_ruin(0.40, 1.0, 0.01, 0.5)
    assert ror == 1.0


def test_risk_of_ruin_positive_expectancy():
    ror = RiskManager.risk_of_ruin(0.55, 1.8, 0.01, 0.5)
    assert 0.0 <= ror < 1.0


# =============================================================================
# Backtest smoke test
# =============================================================================


def test_backtest_runs_and_returns_metrics():
    df = generate_v3_market_data(n_bars=2_000, seed=42)
    cfg = AppConfig()
    result = BacktestEngineV3(initial_capital=10_000.0, cfg=cfg).run(df)

    assert isinstance(result, BacktestMetrics)
    assert result.total_trades >= 0
    assert 0.0 <= result.win_rate <= 1.0
    assert result.profit_factor >= 0.0


def test_backtest_multiple_seeds_stable():
    cfg = AppConfig()
    for seed in (1, 2, 3):
        df = generate_v3_market_data(n_bars=1_500, seed=seed)
        r = BacktestEngineV3(initial_capital=10_000.0, cfg=cfg).run(df)
        assert r.total_trades >= 0
