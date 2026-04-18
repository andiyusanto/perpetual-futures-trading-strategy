"""
Core technical indicator functions.
Pure numpy/pandas — no side effects, fully type-hinted.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    return pd.Series(data).ewm(span=period, adjust=False).mean().values  # type: ignore[return-value]


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI (EMA-smoothed)."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))  # type: ignore[return-value]


def compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range (Wilder / EMA)."""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values  # type: ignore[return-value]


def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX with +DI / -DI. Returns (adx, plus_di, minus_di)."""
    n = len(high)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    atr_arr = compute_atr(high, low, close, period)
    safe = np.where(atr_arr == 0, 1e-10, atr_arr)
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / safe
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / safe

    denom = np.where(np.abs(plus_di + minus_di) == 0, 1e-10, np.abs(plus_di + minus_di))
    dx = 100.0 * np.abs(plus_di - minus_di) / denom
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    return adx, plus_di, minus_di  # type: ignore[return-value]
