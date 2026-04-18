"""
Centralised configuration via Pydantic BaseSettings.
All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXCHANGE_", env_file=".env", extra="ignore")

    id: str = Field("binanceusdm", description="CCXT exchange ID")
    api_key: str = Field("", description="Exchange API key")
    api_secret: str = Field("", description="Exchange API secret")
    passphrase: Optional[str] = Field(None, description="Passphrase (OKX/Bybit)")
    testnet: bool = Field(True, description="Use exchange testnet")


class TradingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADING_", env_file=".env", extra="ignore")

    symbol: str = Field("BTC/USDT:USDT", description="Trading pair in CCXT format")
    leverage: int = Field(5, description="Leverage multiplier (1-20 recommended)")
    timeframe: str = Field("1m", description="Candle timeframe")
    max_position_pct: float = Field(0.10, description="Max position as fraction of equity")


class StrategyConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # EMA stack
    ema_fast: int = 8
    ema_mid: int = 21
    ema_slow: int = 55

    # Signal gate
    composite_threshold: float = 0.25
    strong_threshold: float = 0.55
    min_signal_agreement: int = 2       # out of 4 signals must agree

    # Volatility classifier (Bollinger / Keltner)
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_mult: float = 1.5
    vol_lookback: int = 120

    # Funding velocity
    funding_vel_period: int = 8
    funding_accel_period: int = 4

    # Limit order entry
    limit_wait_bars: int = 3

    # Chandelier / take-profit
    max_hold_bars: int = 96
    tp_mult_trending: float = 5.0
    tp_mult_normal: float = 3.0

    # Liquidity filter
    volume_ma_bars: int = 720
    volume_min_ratio: float = 0.30


class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    max_drawdown_pct: float = Field(15.0)
    max_daily_loss_pct: float = Field(5.0)
    max_position_size_pct: float = Field(0.10)
    kill_switch_drawdown: float = Field(20.0)
    base_risk_pct: float = 0.01       # fraction of equity risked per trade
    max_open_trades: int = 1          # single-position mode


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    log_level: str = Field("INFO")
    log_dir: str = "logs"
