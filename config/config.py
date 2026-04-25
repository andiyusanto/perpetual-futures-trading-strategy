"""
Centralised configuration via Pydantic BaseSettings.
All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_str_list(v: Any) -> Any:
    """Accept JSON array, comma-separated string, or empty string for List[str] fields."""
    if not isinstance(v, str):
        return v
    v = v.strip()
    if not v:
        return []
    if v.startswith("["):
        return json.loads(v)
    return [s.strip() for s in v.split(",") if s.strip()]


class NotificationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOTIFY_", env_file=".env", extra="ignore")

    telegram_token: str = Field("", description="Telegram Bot API token")
    telegram_chat_id: str = Field("", description="Telegram chat/channel ID")
    discord_webhook_url: str = Field("", description="Discord webhook URL")
    notify_on_trade: bool = Field(True, description="Alert on every open/close")
    notify_on_pnl: bool = Field(True, description="Send daily PnL report")
    daily_report_hour_utc: int = Field(0, description="UTC hour for daily report (0-23)")


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", extra="ignore")

    enabled: bool = Field(True, description="Persist trades to SQLite")
    db_path: str = Field("data/trades.db", description="SQLite file path")


class ExchangeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXCHANGE_", env_file=".env", extra="ignore")

    id: str = Field("binanceusdm", description="CCXT exchange ID")
    api_key: str = Field("", description="Exchange API key")
    api_secret: str = Field("", description="Exchange API secret")
    passphrase: Optional[str] = Field(None, description="Passphrase (OKX/Bybit)")
    testnet: bool = Field(True, description="Use exchange testnet")


class TradingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADING_", env_file=".env", extra="ignore")

    symbol: str = Field("BTC/USDT:USDT", description="Primary trading pair in CCXT format")
    symbols: List[str] = Field(
        default_factory=list,
        description="Multi-symbol list (comma-separated in env: TRADING_SYMBOLS). "
                    "If non-empty, used by apfts-multi-bot instead of TRADING_SYMBOL.",
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, v: Any) -> Any:
        return _parse_str_list(v)

    leverage: int = Field(5, description="Leverage multiplier (1-20 recommended)")
    timeframe: str = Field("1m", description="Candle timeframe")
    max_position_pct: float = Field(0.10, description="Max position as fraction of equity")
    use_websocket: bool = Field(
        False,
        description="Replace 30-second REST candle polling with true WebSocket stream. "
                    "Binance USDM only. Reduces entry latency from ~30 s to ~50 ms.",
    )
    live_liq_feed: bool = Field(
        False,
        description="Subscribe to !forceOrder@arr WebSocket for real liquidation events "
                    "instead of estimating clusters from OI changes. Binance USDM only.",
    )


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

    # Funding carry management
    carry_threshold: float = 0.0005       # min |FR| to activate carry logic (0.05%)
    carry_hold_minutes: int = 30          # hold window before funding settlement
    carry_exit_minutes: int = 15          # early-exit window before unfavourable funding


class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    max_drawdown_pct: float = Field(15.0)
    max_daily_loss_pct: float = Field(5.0)
    max_position_size_pct: float = Field(0.10)
    kill_switch_drawdown: float = Field(20.0)
    base_risk_pct: float = 0.01       # fraction of equity risked per trade
    max_open_trades: int = 1          # single-position mode


class ShadowConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHADOW_", env_file=".env", extra="ignore")

    enabled: bool = Field(False, description="Run in shadow mode (no real orders)")
    symbols: List[str] = Field(
        default_factory=list,
        description="Symbols to shadow (comma-separated). Falls back to TRADING_SYMBOL if empty.",
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, v: Any) -> Any:
        return _parse_str_list(v)

    log_level: str = Field("INFO", description="Log level for shadow-specific output")
    compare_real: bool = Field(False, description="Log divergence when shadow and real signals differ")
    slippage_bps: float = Field(3.0, description="Simulated slippage in basis points per fill")
    starting_equity: float = Field(10000.0, description="Simulated starting equity (USDT)")
    fill_timeout_bars: int = Field(2, description="Bars before unfilled limit order is market-filled")


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)

    log_level: str = Field("INFO")
    log_dir: str = "logs"
