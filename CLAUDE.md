# CLAUDE.md - Perpetual Futures Trading Strategy

## Project Overview
This is a **perpetual futures trading bot** for cryptocurrency exchanges (Binance, Bybit, etc.). The bot implements automated trading strategies including grid trading, DCA (Dollar Cost Averaging), and trend-following strategies with risk management.

## Tech Stack & Versions
- **Language**: Python 3.10+
- **Key Libraries**:
  - `ccxt` (^4.0.0) - Exchange integration
  - `pandas` (^2.0.0) - Data analysis
  - `numpy` (^1.24.0) - Calculations
  - `ta` (^0.10.0) - Technical indicators
  - `websocket-client` (^1.5.0) - Real-time data
  - `python-dotenv` (^1.0.0) - Configuration
- **Database**: SQLite (local) / PostgreSQL (production)
- **Testing**: `pytest` + `pytest-asyncio`
- **Logging**: `loguru` for structured logging

## Project Structure
```
perpetual-futures-trading-strategy/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pyproject.toml          # pip install -e .
├── requirements.txt
├── .env.example
├── config/
│   └── config.py           # Pydantic BaseSettings (all config classes)
├── src/
│   ├── core/
│   │   ├── data_buffer.py  # RingBuffer + MarketDataBuffer
│   │   ├── volatility.py   # VolatilityClassifier (shared)
│   │   └── utils.py        # ema, rsi, atr, adx
│   ├── strategy/
│   │   ├── signals.py      # SignalEngineV3, FundingVelocitySignal,
│   │   │                   # OrderBookImbalanceSignal, LiquidationMapper
│   │   ├── exits.py        # ChandelierExit, MarketStructureSL
│   │   └── engine.py       # V3StrategyEngine — shared by backtest + production
│   ├── execution/
│   │   ├── ccxt_client.py  # Async CCXT wrapper
│   │   ├── executor.py     # Order submit + retry
│   │   ├── liquidation_feed.py  # Binance !forceOrder@arr WebSocket
│   │   └── ws_stream.py    # Binance kline + aggTrade WebSocket
│   ├── risk/
│   │   ├── position_sizer.py
│   │   ├── risk_manager.py # RegimeClassifier
│   │   ├── kill_switch.py
│   │   └── funding_carry.py  # FundingCarryManager
│   ├── backtest/
│   │   ├── engine.py       # BacktestEngineV3 + synthetic data generator
│   │   ├── metrics.py      # TradeRecord, BacktestMetrics
│   │   └── optimizer.py    # WalkForwardOptimizer
│   ├── notifications/
│   │   └── notifier.py     # AlertNotifier (Telegram + Discord)
│   ├── persistence/
│   │   └── trade_store.py  # TradeStore (aiosqlite)
│   └── production/
│       ├── bot.py          # Async production bot
│       └── multi_bot.py    # Multi-symbol orchestrator
├── data/                   # SQLite database (volume-mounted in Docker)
├── logs/                   # Structured log files (volume-mounted in Docker)
└── tests/
    └── test_strategy.py
```

## Coding Conventions

### Naming
- **Variables/Functions**: `snake_case` (e.g., `calculate_risk_percentage`, `current_position`)
- **Classes**: `PascalCase` (e.g., `GridStrategy`, `RiskManager`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_LEVERAGE`, `DEFAULT_TIMEFRAME`)
- **Private methods**: Prefix with `_` (e.g., `_validate_order`)

### Architecture Rules
- **Never** put API keys in code - always use environment variables or encrypted config
- **Never** hardcode exchange-specific values - use exchange adapters
- **All async functions** must have proper error handling with retry logic
- **Each strategy** must implement the base `Strategy` ABC with `on_tick()`, `on_order_filled()`, `get_parameters()`
- **Position tracking** must be stateless when possible (recover from exchange on restart)

### Type Hints
- **Always use type hints** for function signatures
- **No `Any` types** - be explicit about return types
- Use `Optional[T]` instead of `T | None` for Python <3.10 compatibility

Example:
```python
def calculate_stop_loss(
    entry_price: float,
    side: str,
    risk_percentage: float,
    current_price: Optional[float] = None
) -> float:
    """Calculate stop loss price based on risk percentage."""
    pass
```

## Testing Standards

### What Makes a Good Test
1. **Mock exchange responses** - never hit real API during tests
2. **Test edge cases**: exchange disconnections, rate limits, invalid symbols
3. **Use real market data** from historical CSVs for backtesting validation
4. **Each strategy must have**:
   - Unit test for parameter validation
   - Integration test with mocked exchange
   - Backtest verification on at least 30 days of data

### Required Test Coverage
- **Critical paths**: Exchange connectors, order management, risk calculations → 100%
- **Strategies**: Main logic branches → 90%+
- **Utils**: 80% baseline

### Running Tests
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Error Handling Patterns

### Exchange Errors
```python
from src.utils.retry import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=1.0)
async def place_order(symbol: str, side: str, amount: float):
    try:
        return await exchange.create_order(...)
    except ExchangeError as e:
        logger.error(f"Order failed: {e}")
        # Categorize error: network, auth, rate limit, invalid params
        if "Rate limit" in str(e):
            await asyncio.sleep(rate_limit_delay)
            raise RetryableError(e)
        raise FatalError(e)
```

### Logging Levels
- **ERROR**: Order failures, exchange disconnections, data corruption
- **WARNING**: Rate limits approaching, partial fills, network retries
- **INFO**: Position opens/closes, strategy signals, daily P&L
- **DEBUG**: Tick processing, order book snapshots, raw websocket messages

## Risk Management Rules (NEVER VIOLATE)

### Position Sizing
- **Maximum position per trade**: 2% of portfolio
- **Maximum total open positions**: 6 (diversification)
- **Maximum leverage**: 3x per position, 5x portfolio average

### Stop Loss Rules
- **Always set stop loss** before opening position (exchange-level when possible)
- **Trailing stop**: Minimum 0.5% from current price for 3x leverage
- **Time-based stops**: Close position if no movement after 4 hours in scalping mode

### What NEVER to do
- ❌ Never trade without stop loss
- ❌ Never exceed 80% of portfolio in open positions
- ❌ Never use market orders for large positions (>$10k notional)
- ❌ Never ignore rate limits (max 10 requests/second for public endpoints)
- ❌ Never store unencrypted keys in config files
- ❌ Never trade on leverage >5x regardless of confidence

## Common Codebase Pitfalls

1. **Websocket reconnection**: Exchange websockets drop randomly - always implement exponential backoff reconnect with max 10 retries
2. **Order state sync**: Exchange may report 'open' but order filled - always query order status before relying on cache
3. **Funding rate timing**: Perpetual futures have funding every 8 hours - account for in P&L calculations
4. **Decimal precision**: Different exchanges have different lot sizes - use `decimal.Decimal` not floats for order amounts
5. **Timeframes**: Always use UTC timestamps, never local time for candles

## Configuration Management

### Environment Variables (required)
```bash
EXCHANGE_NAME=binance          # binance, bybit, okx
API_KEY=your_api_key
API_SECRET=your_api_secret
TESTNET=true                    # Always test first!
MAX_RISK_PER_TRADE=0.02        # 2% risk
DEFAULT_LEVERAGE=3
NOTIFICATION_WEBHOOK=          # Discord/Slack webhook for alerts
```

### Strategy Parameters (config/strategies.yaml)
```yaml
grid_strategy:
  enabled: true
  symbol: "BTC/USDT"
  grid_levels: 10
  grid_spread: 0.02  # 2% between levels
  take_profit: 0.01  # 1% per grid level

dca_strategy:
  enabled: false
  symbol: "ETH/USDT"
  initial_order: 0.1  # ETH
  safety_orders: 3
  safety_order_scale: 1.5  # Each safety order 1.5x larger
```

## CI/CD Integration Notes
- **Never run live trading in CI** - use simulation mode only
- **Backtests should complete in <5 minutes** for daily runs
- **Use `--paper-trading` flag** for exchange integration tests
- **Daily scheduled jobs**: Run backtest on last 7 days, check for strategy degradation (>10% drawdown triggers alert)

## Development Workflow

### Adding a New Strategy
1. Create `src/strategies/my_strategy.py`
2. Inherit from `BaseStrategy` and implement required methods
3. Add YAML config in `config/strategies.yaml`
4. Write tests in `tests/test_my_strategy.py`
5. Run backtest on 3 months historical data before paper trading
6. Paper trade for minimum 1 week before live deployment

### Debugging Live Issues
```bash
# Enable debug logging
LOG_LEVEL=DEBUG python main.py --strategy grid_strategy --paper

# Replay market data from logs
python scripts/replay_from_logs.py --log-file logs/trading_2024-01-01.log

# Check position consistency
python scripts/audit_positions.py --exchange binance --fix
```

## Claude-Specific Instructions

### When I ask you to modify code:
1. **Always check for risk rule violations** first (stop loss, leverage, position size)
2. **Run mental backtest** - would this change increase drawdown?
3. **Preserve exchange abstraction** - don't add exchange-specific hacks without adapter
4. **Update tests** for any logic changes

### Documentation Requirements
- **Every public method** needs docstring with parameters, returns, and exceptions
- **Complex strategy logic** needs comments explaining the trading thesis
- **Configuration changes** must update example config files

### Performance Considerations
- **Avoid pandas in hot paths** (tick processing) - use numpy or plain Python
- **Cache exchange metadata** (symbol info, fees) - refresh daily, not per request
- **Batch order updates** when possible instead of single-order polling
- **Memory limit**: Process <500MB for 7 days of 1-minute candles on 10 symbols

## Quick Commands Reference
```bash
# Backtest a strategy
python main.py --backtest --strategy grid_strategy --start 2024-01-01 --end 2024-01-31

# Paper trade (real-time data, fake orders)
python main.py --paper --strategy dca_strategy --symbol ETH/USDT

# Live trading (requires confirmation prompt)
python main.py --live --strategy trend_strategy --risk-profile conservative

# Generate performance report
python scripts/performance_report.py --period 30d --output report.html
```

## Emergency Procedures
1. **Kill switch**: Set `EMERGENCY_STOP=true` in .env → bot closes all positions immediately
2. **Manual override**: Run `python scripts/close_all_positions.py --exchange binance`
3. **Circuit breaker**: If portfolio down >15% in 1 hour, bot auto-disables all strategies and alerts

---

**Remember**: This is real money. Every code change needs risk review. When in doubt, paper trade first.
```

