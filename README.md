# Quantitative Trading Framework

A modular, multi-asset, multi-exchange Python framework for developing, backtesting, and live-trading quantitative strategies on crypto perpetual futures and spot markets. Hyperliquid and Binance are supported out of the box; any exchange can be plugged in via the registry pattern.

Designed with a clean C++/Python seam: execution-critical hot paths are isolated behind `typing.Protocol` interfaces so they can be replaced with `pybind11` C++ extensions without touching strategy code.

---

## Quick Start

```bash
# Install dependencies (Python 3.10+)
pip install pandas pyarrow numpy streamlit plotly websockets

# Launch the dashboard
streamlit run app/main.py

# Collect data (Hyperliquid ETH — all streams)
python -m src.data.feeds.hyperliquid --coin ETH --mode all

# Run the backtest demo
python trading/strategy_backtest_demo.py
```

---

## Data Directory Layout

The scrapers and app share a single root:

```
data/
├── trades/
│   ├── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet   # raw tick trades
│   ├── HYPERLIQUID_SPOT/<symbol>/*.parquet
│   └── BINANCE_PERPETUALS/<symbol>/*.parquet
├── l2/
│   └── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet   # L2 order book snapshots
├── funding/
│   └── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet   # funding rate snapshots
└── sentiment/                                       # optional: CSV files
```

Each subfolder is written by the scrapers (`src/data/feeds/`) and read by the app (`app/pages/`) and parsers (`src/core/parser.py`).

---

## Package Layout

```
src/
├── core/               # Stable contracts — models, protocols, events, parser
│   ├── models.py       # ALL data types (Side, Position, Trade, FundingSnapshot, …)
│   ├── protocols.py    # typing.Protocol interfaces for C++ interop
│   ├── events.py       # Typed event structs (BarEvent, TradeEvent, L2Event)
│   └── parser.py       # Parsers: OHLCV resampling, L2, funding rate alignment
│
├── risk/               # Risk layer — depends only on core/
│   ├── sizing.py       # Position sizers (Kelly, VolTarget, Fixed, Composite, …)
│   ├── stops.py        # Stop-loss modules (ATR, Trailing, RiskReward, …)
│   └── limits.py       # Kill-switch + daily loss/trade limits
│
├── strategy/           # Pure-Python strategy framework — depends on core/ + risk/
│   ├── base.py         # Signal, Strategy, CrossExchangeStrategy ABC + registries
│   ├── built_in.py     # SingleSignalStrategy, ZPairsSpread, CrossAssetMomentum, …
│   ├── indicators.py   # Stateless indicator functions (ema, rsi, atr, bollinger, …)
│   ├── universe.py     # Universe + auxiliary data sources (DataSource, StaticDataSource)
│   └── overlay.py      # Cross-exchange portfolio overlays
│
├── backtester/         # Vectorized backtest engine — depends on core/ + risk/ + strategy/
│   ├── engine.py       # Backtester (single-asset and multi-asset APIs)
│   ├── costs.py        # Pluggable cost models (fee, slippage, funding, impact, …)
│   └── stress.py       # Signal / cost / regime / Monte Carlo stress tests
│
├── execution/          # Live trading — depends on core/ + risk/ + strategy/
│   ├── live_engine.py          # LiveEngine (single exchange)
│   ├── multi_exchange_engine.py # MultiExchangeEngine (cross-exchange)
│   ├── live_state.py           # LiveState, _AssetLiveState data holders
│   ├── factory.py              # Registry-based executor + feed factory
│   ├── base_executor_feed.py   # BaseExecutor, BaseFeed, BaseBarBuilder ABCs
│   ├── hyperliquid/            # Hyperliquid executor + WebSocket feed
│   └── binance/                # Binance USD-M executor + WebSocket feed
│
├── data/               # Unified data layer — depends only on core/
│   ├── feeds/
│   │   ├── hyperliquid.py       # HL perp/spot multi-stream (trades, L2, funding, wallet)
│   │   ├── binance.py           # Binance trades/depth/funding + batch backfill
│   │   ├── hyperliquid_bridge.py # Live Arbitrum bridge flows
│   │   └── binance_liquidations.py # Global liquidation stream
│   ├── historical/
│   │   └── hyperliquid_l2.py    # Bulk L2 tick download from Hyperliquid S3 archive
│   ├── sentiment/               # X.com, Reddit, Telegram, 4chan scrapers
│   └── auxiliary/               # Macro REST poller, microstructure scanner
│
└── utils/
    ├── plotting.py              # Equity curve + trade visualisation
    ├── sentiment_score.py       # VADER / transformer sentiment scoring
    └── http.py                  # Lightweight HTTP client
```

### Dependency order

```
core/
  ↑
risk/        strategy/        data/
  ↑               ↑
      backtester/   execution/
                        ↑
                      utils/
```

No module imports upward or sideways outside this DAG.

---

## Parsers (`core/parser.py`)

### Timeframes

All parsers accept a `timeframe` string. Supported values:

| Label | Bar size | Bars / year |
|-------|----------|-------------|
| `"1m"` | 1 minute | 525,600 |
| `"2m"` | 2 minutes | 262,800 |
| `"3m"` | 3 minutes | 175,200 |
| `"5m"` | 5 minutes | 105,120 |
| `"10m"` | 10 minutes | 52,560 |
| `"15m"` | 15 minutes | 35,040 |
| `"30m"` | 30 minutes | 17,520 |
| `"1h"` | 1 hour | 8,760 |
| `"2h"` | 2 hours | 4,380 |
| `"4h"` | 4 hours | 2,190 |
| `"6h"` | 6 hours | 1,460 |
| `"8h"` | 8 hours | 1,095 |
| `"12h"` | 12 hours | 730 |
| `"1d"` | 1 day | 365 |

```python
from core.parser import TIMEFRAMES, timeframe_to_seconds, timeframe_to_bars_per_year
```

### OHLCV

```python
from core.parser import trades_to_ohlcv

# Resample raw trade ticks into any bar size
eth_1m  = trades_to_ohlcv("data/trades/HYPERLIQUID_PERPETUALS/ETH", timeframe="1m")
eth_5m  = trades_to_ohlcv("data/trades/HYPERLIQUID_PERPETUALS/ETH", timeframe="5m")
eth_1h  = trades_to_ohlcv("data/trades/HYPERLIQUID_PERPETUALS/ETH", timeframe="1h")
eth_1d  = trades_to_ohlcv("data/trades/HYPERLIQUID_PERPETUALS/ETH", timeframe="1d")
# → DataFrame(index=DatetimeIndex, columns=[open, high, low, close, volume])
```

### L2 Order Book

```python
from core.parser import l2_to_orderbook, parse_l2, align_l2_to_ohlcv

# Load and parse L2 snapshots (aligned 1:1 with OHLCV bars by default)
snapshots = l2_to_orderbook("data/l2/HYPERLIQUID_PERPETUALS/ETH", ohlcv_data=eth_1m)

# Manual alignment with different methods
raw_snaps = l2_to_orderbook("data/l2/...", aligned=False)
aligned   = align_l2_to_ohlcv(raw_snaps, eth_1m, method="last")     # default
aligned   = align_l2_to_ohlcv(raw_snaps, eth_1m, method="nearest")
aligned   = align_l2_to_ohlcv(raw_snaps, eth_1m, method="vwap")     # merge within bar
```

### Funding Rate

```python
from core.parser import funding_to_snapshots, align_funding_to_ohlcv

# Parse raw funding parquet → list[FundingSnapshot]
fund_snaps = funding_to_snapshots("data/funding/HYPERLIQUID_PERPETUALS/ETH")

# Align one snapshot per OHLCV bar (same interface as L2 alignment)
aligned_funding = align_funding_to_ohlcv(fund_snaps, eth_1m, method="last")
# → each entry has: .rate, .rate_annualized, .mark_price, .oracle_price
```

`FundingSnapshot` fields:

| Field | Type | Description |
|---|---|---|
| `timestamp` | `pd.Timestamp` | When this rate was published |
| `rate` | `float` | Per-period rate (e.g. `0.0001` = 1 bps per 8h) |
| `rate_annualized` | `float` | Annualised rate in bps |
| `oracle_price` | `float` | Spot / index price from the exchange |
| `mark_price` | `float` | Fair price used for funding and liquidation calcs |

---

## Data Models (`core/models.py`)

All data types live in one file with no internal dependencies:

| Type | Purpose |
|---|---|
| `Side` | `LONG` / `SHORT` / `FLAT` enum |
| `OrderBookLevel` | Single price/size level |
| `OrderBookSnapshot` | Full L2 snapshot with timestamp; `.mid`, `.spread`, `.vwap_fill_price()` |
| `FundingSnapshot` | Funding rate + mark/oracle prices |
| `Trade` | Completed trade record (entry, exit, pnl, fees, slippage) |
| `Position` | Open position state |
| `SignalResult` | Signal output (side, weight, confidence, reason, stop/tp prices) |
| `FillResult` | Order fill record from executor |
| `BacktestConfig` | Fee rates, slippage, funding, leverage for backtests |
| `LiveConfig` | Full config for live trading (credentials, symbols, risk limits) |
| `ExchangeCredentials` | API keys + testnet flag per exchange |

---

## Writing a Strategy

### Single-asset signal

```python
from strategy.base import Signal, SignalResult, register_signal
from strategy.indicators import ema
from core.models import Side

@register_signal("ema_cross")
class EMACross(Signal):
    def __init__(self, fast=12, slow=26, **kw):
        super().__init__(**kw)
        self.fast, self.slow = fast, slow

    @property
    def params(self):
        return {"fast": self.fast, "slow": self.slow}

    def setup(self, data, l2=None):
        data["ema_f"] = ema(data["close"], self.fast)
        data["ema_s"] = ema(data["close"], self.slow)

    def generate(self, data, idx):
        if idx < self.slow:
            return SignalResult()
        if data["ema_f"].iat[idx] > data["ema_s"].iat[idx]:
            return SignalResult(target_side=Side.LONG, target_weight=1.0, reason="cross_up")
        if data["ema_f"].iat[idx] < data["ema_s"].iat[idx]:
            return SignalResult(target_side=Side.SHORT, target_weight=1.0, reason="cross_dn")
        return SignalResult()
```

### Multi-asset strategy

```python
from strategy.base import Strategy, StrategyContext, PortfolioTarget, register_strategy
from core.models import Side

@register_strategy("momentum")
class MomentumStrategy(Strategy):
    def setup(self, universe):
        pass  # precompute indicators from universe.ohlcv("ETH"), etc.

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget()
        for symbol in ctx.universe.symbols:
            closes = ctx.universe.close(symbol)
            ret = closes.pct_change(20).iat[ctx.bar_idx]
            side = Side.LONG if ret > 0 else Side.SHORT
            target.set(symbol, side=side, weight=0.5)
        return target
```

---

## Backtesting

```python
from backtester.engine import Backtester
from backtester.costs import CompositeCostModel, aggressive_cost_stack
from core.models import BacktestConfig
from core.parser import trades_to_ohlcv, l2_to_orderbook, funding_to_snapshots, align_funding_to_ohlcv
from strategy.universe import Universe

timeframe = "1h"
eth_ohlcv = trades_to_ohlcv("data/trades/HYPERLIQUID_PERPETUALS/ETH", timeframe=timeframe)

# Optional: load and attach L2 + funding snapshots
l2_snaps   = l2_to_orderbook("data/l2/HYPERLIQUID_PERPETUALS/ETH", eth_ohlcv)
fund_snaps = align_funding_to_ohlcv(
    funding_to_snapshots("data/funding/HYPERLIQUID_PERPETUALS/ETH"),
    eth_ohlcv,
)

universe = Universe(symbols=["ETH"])
universe.add_asset("ETH", eth_ohlcv, l2=l2_snaps, funding=fund_snaps)

config = BacktestConfig(taker_fee_bps=5.0, slippage_bps=1.0)
costs  = CompositeCostModel(models=aggressive_cost_stack())

bt = Backtester(signal=EMACross(fast=12, slow=26), config=config, cost_model=costs)

# timeframe drives accurate annualisation of Sharpe, vol, etc.
result = bt.run(universe=universe, timeframe=timeframe)
print(result.summary())
result.save("ema_cross_1h")
```

`BacktestResult` interface:

| Method / field | Description |
|---|---|
| `.summary()` | dict: Sharpe, Calmar, max DD, win rate, total fees, … |
| `.trades_df()` | DataFrame of all closed trades |
| `.equity_curve` | `pd.Series` indexed by timestamp |
| `.signal_log` | per-bar signal values |
| `.plot_equity(path)` | saves equity + drawdown PNG |
| `.save(run_name)` | saves `log.json`, `trades.csv`, `equity_curve.png`, `signal_log.csv` |
| `.trades_by_symbol(sym)` | filter trades in multi-asset runs |

### Vectorised Fast Path

When two conditions are met, the engine bypasses the Python bar loop entirely and runs a fully NumPy-vectorised backtest that is typically **10–50× faster** on long histories:

| Condition | What enables it |
|---|---|
| Stop-loss = `NopStopLoss` | Stops require bar-by-bar high/low checks; without stops the whole array is available upfront |
| Sizer is `vectorizable` | Position size must be computable from price alone (no equity dependency). `FixedNotionalSizer(notional=…)` qualifies; equity-fraction sizers do not |

**How it works internally:**

1. **Batch signal generation** — `strategy.generate_all(universe)` is called once, returning two dicts of `int8`/`float64` arrays (sides and weights per symbol) without creating any `StrategyContext`/`PortfolioTarget` objects.

2. **Trade boundary detection** (O(M), M = number of trades) — `np.diff` on the sides array finds entry and close bars in one pass. Force-closes at end-of-data are appended explicitly.

3. **Vectorised sizing** — `sizer.compute_vectorized(entry_prices, weights, config)` computes all position sizes in a single numpy call.

4. **Cost arrays** (O(M) Python loop) — entry and exit fees are computed per-trade (not per-bar), then scattered onto sparse arrays via `np.add.at`.

5. **Equity curve via cumsum** — `np.cumsum(close_events - entry_fees)` gives the realised equity at every bar. Unrealised PnL is added via a *forward-fill trick*: `np.maximum.accumulate` on the "last entry bar seen so far" index propagates the entry price and size forward without a Python loop.

```python
from backtester.engine import Backtester
from risk.stops import NopStopLoss
from risk.sizing import FixedNotionalSizer

# Both conditions satisfied → vectorised path activates automatically
bt = Backtester(
    signal=MySignal(),
    stop_loss=NopStopLoss(),
    sizer=FixedNotionalSizer(notional=10_000),
)
result = bt.run(universe=universe, timeframe="1h")
print(result.meta["vectorized"])  # True
```

The result is numerically identical to the sequential path. The equity curve at every bar reflects true realized equity (entry and exit fees deducted at the bars where they occur). To confirm which path ran, check `result.meta["vectorized"]`.

**When the fast path does NOT activate:**
- Any stop-loss other than `NopStopLoss` (ATR stops, trailing stops, etc.)
- `FixedNotionalSizer(equity_pct=…)` (equity-dependent — not vectorizable)
- Any sizer whose `.vectorizable` property returns `False`
- Strategies that return `None` from `generate_all()` (falls back to per-bar loop)

### Stress testing

```python
from backtester.stress import SignalStressTest, CostStressTest, MonteCarloStress

# Sweep signal parameters
sweep = SignalStressTest(
    signal_cls=EMACross,
    param_grid={"fast": [5, 8, 12, 20], "slow": [21, 26, 50]},
)
result = sweep.run(data=eth_ohlcv, timeframe=timeframe)
result.plot_heatmap("fast", "slow", z="sharpe_ratio")

# Bootstrap trade sequence
mc = MonteCarloStress(n_samples=1000)
mc.run(backtest_result).plot_confidence_bands()
```

---

## Live Trading

### Single exchange

```python
from execution.live_engine import LiveEngine
from core.models import LiveConfig
from core.parser import timeframe_to_seconds
from risk.sizing import VolatilityTargetSizer
from risk.stops import ATRStop

config = LiveConfig(
    exchange="hyperliquid",
    account_address="0x...",
    secret_key="0x...",
    use_testnet=True,
    symbol="ETH",
    bar_interval_s=timeframe_to_seconds("5m"),   # 300 seconds
    warmup_bars=200,
    risk_per_trade=0.02,
    max_daily_loss_pct=3.0,
)

engine = LiveEngine(
    signal=EMACross(fast=12, slow=26),
    config=config,
    sizer=VolatilityTargetSizer(target_vol=0.15),
    stop_loss=ATRStop(atr_mult_sl=2.0, atr_mult_tp=3.0),
)
engine.start()
```

### Cross-exchange (funding arb, delta-neutral)

```python
from execution.multi_exchange_engine import MultiExchangeEngine
from strategy.base import CrossExchangeStrategy

engine = MultiExchangeEngine(
    strategy=FundingArbStrategy(),
    credentials=[hl_cred, binance_cred],
    config=live_config,
)
engine.start()
```

---

## Data Collection

### Hyperliquid (trades + L2 + funding)

```bash
# All streams for ETH perpetual
python -m src.data.feeds.hyperliquid --coin ETH --mode all

# Just trades and L2 (no funding; spot does not have funding)
python -m src.data.feeds.hyperliquid --coin ETH --mode l2/trades

# L2 depth 5 levels
python -m src.data.feeds.hyperliquid --coin ETH --mode l2 --depth 5

# Wallet fills + order updates
python -m src.data.feeds.hyperliquid --wallet 0xYOUR_ADDRESS
```

### Binance

```bash
python -m src.data.feeds.binance --coin ETHUSDT --market futures --streams trades l2 funding
```

Data is saved as parquet chunks under `data/<stream>/<exchange>/<symbol>/`.

---

## Streamlit Dashboard

```bash
streamlit run app/main.py
```

Three pages:

| Page | Contents |
|---|---|
| **Data Explorer** | OHLCV charts with configurable bar timeframe; L2 depth + spread; funding rate over time; sentiment scatter; macro indicators |
| **Backtester** | Signal selector + auto-generated param fields; bar timeframe selector; BacktestConfig (fees, slippage, leverage); Sizer; StopLoss; parameter sweep heatmap |
| **Live Trading** | Exchange credentials; bar timeframe dropdown; risk limits; signal/sizer/stop forms; live equity and position dashboard |

---

## Adding an Exchange

1. Create `src/execution/<exchange>/` with executor and feed modules inheriting from `BaseExecutor` and `BaseFeed`.
2. Register it:

```python
from execution.factory import register_exchange
from .my_executor import MyExecutor
from .my_feed import MyFeed

register_exchange(
    "myexchange",
    executor_factory=lambda cred: MyExecutor(cred.api_key, cred.api_secret, cred.testnet),
    feed_factory=lambda symbol, testnet, **_: MyFeed(symbol, testnet),
)
```

3. Pass `ExchangeCredentials(exchange="myexchange", ...)` to `LiveEngine` — the factory resolves the rest.

---

## C++/Python Migration Path

Execution-critical hot paths can be replaced with pybind11 C++ extensions one module at a time, without touching any strategy code:

| Module | Current | Future C++ target |
|---|---|---|
| `core/protocols.py` `BarBuilderProtocol` | Python bar aggregator | Lock-free ring buffer |
| `execution/base_executor_feed.py` `BaseBarBuilder` | Pure Python | C++ OHLCV aggregator |
| `execution/live_state.py` | Python dataclass | C++ POD struct |
| `execution/live_engine.py` `_process_bar` | Python per-bar loop | C++ event loop |

C++ classes that satisfy the `Protocol` shapes in `core/protocols.py` are accepted anywhere in the stack without Python inheritance — no `register()` calls, no base class required.

---

## Demo Scripts

| Script | What it shows |
|---|---|
| `trading/strategy_backtest_demo.py` | EMA cross signal, full backtest with cost model, parameter sweep |
| `trading/strategy_live_demo.py` | Single-asset live run on Hyperliquid testnet |
| `trading/p_strategy.py` | Cross-exchange funding-rate arbitrage (delta-neutral, multi-exchange) |
