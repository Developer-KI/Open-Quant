# Quantitative Trading Framework
---

## About the Project

My focus is on **doing things right**: clean dependency graphs, protocol-driven interfaces, statistically sound backtest methodology (no look-ahead, proper train/test/validate splits, deflated Sharpe ratio), and code that is easy to extend without being over-engineered.

**Links:** [ivanov.r.kiril@abv.bg](mailto:ivanov.r.kiril@abv.bg)

---

## Quick Start

```bash
# Python 3.10+
pip install pandas pyarrow numpy streamlit plotly websockets scipy python-dotenv

# Launch the dashboard
streamlit run app/Strategy_Explorer.py

# Collect Hyperliquid ETH data (trades + L2 + funding)
python -m src.data.feeds.hyperliquid --coin ETH --mode all

# Run the full strategy demo (backtest + hypothesis tests)
python trading/backtest_demo.py

# Run Alpaca paper-trading demo
python trading/alpaca_livetest_demo.py --symbol SPY
```

Create a `.env` file in the project root for credentials:

```env
# Alpaca paper trading
ALP_PAPER_KEY=your_key
ALP_PAPER_SECRET=your_secret

# Alpaca live trading (if needed)
ALP_LIVE_KEY=your_key
ALP_LIVE_SECRET=your_secret

# Optional: LSE historical data
LSE_DATA=your_key
```

---

## Full User Guide

### Step 1 — Install & Project Setup

```
Trading/
├── src/            # All library code
├── app/            # Streamlit dashboard
├── trading/        # Runnable demo scripts
├── data/           # Collected market data (auto-created by scrapers)
├── backtest_runs/  # Saved backtest results (auto-created)
└── .env            # Credentials (never commit this)
```

Add `src/` to your Python path (or use `app/_path_setup.py` as a reference). The demo scripts in `trading/` handle this automatically.

### Step 2 — Collect Data

Data is saved as Parquet chunks under `data/<stream>/<exchange>/<symbol>/`.

**Hyperliquid** (crypto perpetuals and spot):

```bash
# All streams: trades, L2 order book, funding rate
python -m src.data.feeds.hyperliquid --coin ETH --mode all

# Trades only
python -m src.data.feeds.hyperliquid --coin ETH --mode trades

# L2 order book, 5 levels of depth
python -m src.data.feeds.hyperliquid --coin ETH --mode l2 --depth 5

# L2 and trades combined (no funding — for spot symbols)
python -m src.data.feeds.hyperliquid --coin ETH --mode l2/trades

# Monitor a wallet for fills and order updates
python -m src.data.feeds.hyperliquid --wallet 0xYOUR_ADDRESS

# Bulk historical L2 download from Hyperliquid S3 archive
python -m src.data.historical.hyperliquid --coin ETH --start 2024-01-01
```

**Binance** (USD-M futures):

```bash
python -m src.data.feeds.binance --coin ETHUSDT --market futures --streams trades l2 funding
```

**Alpaca** (US equities/ETFs/crypto):

```bash
python -m src.data.feeds.alpaca --symbol SPY --timeframe 1Min
```

**Data directory layout:**

```
data/
├── trades/
│   ├── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet
│   ├── HYPERLIQUID_SPOT/<symbol>/*.parquet
│   └── BINANCE_PERPETUALS/<symbol>/*.parquet
├── l2/
│   └── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet
├── funding/
│   └── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet
└── sentiment/
```

### Step 3 — Load & Parse Data

```python
from core.parser import trades_to_ohlcv, l2_to_orderbook, funding_to_snapshots, align_funding_to_ohlcv

# Resample raw tick trades into any bar size
eth_1h = trades_to_ohlcv("data/trades/HYPERLIQUID_PERPETUALS/ETH", timeframe="1h")
# → DataFrame(DatetimeIndex, columns=[open, high, low, close, volume])

# Load L2 snapshots aligned 1:1 with OHLCV bars
l2_snaps = l2_to_orderbook("data/l2/HYPERLIQUID_PERPETUALS/ETH", ohlcv_data=eth_1h)

# Load funding rates aligned to OHLCV bars
fund_snaps = align_funding_to_ohlcv(
    funding_to_snapshots("data/funding/HYPERLIQUID_PERPETUALS/ETH"),
    eth_1h,
)
```

Supported timeframes: `1s 2s 5s 10s 15s 30s 1m 2m 3m 5m 10m 15m 30m 1h 2h 4h 6h 8h 12h 1d`

### Step 4 — Write a Strategy

Subclass `SingleAssetStrategy` and implement two methods:

- `setup_data(data, l2)` — called once before the backtest to precompute indicators into `data`
- `bar(data, idx) → Allocation` — called on each bar to produce a trading signal

```python
from core.models import Allocation, Side
from strategy.built_in import SingleAssetStrategy
from strategy.indicators import ema, rsi

class EmaRsiStrategy(SingleAssetStrategy):
    def __init__(self, symbol: str, fast: int = 50, slow: int = 200, **kw):
        super().__init__(symbol=symbol, **kw)
        self.fast = fast
        self.slow = slow

    @property
    def params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow}

    def setup_data(self, data, l2=None):
        data["ema_fast"] = ema(data["close"], self.fast)
        data["ema_slow"] = ema(data["close"], self.slow)
        data["rsi"]      = rsi(data["close"], 14)

    def bar(self, data, idx: int) -> Allocation:
        if idx < self.slow:
            return Allocation()   # flat — not enough bars yet

        ema_f = data["ema_fast"].iat[idx]
        ema_s = data["ema_slow"].iat[idx]
        rsi_v = data["rsi"].iat[idx]

        if ema_f > ema_s and rsi_v < 80:
            return Allocation(side=Side.LONG, weight=1.0, reason=f"EMA cross up | RSI={rsi_v:.0f}")

        return Allocation()  # flat / hold
```

**`Allocation` fields:**

| Field | Type | Description |
|---|---|---|
| `side` | `Side` | `LONG`, `SHORT`, or `FLAT` (default) |
| `weight` | `float` | Position size fraction 0–1 (multiplied by sizer) |
| `confidence` | `float` | Optional signal confidence 0–1 |
| `reason` | `str` | Log message for debugging |
| `stop_loss` | `float \| None` | Absolute stop price (used by `FixedFractionalSizer`) |
| `take_profit` | `float \| None` | Absolute take-profit price |

**Available indicators** (`from strategy.indicators import ...`):

| Function | Signature |
|---|---|
| `ema` | `(series, span)` |
| `sma` | `(series, window)` |
| `rsi` | `(series, period=14)` |
| `atr` | `(high, low, close, period=14)` |
| `bollinger` | `(series, window=20, num_std=2.0) → (mid, upper, lower)` |
| `vwap_rolling` | `(price, volume, window)` |
| `order_flow_imbalance` | `(bid_vol, ask_vol, window=20)` |

**Built-in strategies** (`from strategy.built_in import ...`):

| Class | Description |
|---|---|
| `SingleAssetStrategy` | Base class — implement `bar()` |
| `CompositeStrategy` | Combines multiple strategies with weights and threshold |
| `BuyAndHoldStrategy` | Always long at full weight |

### Step 5 — Backtest

```python
from core.models import BacktestConfig
from core.universe import Universe
from backtester.engine import Backtester
from backtester.costs import CompositeCostModel, default_cost_stack
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

# Build universe
universe = Universe(symbols=["ETH"])
universe.add_asset("ETH", eth_1h, l2=l2_snaps, funding=fund_snaps)

# Configure backtest
config = BacktestConfig(
    initial_capital=100_000.0,
    taker_fee_bps=5.0,       # 0.05%
    slippage_bps=1.0,
    leverage=1.0,
    max_position_pct=1.0,
)
cost_model = CompositeCostModel(default_cost_stack())

# Run
bt = Backtester(
    strategy=EmaRsiStrategy(symbol="ETH", fast=50, slow=200),
    config=config,
    sizer=FixedNotionalSizer(notional=10_000),
    stop_loss=NopStopLoss(),
    cost_model=cost_model,
)
result = bt.run(universe=universe, timeframe="1h")

# Results
print(result.summary())
result.save("ema_rsi_eth_1h")   # saves log.json, trades.csv, equity_curve.png, signal_log.csv
```

**`BacktestResult` interface:**

| Method / field | Description |
|---|---|
| `.summary()` | dict: Sharpe, Sortino, Calmar, max DD, win rate, total fees, annualised return/vol |
| `.trades_df()` | DataFrame of all closed trades |
| `.equity_curve` | `pd.Series` indexed by timestamp |
| `.signal_log` | Per-bar signal values |
| `.plot_equity(path)` | Saves equity + drawdown PNG |
| `.save(run_name)` | Saves all outputs to `backtest_runs/<run_name>/` |
| `.trades_by_symbol(sym)` | Filter trades in multi-asset runs |
| `.meta["vectorized"]` | `True` if the fast NumPy path ran |

**Vectorised fast path** — activates automatically (10–50x faster) when:
1. `stop_loss=NopStopLoss()` (stops need bar-by-bar high/low; without them the full array is available)
2. `sizer=FixedNotionalSizer(notional=N)` (price-only, no equity dependency)

### Step 6 — Hypothesis Testing & Validation

The `hypothesis` module provides rigorous statistical tools to validate that a strategy has a genuine edge — not just in-sample luck.

**Recommended workflow: Train → Test → Validate**

```python
from hypothesis import (
    TrainTestValidateSplit,
    HypothesisTests,
    PermutationTest,
    BootstrapCI,
    WalkForwardAnalysis,
    DeflatedSharpeRatio,
    report as hypothesis_report,
)

# Split data into 60% train / 20% test / 20% validate (with 10-bar embargo)
ttv = TrainTestValidateSplit.by_fractions(
    universe, train_frac=0.60, test_frac=0.20, embargo_bars=10
)
print(ttv)  # shows date ranges for each split
```

**Phase 1 — Train (strategy design):** Explore and develop strategies on training data only.

```python
train_result = Backtester(strategy=my_strategy, ...).run(universe=ttv.train, timeframe="1h")

# Walk-forward analysis: checks consistency across sub-periods
wfa = WalkForwardAnalysis(
    strategy_cls=EmaRsiStrategy,
    strategy_params={"fast": 50, "slow": 200},
    fixed_params={"symbol": "ETH"},
    config=config, cost_model=cost_model, sizer=sizer, stop_loss=stop_loss,
)
wf = wfa.run(universe=ttv.train, timeframe="1h", n_splits=5, split_method="expanding")
print(f"Consistency: {wf.consistency_score:.0%}  IS/OOS efficiency: {wf.efficiency_ratio:.2f}")
```

**Phase 2 — Test (parameter optimisation):** Tune parameters on the test set. Track the number of trials for DSR correction.

```python
from backtester.stress import ParamSweep

sweep = ParamSweep(
    strategy_cls=EmaRsiStrategy,
    param_grid={"fast": [20, 50, 100], "slow": [100, 150, 200]},
    config=config, cost_model=cost_model, sizer=sizer, stop_loss=stop_loss,
).run(universe=ttv.test, timeframe="1h")

best = sweep.best("sharpe_ratio")
n_trials = 3 * 3  # for DSR input
```

**Phase 3 — Validate (blind final evaluation):** Run the tuned strategy once on the held-out validate set. This number is your true performance estimate.

```python
val_result = Backtester(strategy=best_strategy, ...).run(universe=ttv.validate, timeframe="1h")

# Full statistical battery
tests = HypothesisTests.run_all(val_result)
print(hypothesis_report(tests))

# Permutation test — is the Sharpe ratio better than random trade ordering?
pt = PermutationTest(metric="sharpe_ratio", n_permutations=2_000).run(val_result)
print(f"p={pt.p_value:.4f}  {'Significant' if pt.reject_null else 'Not significant'}")

# Bootstrap 95% confidence intervals
cis = BootstrapCI(n_bootstrap=2_000, ci=0.95).run(val_result)

# Deflated Sharpe Ratio — corrects for the number of parameter combos you tried
dsr = DeflatedSharpeRatio()
d = dsr.compute(val_result, n_trials=n_trials)
print(f"DSR: SR={d.observed_sharpe:.3f}  deflated={d.deflated_sharpe:.3f}  {'Genuine edge' if d.reject_null else 'Likely overfit'}")
```

**Hypothesis tools reference:**

| Class | What it checks |
|---|---|
| `HypothesisTests.run_all(result)` | Sharpe > 0, mean return > 0, win rate > 50%, normality, autocorrelation, stationarity |
| `HypothesisTests.compare(r1, r2)` | Is strategy 1 statistically better than strategy 2? |
| `PermutationTest` | Non-parametric: is the metric better than random permutations of the trade sequence? |
| `BootstrapCI` | Bootstrap confidence intervals for any metric |
| `WalkForwardAnalysis` | Expanding or rolling window sub-period consistency |
| `DeflatedSharpeRatio` | Sharpe corrected for multiple testing (Minkowski's formula) |
| `MultipleComparisonCorrection` | Bonferroni / Holm correction for family-wise error rate |
| `ProbabilityOfBacktestOverfitting` | Bailey et al. CPCV-based overfit probability |
| `TrainTestValidateSplit` | Three-way holdout with configurable fractions and embargo |

### Step 7 — Stress Testing

```python
from backtester.stress import MonteCarloStress, ParamSweep, RegimeStressTest

# Monte Carlo bootstrap — distribution of outcomes from trade resampling
mc = MonteCarloStress(n_simulations=1_000, method="bootstrap")
mc_res = mc.run(backtest_result)
m = mc_res.meta
print(f"Median return: {m['median_return']:.2f}%  5th: {m['5th_pctl_return']:.2f}%  95th: {m['95th_pctl_return']:.2f}%")

# Parameter sweep heatmap
sweep = ParamSweep(
    strategy_cls=EmaRsiStrategy,
    param_grid={"fast": [20, 50, 100], "slow": [100, 150, 200, 250]},
    config=config, cost_model=cost_model, sizer=sizer, stop_loss=stop_loss,
)
result = sweep.run(universe=universe, timeframe="1h")
result.plot_heatmap("fast", "slow", z="sharpe_ratio")

# Regime stress test — how does the strategy perform in different vol / trend regimes?
rst = RegimeStressTest(regime_fn=RegimeStressTest.trend_regime, config=config, cost_model=cost_model)
regime_summary = rst.run(strategy=my_strategy, universe=universe)
print(regime_summary.summary)
```

### Step 8 — Position Sizing

```python
from strategy.sizing import FixedNotionalSizer, FixedFractionalSizer, VolatilityTargetSizer

# Fixed dollar notional per trade
sizer = FixedNotionalSizer(notional=10_000)

# Fixed fraction of equity (e.g. 10% per trade)
sizer = FixedNotionalSizer(equity_pct=0.10)

# Risk a fixed fraction of equity per trade (uses stop distance if available)
sizer = FixedFractionalSizer(risk_frac=0.02)   # risk 2% of equity per trade

# Target a fixed annualised volatility (scales size to hit ~15% annual vol)
sizer = VolatilityTargetSizer(target_vol=0.15)
```

### Step 9 — Stop-Loss Modules

```python
from strategy.stops import NopStopLoss, FixedPercentStop, ATRStop, TrailingStop

# No stop (enables vectorised backtest fast path)
stop = NopStopLoss()

# Fixed percentage SL + optional TP from entry
stop = FixedPercentStop(sl_pct=2.0, tp_pct=4.0)

# ATR-based SL and TP
stop = ATRStop(atr_mult_sl=2.0, atr_mult_tp=3.0, atr_period=14)

# Trailing stop (moves with the market, locks in profit)
stop = TrailingStop(trail_pct=1.5)
```

### Step 10 — Live Trading

**Alpaca (US equities / ETFs / crypto, paper and live):**

```python
from core.models import LiveConfig, ExchangeCredentials
from execution import Engine as LiveEngine
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

cred = ExchangeCredentials(
    exchange="alpaca",
    api_key="ALP_PAPER_KEY",
    api_secret="ALP_PAPER_SECRET",
    testnet=True,   # paper trading
)

config = LiveConfig(
    exchange="alpaca",
    use_testnet=True,
    exchanges=[cred],
    symbol="SPY",
    bar_interval_s=60,       # 1-minute bars
    warmup_bars=300,          # bars before first trade
    max_position_pct=0.10,
    leverage=1.0,
    max_daily_trades=20,
    max_daily_loss_pct=3.0,
    trade_log_csv="trades.csv",
)

engine = LiveEngine(
    strategy=EmaRsiStrategy(symbol="SPY", fast=50, slow=200),
    config=config,
    sizer=FixedNotionalSizer(notional=10_000),
    stop_loss=NopStopLoss(),
)
engine.start()   # press 'q' + Enter to flatten & stop
```

**Hyperliquid (crypto perpetuals, testnet and mainnet):**

```python
from execution import Engine as LiveEngine

config = LiveConfig(
    exchange="hyperliquid",
    account_address="0x...",
    secret_key="0x...",
    use_testnet=True,
    symbol="ETH",
    bar_interval_s=300,     # 5-minute bars
    warmup_bars=200,
    risk_per_trade=0.02,
    max_daily_loss_pct=3.0,
)

engine = LiveEngine(
    strategy=EmaRsiStrategy(symbol="ETH"),
    config=config,
    sizer=VolatilityTargetSizer(target_vol=0.15),
    stop_loss=ATRStop(atr_mult_sl=2.0, atr_mult_tp=3.0),
)
engine.start()
```

**Cross-exchange (funding arbitrage, delta-neutral):**

```python
from execution.multi_exchange_engine import MultiExchangeEngine

engine = MultiExchangeEngine(
    strategy=FundingArbStrategy(),
    credentials=[hl_cred, binance_cred],
    config=live_config,
)
engine.start()
```

---

## Streamlit Dashboard

```bash
streamlit run app/Strategy_Explorer.py
```

| Page | Contents |
|---|---|
| **Strategy Explorer** (main) | OHLCV charts with configurable timeframe; EMA, SMA, Bollinger, RSI, ATR, MACD overlays; volatility regime chart; returns distribution + descriptive stats; strategy signal overlay; raw data download |
| **Backtester** | Signal/strategy selector with auto-generated parameter fields; bar timeframe selector; fee/slippage/leverage config; sizer and stop-loss config; hypothesis tests (DSR, permutation, bootstrap CIs); parameter sweep heatmap; regime stress test; Monte Carlo |

Both pages load data from the LSE API — set `LSE_DATA=your_key` in `.env` or enter it in the sidebar.

---

## Package Layout

```
src/
├── core/                   # Stable contracts — no upward imports
│   ├── models.py           # All data types (Side, Allocation, Position, Trade, …)
│   ├── protocols.py        # typing.Protocol interfaces for C++ interop
│   ├── events.py           # Typed event structs (BarEvent, TradeEvent, L2Event)
│   ├── universe.py         # Universe — holds OHLCV + L2 + funding per symbol
│   └── parser.py           # OHLCV resampling, L2/funding alignment
│
├── strategy/               # Pure-Python strategy framework
│   ├── base.py             # Strategy ABC + Signal ABC + registries
│   ├── built_in.py         # SingleAssetStrategy, CompositeStrategy, …
│   ├── indicators.py       # Stateless indicator functions
│   ├── sizing.py           # Position sizers (FixedNotional, FixedFractional, VolTarget, …)
│   └── stops.py            # Stop-loss modules (NopStop, FixedPct, ATR, Trailing, …)
│
├── backtester/             # Vectorised backtest engine
│   ├── engine.py           # Backtester + BacktestResult
│   ├── costs.py            # Pluggable cost models (fee, slippage, funding, impact)
│   └── stress.py           # ParamSweep, MonteCarloStress, RegimeStressTest
│
├── hypothesis/             # Statistical validation
│   ├── tests.py            # HypothesisTests, PermutationTest, BootstrapCI
│   ├── walk_forward.py     # WalkForwardAnalysis
│   ├── overfitting.py      # DeflatedSharpeRatio, ProbabilityOfBacktestOverfitting
│   └── splits.py           # TrainTestValidateSplit, WalkForwardSplits
│
├── execution/              # Live trading engines
│   ├── engine.py           # LiveEngine (single exchange)
│   ├── multi_exchange_engine.py  # MultiExchangeEngine
│   ├── factory.py          # Registry-based executor + feed factory
│   ├── hyperliquid/        # Hyperliquid executor + WebSocket feed
│   ├── binance/            # Binance USD-M executor + WebSocket feed
│   └── alpaca/             # Alpaca executor + IEX bar feed
│
└── data/
    ├── feeds/              # Live WebSocket scrapers (Hyperliquid, Binance, Alpaca)
    ├── historical/         # Bulk historical downloaders (HL S3, Binance REST, Alpaca)
    └── auxiliary/          # Sentiment (Reddit, Telegram, X, 4chan), macro, microstructure

app/
├── main.py                 # Streamlit entry point
└── pages/
    ├── 1_Visualizer.py     # Data explorer page
    ├── 2_Backtest.py       # Backtester page
    └── 3_Live.py           # Live trading page

trading/
├── backtest_demo.py        # Full TTV workflow with hypothesis tests
└── alpaca_livetest_demo.py # Alpaca paper-trading live demo
```

---

## Adding an Exchange

1. Create `src/execution/<exchange>/` with an executor and feed module inheriting from `BaseExecutor` and `BaseFeed`.
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
