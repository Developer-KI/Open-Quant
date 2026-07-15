# Quantitative Trading Framework

## About

A modular, multi-asset, multi-exchange Python framework for developing, backtesting, stress-testing, and live-trading quantitative strategies on crypto perpetual futures, US equities, and spot. Built with Hyperliquid, Binance Futures, and Alpaca support out of the box, and extensible to any exchange via structural protocols.

My focus is on **doing things right**: clean dependency graphs, protocol-driven interfaces, statistically sound backtest methodology, and code that is easy to extend without being over-engineered.

**Contact:** [ivanov.r.kiril@abv.bg](mailto:ivanov.r.kiril@abv.bg)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Package Layout](#package-layout)
3. [Architecture](#architecture)
4. [User Guide](#user-guide)
   - [Step 1 — Load Data](#step-1--load-data)
   - [Step 2 — Write a Strategy](#step-2--write-a-strategy)
   - [Step 3 — Backtest](#step-3--backtest)
   - [Step 4 — Position Sizing](#step-4--position-sizing)
   - [Step 5 — Stop-Loss Modules](#step-5--stop-loss-modules)
   - [Step 6 — Hypothesis Testing & Validation](#step-6--hypothesis-testing--validation)
   - [Step 7 — Stress Testing](#step-7--stress-testing)
   - [Step 8 — Live Trading](#step-8--live-trading)
5. [Extending the Framework](#extending-the-framework)
   - [Adding a New Exchange](#adding-a-new-exchange)
   - [Custom Data Sources](#custom-data-sources)
   - [Custom Sizers, Stops, and Cost Models](#custom-sizers-stops-and-cost-models)

---

## Quick Start

```bash
# Python 3.10+
pip install pandas pyarrow numpy streamlit plotly websockets scipy python-dotenv

# Launch the strategy explorer dashboard
streamlit run app/Strategy_Explorer.py

# Run the full backtest demo (single-asset + multi-asset + multi-exchange + hypothesis tests)
python trading/backtest_demo.py

# Run Alpaca paper-trading live demo
python trading/alpaca_livetest_demo.py --symbol SPY
```

Create a `.env` file in the project root:

```env
# Hyperliquid
HL_ACCOUNT_ADDRESS=0x...
HL_SECRET_KEY=0x...

# Alpaca paper trading
ALP_PAPER_KEY=your_key
ALP_PAPER_SECRET=your_secret

# Alpaca live trading
ALP_LIVE_KEY=your_key
ALP_LIVE_SECRET=your_secret

# London Strategic Edge historical data (dashboard + backtest demos)
LSE_DATA=your_key
```

---

## Package Layout

```
src/
├── core/                        # Stable contracts — no upward imports
│   ├── models.py                # Side, Allocation, Position, Trade, FillResult,
│   │                            # BacktestConfig, LiveConfig, ExchangeCredentials, …
│   ├── protocols.py             # typing.Protocol interfaces (C++ interop seam)
│   ├── events.py                # BarEvent, TradeEvent, L2Event structs
│   ├── universe.py              # Universe — holds OHLCV + L2 + funding per symbol
│   ├── feeds.py                 # BaseFeed, BaseBarBuilder base classes
│   └── parser.py                # trades_to_ohlcv, l2_to_orderbook, funding helpers
│
├── strategy/                    # Pure-Python strategy framework
│   ├── base.py                  # Strategy, StrategyContext, PortfolioTarget, registries
│   ├── built_in.py              # SingleAssetStrategy, CompositeStrategy,
│   │                            # PerAssetStrategy, ZPairsSpreadStrategy, …
│   ├── indicators.py            # Stateless indicator functions
│   ├── sizing.py                # Sizer hierarchy (FixedNotional, VolTarget, Kelly, …)
│   ├── stops.py                 # StopLoss hierarchy (NopStop, ATR, Trailing, …)
│   └── overlay.py               # NetExposureOverlay, DeltaNeutralOverlay
│
├── testing/
│   ├── backtester/
│   │   ├── engine.py            # Backtester + BacktestResult
│   │   ├── costs.py             # Pluggable cost models (fee, slippage, impact, funding)
│   │   └── stress.py            # ParamSweep, MonteCarloStress, RegimeStressTest
│   └── hypothesis/
│       ├── tests.py             # HypothesisTests, PermutationTest, BootstrapCI
│       ├── walk_forward.py      # WalkForwardAnalysis
│       ├── overfitting.py       # DeflatedSharpeRatio, ProbabilityOfBacktestOverfitting
│       └── splits.py            # TrainTestValidateSplit, WalkForwardSplits
│
├── execution/                   # Live trading engine (single- and multi-exchange)
│   ├── engine.py                # Engine — handles single, per-exchange, cross-exchange
│   ├── executor.py              # BaseExecutor ABC
│   ├── portfolio.py             # MultiExchangePortfolio
│   ├── factory.py               # Registry-based executor + feed factory
│   ├── state.py                 # LiveState, _AssetLiveState, _ManualKillSwitch
│   ├── alpaca/                  # Alpaca executor (paper + live)
│   ├── binance/                 # Binance USD-M executor + WebSocket feed
│   └── hyperliquid/             # Hyperliquid executor + WebSocket feed
│
└── data/
    ├── feeds/                   # Live WebSocket feeds (Hyperliquid, Binance, Alpaca)
    └── auxiliary/macro/         # Macro / on-chain data helpers

app/
├── Strategy_Explorer.py         # Streamlit dashboard (EDA + backtester + hypothesis + stress)
└── components/                  # Chart builders, sidebar forms, data fetching, engine runner

trading/
├── backtest_demo.py             # End-to-end demo: single/multi-asset, TTV workflow, tests
└── alpaca_livetest_demo.py      # Alpaca paper-trading live demo
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          BACKTESTER                                │
│  Strategy ──→ Engine loop ──→ Sizer ──→ StopLoss ──→ CostModel     │
│    ↑          (bar-by-bar)                                         │
│    │                                                               │
│  Universe (OHLCV + L2 + FundingSnapshots + Aux DataSources)        │
└────────────────────────────────────────────────────────────────────┘
          ↕ same Strategy interface
┌────────────────────────────────────────────────────────────────────┐
│                        LIVE ENGINE                                 │
│  Strategy ──→ Engine._process_bar() ──→ Sizer ──→ StopLoss         │
│    ↑          (on each bar close)           │                      │
│    │                                        ↓                      │
│  rolling Universe                      BaseExecutor                │
│    ↑                          (HyperliquidExecutor / BinanceExec   │
│    │                           / AlpacaExecutor)                   │
│    │                                                               │
│  BarBuilder ←── BaseFeed (WebSocket)                               │
│    (trades → OHLCV)    (L2 book + trades + candles)                │
│                                                                    │
│  _ManualKillSwitch (stdin listener: press 'q' + Enter to flatten)  │
└────────────────────────────────────────────────────────────────────┘
          ↕ extends to multiple exchanges
┌────────────────────────────────────────────────────────────────────┐
│                    MULTI-EXCHANGE ENGINE                           │
│  Strategy ──→ PortfolioTarget.exchange_allocations                  │
│    ↑                          │                                    │
│    │                          ├──→ HyperliquidExecutor             │
│    │                          └──→ BinanceExecutor                 │
│    │                                                               │
│  Per-exchange Universes + MultiExchangePortfolio                   │
│  PortfolioOverlays (net exposure cap, delta-neutral, …)            │
└────────────────────────────────────────────────────────────────────┘
```

The **C++ seam** lives in `core/protocols.py`. `ExecutorProtocol`, `FeedProtocol`, and `BarBuilderProtocol` are `typing.Protocol` definitions — pybind11-wrapped C++ classes satisfy them structurally without inheriting from any Python base class. The live engine and backtester only depend on these Protocols, so Python and C++ implementations are interchangeable.

**Dependency chain:**

```
core/        Shared models, protocols, universe (no upstream deps)
     ↑
strategy/    Strategy logic, indicators, sizing, stops, overlays
     ↑
testing/     Backtester engine, cost models, stress tests, hypothesis tests
     ↑
execution/   Live engine, exchange adapters, WebSocket feeds
```

---

## User Guide

### Step 1 — Load Data

Load raw OHLCV from any source and build a `Universe`. For backtest demos the project uses LSE (London Strategic Edge); for live it's exchange WebSocket feeds.

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

Then wrap the data in a `Universe`:

```python
from core.universe import Universe, StaticDataSource

universe = Universe(symbols=["ETH"])
universe.add_asset("ETH", eth_1h, l2=l2_snaps, funding=fund_snaps)

# Optional: auxiliary data sources (sentiment, on-chain, macro, etc.)
universe.add_data_source(StaticDataSource("sentiment", sentiment_df))
```

---

### Step 2 — Write a Strategy

All strategies subclass `Strategy`. The base class unifies single- and multi-exchange behaviour: if one exchange is used it behaves like a plain single-exchange strategy; if multiple exchanges are passed in `setup()` it receives all of them in context.

For the common single-asset case, subclass `SingleAssetStrategy` instead — it handles universe wiring automatically and provides a simpler `bar()` interface.

#### Single-asset strategy

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
            return Allocation()

        if data["ema_fast"].iat[idx] > data["ema_slow"].iat[idx] and data["rsi"].iat[idx] < 80:
            return Allocation(side=Side.LONG, weight=1.0, reason="EMA cross up")

        return Allocation()
```

#### Multi-asset / multi-exchange strategy

For more control — or when operating across multiple exchanges simultaneously — subclass `Strategy` directly and implement `generate()`, which returns a `PortfolioTarget`:

```python
from strategy.base import Strategy, StrategyContext, PortfolioTarget, register_strategy
from core.models import Allocation, Side


@register_strategy("momentum_basket")
class MomentumBasketStrategy(Strategy):
    def __init__(self, lookback: int = 20, top_n: int = 2, total_weight: float = 0.8, **kw):
        super().__init__(**kw)
        self.lookback = lookback
        self.top_n = top_n
        self.total_weight = total_weight

    @property
    def params(self):
        return {"lookback": self.lookback, "top_n": self.top_n, "total_weight": self.total_weight}

    def setup(self, universe):
        self._symbols = universe.symbols

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget(timestamp=ctx.timestamp)
        if ctx.bar_idx < self.lookback:
            return target

        scores = {}
        for sym in self._symbols:
            close = ctx.universe.ohlcv(sym)["close"]
            cur, prev = close.iat[ctx.bar_idx], close.iat[ctx.bar_idx - self.lookback]
            if prev > 0:
                scores[sym] = (cur - prev) / prev

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        per_w = self.total_weight / self.top_n

        for sym, mom in ranked[:self.top_n]:
            target[sym] = Allocation(side=Side.LONG, weight=per_w,
                                     confidence=min(abs(mom) * 10, 1.0),
                                     reason=f"mom={mom:.4f}")
        target.normalize(self.total_weight)
        return target
```

For multi-exchange strategies, key allocations by `(exchange, symbol)` tuple:

```python
def generate(self, ctx: StrategyContext) -> PortfolioTarget:
    target = PortfolioTarget(timestamp=ctx.timestamp)
    target[("hyperliquid", "ETH")] = Allocation(side=Side.LONG,  weight=0.3)
    target[("binance",     "ETH")] = Allocation(side=Side.SHORT, weight=0.3)
    return target
```

#### Reference tables

**`Allocation` fields:**

| Field | Type | Description |
|---|---|---|
| `side` | `Side` | `LONG`, `SHORT`, or `FLAT` (default) |
| `weight` | `float` | Position size fraction 0–1 (scaled by sizer) |
| `confidence` | `float` | Optional signal confidence 0–1 |
| `reason` | `str` | Debug / signal log string |
| `stop_loss` | `float \| None` | Absolute stop price |
| `take_profit` | `float \| None` | Absolute take-profit price |

**`StrategyContext` fields:**

| Field | Description |
|---|---|
| `universe` | Active `Universe` (single-exchange) |
| `universes` | `dict[str, Universe]` — all exchanges |
| `equity` | Total equity across all exchanges |
| `equity_by_exchange` | `dict[str, float]` — equity per exchange |
| `positions` | `dict[str, Position]` — primary exchange positions |
| `all_positions` | `dict[str, dict[str, Position]]` — positions per exchange |
| `bar_idx` | Current bar index |
| `timestamp` | Current bar timestamp |
| `trade_history` | List of closed `Trade` objects |

Key methods: `ctx.price(sym)`, `ctx.ohlcv(sym)`, `ctx.l2(sym)`, `ctx.funding(sym)`, `ctx.is_positioned(sym)`, `ctx.net_exposure()` — all accept an optional `exchange=` keyword.

**`PortfolioTarget` interface:**

```python
target["ETH"]                   # single-exchange allocation
target[("nyse", "AAPL")]        # multi-exchange allocation
target.is_multi_exchange         # True when exchange_allocations is populated
target.for_exchange("nyse")      # dict[str, Allocation] for one exchange
target.exchanges                 # list of exchange names present
target.normalize(max_total=1.0)  # scale weights down proportionally
```

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
| `book_imbalance` | `(l2_snapshot)` |

**Built-in strategies** (`from strategy.built_in import ...`):

| Class | Description |
|---|---|
| `SingleAssetStrategy` | Base for single-symbol strategies — implement `bar()` |
| `CompositeStrategy` | Combines multiple strategies with weights and a vote threshold |
| `PerAssetStrategy` | Runs one `SingleAssetStrategy` instance per symbol in the universe |
| `ZPairsSpreadStrategy` | Z-score pairs trading (registered as `"pairs_z_spread"`) |
| `CrossAssetMomentumStrategy` | Long top N / short bottom N by return (registered as `"cross_asset_momentum"`) |
| `MeanReversionBasketStrategy` | Z-score + RSI mean reversion (registered as `"mean_reversion_basket"`) |

**Portfolio overlays** (`from strategy.overlay import ...`):

| Class | What it does |
|---|---|
| `NetExposureOverlay` | Caps net directional exposure across all exchanges per symbol |
| `DeltaNeutralOverlay` | Auto-hedges residual exposure on a specified hedge exchange |

---

### Step 3 — Backtest

#### Single-exchange backtest

```python
from core.models import BacktestConfig
from core.universe import Universe
from testing.backtester.engine import Backtester
from testing.backtester.costs import CompositeCostModel, default_cost_stack
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

universe = Universe(symbols=["ETH"])
universe.add_asset("ETH", eth_1h, l2=l2_snaps, funding=fund_snaps)

config = BacktestConfig(initial_capital=100_000.0, leverage=1.0, max_position_pct=1.0)

bt = Backtester(
    strategy=EmaRsiStrategy(symbol="ETH", fast=50, slow=200),
    config=config,
    sizer=FixedNotionalSizer(notional=10_000),
    stop_loss=NopStopLoss(),
    cost_model=CompositeCostModel(default_cost_stack()),
)

result = bt.run(universe=universe, timeframe="1h")
print(result.summary())
result.save("ema_rsi_eth_1h")   # → backtest_runs/ema_rsi_eth_1h/
```

#### Multi-exchange backtest

Pass `universes` (a dict of exchange name → `Universe`) instead of `universe`. Use `exchange_costs` for per-exchange fee models and `capital_by_exchange` to split the starting capital:

```python
from testing.backtester.costs import ExchangeFeeCost, FixedSlippageCost

u_hl = Universe(symbols=["ETH"]); u_hl.add_asset("ETH", eth_df)
u_bn = Universe(symbols=["ETH"]); u_bn.add_asset("ETH", eth_df)

bt = Backtester(
    strategy=my_cross_exchange_strategy,
    config=BacktestConfig(initial_capital=100_000.0),
    exchange_costs={
        "hyperliquid": CompositeCostModel([ExchangeFeeCost(maker_bps=2,  taker_bps=2.5)]),
        "binance":     CompositeCostModel([ExchangeFeeCost(maker_bps=2,  taker_bps=4.0),
                                           FixedSlippageCost(slippage_bps=1)]),
    },
    capital_by_exchange={"hyperliquid": 50_000.0, "binance": 50_000.0},
)

result = bt.run(universes={"hyperliquid": u_hl, "binance": u_bn}, timeframe="1h")
print(result.equity_curves_by_exchange)
result.save("cross_exchange_demo")
```

#### `BacktestResult` interface

| Member | Description |
|---|---|
| `.summary()` | `dict` — Sharpe, Sortino, Calmar, max DD, win rate, total fees, CAGR, and more |
| `.equity_curve` | `pd.Series` — total equity across all exchanges |
| `.equity_curves_by_exchange` | `dict[str, pd.Series]` — per-exchange equity (multi-exchange) |
| `.trades_df()` | `DataFrame` of all closed trades |
| `.trades_by_symbol(sym)` | Per-symbol trade filter (multi-asset runs) |
| `.plot_equity()` | Equity curve + drawdown chart |
| `.to_csv(path)` | Export trades to CSV |
| `.save(run_name)` | Write `log.json`, `trades.csv`, `equity_curve.png` to `backtest_runs/<run_name>/` |
| `.positions_log` / `.allocation_log` | Per-bar, per-asset logs (multi-asset only) |

**Vectorised fast path** — activates automatically (10–50×) when using `NopStopLoss()` and `FixedNotionalSizer`. Single-exchange only.

---

### Step 4 — Position Sizing

```python
from strategy.sizing import (
    FixedNotionalSizer,     # fixed dollar notional per trade
    FixedFractionalSizer,   # risk a fixed fraction of equity per trade
    VolatilityTargetSizer,  # scale size to target ~15% annual vol
    KellySizer,             # Kelly criterion (uses trade history)
    AntiMartingaleSizer,    # increase after wins, decrease after losses
    DrawdownScalingSizer,   # reduce size when drawdown exceeds threshold
    L2LiquiditySizer,       # scale to available order book depth
    CompositeSizer,         # take min/max/mean across multiple sizers
)

sizer = FixedNotionalSizer(notional=10_000)
sizer = FixedFractionalSizer(risk_frac=0.02)
sizer = VolatilityTargetSizer(target_vol=0.15, lookback=20)
sizer = KellySizer(kelly_frac=0.5, min_trades=20)

# Composite — most conservative of vol-target and Kelly
sizer = CompositeSizer(
    sizers=[VolatilityTargetSizer(target_vol=0.15), KellySizer(kelly_frac=0.5)],
    mode="min",
)
```

Sizers can be a single shared instance or a `dict[symbol, Sizer]` for per-asset overrides.

---

### Step 5 — Stop-Loss Modules

```python
from strategy.stops import (
    NopStopLoss,          # no stop; required for vectorised fast path
    FixedPercentStop,     # fixed % SL + optional TP
    ATRStop,              # ATR-based SL and TP
    TrailingStop,         # trailing %, locks in profit
    TrailingATRStop,      # trailing + ATR-based
    BreakevenStop,        # move SL to breakeven after initial profit target
    TimeStop,             # exit after N bars
    RiskRewardStop,       # SL + auto-computed TP from R:R ratio
    EmbeddedStop,         # delegate to exchange native stop order
    CompositeStopLoss,    # first to trigger wins
)

stop = NopStopLoss()
stop = FixedPercentStop(sl_pct=2.0, tp_pct=4.0)
stop = ATRStop(atr_mult_sl=2.0, atr_mult_tp=3.0)
stop = TrailingStop(trail_pct=1.5)
stop = CompositeStopLoss([TrailingATRStop(atr_mult=2.5), TimeStop(max_bars=48)])
```

Stop-losses are stateful per position: `on_entry()` initializes, `update()` runs each bar, `check()` returns a `StopResult`.

---

### Step 6 — Hypothesis Testing & Validation

**Recommended workflow: Train → Test → Validate**

```python
from testing.hypothesis import (
    TrainTestValidateSplit,
    HypothesisTests,
    PermutationTest,
    BootstrapCI,
    WalkForwardAnalysis,
    DeflatedSharpeRatio,
    report as hypothesis_report,
)

# Split data 60% train / 20% test / 20% validate (10-bar embargo between splits)
ttv = TrainTestValidateSplit.by_fractions(universe, train_frac=0.60, test_frac=0.20, embargo_bars=10)
```

**Phase 1 — Train:** Develop the strategy. Check walk-forward consistency.

```python
train_result = Backtester(strategy=my_strategy, ...).run(universe=ttv.train, timeframe="1h")

wfa = WalkForwardAnalysis(
    strategy_cls=EmaRsiStrategy,
    strategy_params={"fast": 50, "slow": 200},
    fixed_params={"symbol": "ETH"},
    config=config, cost_model=cost, sizer=sizer, stop_loss=stop,
)
wf = wfa.run(universe=ttv.train, timeframe="1h", n_splits=5, split_method="expanding")
print(f"Consistency: {wf.consistency_score:.0%}  IS/OOS efficiency: {wf.efficiency_ratio:.2f}")
```

**Phase 2 — Test:** Optimise parameters. Track the number of trials for DSR correction.

```python
from testing.backtester.stress import ParamSweep

sweep = ParamSweep(
    strategy_cls=EmaRsiStrategy,
    param_grid={"fast": [20, 50, 100], "slow": [100, 150, 200]},
    config=config, cost_model=cost, sizer=sizer, stop_loss=stop,
).run(universe=ttv.test, timeframe="1h")

best = sweep.best("sharpe_ratio")
n_trials = 3 * 3
```

**Phase 3 — Validate:** Run the tuned strategy once on the held-out set. This is the honest number.

```python
val_result = Backtester(strategy=best_strategy, ...).run(universe=ttv.validate, timeframe="1h")

# Full statistical battery
tests = HypothesisTests.run_all(val_result)
print(hypothesis_report(tests))

# Permutation test
pt = PermutationTest(metric="sharpe_ratio", n_permutations=2_000).run(val_result)
print(f"p={pt.p_value:.4f}  {'Significant' if pt.reject_null else 'Not significant'}")

# Bootstrap 95% CIs
cis = BootstrapCI(n_bootstrap=2_000, ci=0.95).run(val_result)

# Deflated Sharpe — corrects for the number of param combos tried
dsr = DeflatedSharpeRatio().compute(val_result, n_trials=n_trials)
print(f"DSR: {dsr.deflated_sharpe:.3f}  {'Genuine edge' if dsr.reject_null else 'Likely overfit'}")
```

**Hypothesis tools reference:**

| Class | What it checks |
|---|---|
| `HypothesisTests.run_all(result)` | Sharpe > 0, mean return > 0, win rate > 50%, normality, autocorrelation, stationarity |
| `HypothesisTests.compare(r1, r2)` | Is strategy 1 statistically better than strategy 2? |
| `PermutationTest` | Is the metric better than random permutations of the trade sequence? |
| `BootstrapCI` | Bootstrap confidence intervals for any metric |
| `WalkForwardAnalysis` | Expanding or rolling sub-period consistency |
| `DeflatedSharpeRatio` | Sharpe corrected for multiple testing (Bailey & López de Prado) |
| `MultipleComparisonCorrection` | Bonferroni / BH correction for family-wise error rate |
| `ProbabilityOfBacktestOverfitting` | CPCV-based overfit probability |
| `TrainTestValidateSplit` | Three-way holdout with configurable fractions and embargo |

---

### Step 7 — Stress Testing

```python
from testing.backtester.stress import MonteCarloStress, ParamSweep, RegimeStressTest

# Monte Carlo bootstrap — distribution of outcomes from trade resampling
mc = MonteCarloStress(n_simulations=1_000, method="bootstrap")
mc_res = mc.run(backtest_result)
m = mc_res.meta
print(f"Median return: {m['median_return']:.2f}%  5th: {m['5th_pctl_return']:.2f}%")

# Parameter sweep heatmap
sweep = ParamSweep(
    strategy_cls=EmaRsiStrategy,
    param_grid={"fast": [20, 50, 100], "slow": [100, 150, 200, 250]},
    config=config, cost_model=cost, sizer=sizer, stop_loss=stop,
)
res = sweep.run(universe=universe, timeframe="1h")
res.plot_heatmap("fast", "slow", z="sharpe_ratio")

# Cost stress test — find where alpha breaks down under higher fees
from testing.backtester.stress import CostStressTest
cst = CostStressTest(cost_grid={
    "ExchangeFeeCost": {"taker_bps": [3, 5, 8, 12]},
    "SpreadCost":      {"default_spread_bps": [1, 2, 4, 8]},
})
cst.run(strategy=my_strategy, universe=universe)

# Regime stress test — performance across vol / trend / volume regimes
rst = RegimeStressTest(regime_fn=RegimeStressTest.trend_regime, config=config, cost_model=cost)
regime_summary = rst.run(strategy=my_strategy, universe=universe)
print(regime_summary.summary)
```

---

### Step 8 — Live Trading

The `Engine` handles three modes from a single class:

| Mode | Constructor argument |
|---|---|
| Single exchange | `strategy=my_strategy` |
| Independent strategy per exchange | `per_exchange_strategies={"binance": s1, "hyperliquid": s2}` |
| Cross-exchange strategy (funding arb, stat arb) | `cross_strategy=my_strategy` |

**Alpaca (US equities / ETFs, paper and live):**

```python
from core.models import LiveConfig, ExchangeCredentials
from execution.engine import Engine
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

config = LiveConfig(
    exchanges=[ExchangeCredentials(
        exchange="alpaca",
        api_key="ALP_PAPER_KEY",
        api_secret="ALP_PAPER_SECRET",
        testnet=True,
    )],
    symbol="SPY",
    bar_interval_s=60,
    warmup_bars=300,
    max_position_pct=0.10,
    leverage=1.0,
    max_daily_trades=20,
    max_daily_loss_pct=3.0,
)
engine = Engine(
    strategy=EmaRsiStrategy(symbol="SPY", fast=50, slow=200),
    config=config,
    sizer=FixedNotionalSizer(notional=10_000),
    stop_loss=NopStopLoss(),
)
engine.start()   # press 'q' + Enter to flatten & stop
```

**Hyperliquid (crypto perpetuals):**

```python
from strategy.stops import TrailingATRStop, TimeStop, CompositeStopLoss
from strategy.sizing import VolatilityTargetSizer

config = LiveConfig(
    exchanges=[ExchangeCredentials(
        exchange="hyperliquid",
        account_address="0x...",
        secret_key="0x...",
        testnet=True,
    )],
    symbols=["ETH", "BTC"],
    bar_interval_s=300,
    warmup_bars=200,
    leverage=2.0,
    max_daily_loss_pct=3.0,
)
engine = Engine(
    strategy=my_strategy,
    config=config,
    sizer=VolatilityTargetSizer(target_vol=0.15),
    stop_loss=CompositeStopLoss([TrailingATRStop(atr_mult=2.5), TimeStop(max_bars=48)]),
)
engine.start()
```

**Multi-exchange (cross-exchange or independent strategies per exchange):**

```python
from strategy.overlay import NetExposureOverlay

config = LiveConfig(
    exchanges=[
        ExchangeCredentials(exchange="hyperliquid", account_address="0x...", secret_key="0x...", testnet=True),
        ExchangeCredentials(exchange="binance", api_key="...", api_secret="...", testnet=True,
                            symbol_map={"ETH": "ETHUSDT", "BTC": "BTCUSDT"}),
    ],
    symbols=["ETH", "BTC"],
    bar_interval_s=60,
    warmup_bars=200,
    leverage=2.0,
)

# Option A: cross-exchange strategy (funding arb, stat arb, hedging)
engine = Engine(cross_strategy=my_arb_strategy, config=config)

# Option B: independent strategies per exchange + risk overlay
engine = Engine(
    per_exchange_strategies={"hyperliquid": momentum_strategy, "binance": mean_reversion_strategy},
    overlay=NetExposureOverlay(max_net_weight=0.5),
    config=config,
)
engine.start()
```

**Risk controls built into the engine:**

- **Daily loss kill switch** — flattens all positions and shuts down when daily PnL loss exceeds `max_daily_loss_pct`.
- **Manual kill switch** — press `q` + Enter to immediately flatten all positions. Silently disables on non-interactive terminals.
- **Daily trade limit** — `max_daily_trades` caps new trades per day.
- **Position sync** — each bar, local state is compared against the exchange and mismatches are logged.
- **Leverage** — applied to each symbol at startup via `config.leverage` and `config.margin_type`.

---

## Extending the Framework

### Adding a New Exchange

Implement `BaseExecutor` and register it in `factory.py`:

```python
# execution/myexchange/myexchange_executor.py
from execution.executor import BaseExecutor
from core.models import Side, Position, FillResult, FundingSnapshot


class MyExchangeExecutor(BaseExecutor):
    @property
    def exchange_name(self) -> str:
        return "myexchange"

    def get_equity(self) -> float: ...
    def get_position(self, symbol) -> Position: ...
    def get_mid_price(self, symbol) -> float: ...
    def get_open_orders(self, symbol) -> list[dict]: ...
    def market_order(self, symbol, side, size, reduce_only=False) -> FillResult: ...
    def limit_order(self, symbol, side, size, price, reduce_only=False) -> FillResult: ...
    def cancel_all(self, symbol) -> int: ...
    def close_position(self, symbol) -> FillResult: ...
    def set_leverage(self, symbol, leverage, cross=True): ...
    def fetch_historical_candles(self, symbol, interval, start_ms, end_ms) -> list[dict]: ...
    def fetch_funding_rate(self, symbol) -> FundingSnapshot | None: ...
```

The same contract is also expressed as `ExecutorProtocol` in `core/protocols.py` — a pybind11-wrapped C++ class satisfies it structurally without any Python base class.

Register in `execution/factory.py`:

```python
elif name == "myexchange":
    from .myexchange.myexchange_executor import MyExchangeExecutor
    return MyExchangeExecutor(...)
```

Similarly for `BaseFeed`: implement `start(on_trade, on_candle, on_l2)`, `stop()`, and the `latest_l2` property, then register in the feed registry.

### Custom Data Sources

```python
from core.universe import StaticDataSource, CallableDataSource

# Static (backtesting — pre-loaded DataFrame)
universe.add_data_source(StaticDataSource("sentiment", sentiment_df))

# Live (queries an API on each call)
def fetch_onchain(symbols, start=None, end=None):
    return pd.DataFrame(...)

universe.add_data_source(CallableDataSource("onchain", fetch_onchain))

# Access in strategy:
# ctx.aux("sentiment")  → DataFrame up to current bar
```

### Custom Sizers, Stops, and Cost Models

All three follow the same pattern — subclass the ABC, implement the core method, expose `params`:

```python
from strategy.sizing import Sizer, SizingContext


class MyCustomSizer(Sizer):
    def __init__(self, param_a: float = 0.5):
        self.param_a = param_a

    @property
    def params(self):
        return {"param_a": self.param_a}

    def compute(self, ctx: SizingContext) -> float:
        return ctx.equity * self.param_a / ctx.price
```

Cost models return total cost in quote currency for a fill. Stack them via `CompositeCostModel`.
