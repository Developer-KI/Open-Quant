# Strategy Development Guide

A focused cookbook for writing, testing, and validating trading strategies with this framework. Designed for AI agents or developers who want step-by-step recipes without reading the full codebase.

---

## The minimal strategy

Every strategy is a Python class. You implement two methods; the engine does everything else.

```python
from core.models import Allocation, Side
from strategy.built_in import SingleAssetStrategy

class MyStrategy(SingleAssetStrategy):
    def __init__(self, symbol: str, window: int = 20, **kw):
        super().__init__(symbol=symbol, **kw)
        self.window = window

    @property
    def params(self) -> dict:
        return {"window": self.window}

    def setup_data(self, data, l2=None):
        # Called once. Add columns to `data` (a pandas DataFrame).
        data["sma"] = data["close"].rolling(self.window).mean()

    def bar(self, data, idx: int) -> Allocation:
        # Called on every bar. Return Allocation(side=...) to trade, Allocation() to stay flat.
        if idx < self.window:
            return Allocation()
        if data["close"].iat[idx] > data["sma"].iat[idx]:
            return Allocation(side=Side.LONG, weight=1.0)
        return Allocation()
```

**Rules:**
- Never index beyond `idx` in `bar()` — that is look-ahead bias.
- Return `Allocation()` (no arguments) to stay flat or hold the current position.
- `setup_data` runs before the backtest starts, so it can be slow (vectorised pandas is fine).

---

## Indicator recipes

Import from `strategy.indicators`:

```python
from strategy.indicators import ema, sma, rsi, atr, bollinger, vwap_rolling

# EMA crossover columns
data["fast"] = ema(data["close"], 12)
data["slow"] = ema(data["close"], 26)

# RSI
data["rsi"] = rsi(data["close"], 14)

# ATR for position sizing or stop placement
data["atr"] = atr(data["high"], data["low"], data["close"], 14)

# Bollinger Bands
data["mid"], data["upper"], data["lower"] = bollinger(data["close"], 20, 2.0)

# Rolling VWAP
data["vwap"] = vwap_rolling(data["close"], data["volume"], 20)
```

---

## Common strategy patterns

### Trend-following (EMA cross + RSI filter)

```python
def setup_data(self, data, l2=None):
    data["ema_f"] = ema(data["close"], self.fast)
    data["ema_s"] = ema(data["close"], self.slow)
    data["rsi"]   = rsi(data["close"], 14)

def bar(self, data, idx):
    if idx < self.slow:
        return Allocation()
    f, s, r = data["ema_f"].iat[idx], data["ema_s"].iat[idx], data["rsi"].iat[idx]
    if f > s and r < 70:
        return Allocation(side=Side.LONG,  weight=1.0)
    if f < s and r > 30:
        return Allocation(side=Side.SHORT, weight=1.0)
    return Allocation()
```

### Mean reversion (Bollinger Bands)

```python
def setup_data(self, data, l2=None):
    data["mid"], data["up"], data["lo"] = bollinger(data["close"], 20, 2.0)

def bar(self, data, idx):
    if idx < 20:
        return Allocation()
    c, mid, up, lo = data["close"].iat[idx], data["mid"].iat[idx], data["up"].iat[idx], data["lo"].iat[idx]
    if c < lo:
        return Allocation(side=Side.LONG,  weight=1.0, reason="oversold")
    if c > up:
        return Allocation(side=Side.SHORT, weight=1.0, reason="overbought")
    if Side.LONG  == Side.LONG  and c > mid:   # simplification — track position externally
        return Allocation()
    return Allocation()
```

### Volatility filter (sit out extreme vol)

```python
def setup_data(self, data, l2=None):
    rv = data["close"].pct_change().rolling(20).std()
    data["rv"]   = rv
    data["rv_hi"] = rv.expanding(min_periods=20).quantile(0.66)

def bar(self, data, idx):
    rv, rv_hi = data["rv"].iat[idx], data["rv_hi"].iat[idx]
    if rv != rv or rv_hi != rv_hi:
        return Allocation()
    if rv >= rv_hi:
        return Allocation(reason=f"vol filter: rv={rv:.4f}")
    # … run your signal logic here …
```

### Composite strategy (vote of two signals)

```python
from strategy.built_in import CompositeStrategy

composite = CompositeStrategy(
    symbol="ETH",
    strategies=[
        EmaRsiStrategy(symbol="ETH", fast=50, slow=200),
        BollingerStrategy(symbol="ETH", window=20),
    ],
    weights=[0.6, 0.4],   # weighted vote
    threshold=0.4,         # minimum weighted vote to enter
)
```

---

## ATR-based stop loss

Pass an absolute stop price in `Allocation.stop_loss`. The engine enforces it.

```python
from strategy.stops import ATRStop

stop = ATRStop(atr_mult_sl=2.0, atr_mult_tp=3.0, atr_period=14)
```

Or set it manually inside `bar()`:

```python
def bar(self, data, idx):
    atr_val = data["atr"].iat[idx]
    price = data["close"].iat[idx]
    sl = price - 2 * atr_val   # for a long
    return Allocation(side=Side.LONG, weight=1.0, stop_loss=sl)
```

---

## Backtesting checklist

```python
from core.models import BacktestConfig
from core.universe import Universe
from backtester.engine import Backtester
from backtester.costs import CompositeCostModel, default_cost_stack
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

universe = Universe(symbols=["ETH"])
universe.add_asset("ETH", ohlcv)

result = Backtester(
    strategy=MyStrategy(symbol="ETH"),
    config=BacktestConfig(initial_capital=100_000.0, taker_fee_bps=5.0, slippage_bps=1.0),
    sizer=FixedNotionalSizer(notional=10_000),
    stop_loss=NopStopLoss(),
    cost_model=CompositeCostModel(default_cost_stack()),
).run(universe=universe, timeframe="1h")

s = result.summary()
print(f"Sharpe={s['sharpe_ratio']:.2f}  MaxDD={s['max_drawdown_pct']:.1f}%  WinRate={s['win_rate_pct']:.1f}%")
result.save("my_strategy_v1")
```

Red flags in the summary:
- Sharpe > 3 with few trades → almost certainly look-ahead bias or data snooping
- Win rate near 100% → check that your exit logic isn't using future prices
- MaxDD = 0% → the strategy never entered, or stops are too tight

---

## Parameter sweep

Use after the backtest looks reasonable, on the **test** split only:

```python
from backtester.stress import ParamSweep

sweep = ParamSweep(
    strategy_cls=MyStrategy,
    param_grid={"window": [10, 20, 50, 100]},
    config=config, cost_model=cost_model, sizer=sizer, stop_loss=stop_loss,
)
df = sweep.run(universe=ttv.test, timeframe="1h")
best = df.best("sharpe_ratio")
print(best)
```

Track how many parameter combinations you try — you need this count for Deflated Sharpe Ratio.

---

## Validation pipeline (the correct order)

```
1. Explore on TRAIN      →  pick a strategy family, check walk-forward consistency
2. Optimise on TEST      →  grid search, pick best params, count n_trials
3. Evaluate on VALIDATE  →  run once, full hypothesis battery, never go back
```

```python
from hypothesis import (
    TrainTestValidateSplit, HypothesisTests, WalkForwardAnalysis,
    PermutationTest, BootstrapCI, DeflatedSharpeRatio, report
)

ttv = TrainTestValidateSplit.by_fractions(universe, train_frac=0.60, test_frac=0.20, embargo_bars=10)

# --- TRAIN ---
train_result = Backtester(strategy=MyStrategy(symbol="ETH", window=20), ...).run(ttv.train, "1h")

wfa = WalkForwardAnalysis(
    strategy_cls=MyStrategy, strategy_params={"window": 20},
    fixed_params={"symbol": "ETH"},
    config=config, cost_model=cost_model, sizer=sizer, stop_loss=stop_loss,
).run(universe=ttv.train, timeframe="1h", n_splits=5)
print(f"WFA consistency: {wfa.consistency_score:.0%}")

# --- TEST ---
sweep = ParamSweep(strategy_cls=MyStrategy, param_grid={"window": [10, 20, 30, 50]}, ...).run(ttv.test, "1h")
best_window = int(sweep.best("sharpe_ratio")["window"])
n_trials = 4

# --- VALIDATE (run once) ---
val = Backtester(strategy=MyStrategy(symbol="ETH", window=best_window), ...).run(ttv.validate, "1h")
print(report(HypothesisTests.run_all(val)))

pt = PermutationTest(metric="sharpe_ratio", n_permutations=2_000).run(val)
dsr = DeflatedSharpeRatio().compute(val, n_trials=n_trials)
ci  = BootstrapCI(n_bootstrap=2_000, ci=0.95).run(val)

print(f"Permutation p={pt.p_value:.4f}")
print(f"DSR={dsr.deflated_sharpe:.3f}  genuine={'yes' if dsr.reject_null else 'no'}")
```

---

## Deploying a strategy live

```python
from core.models import LiveConfig, ExchangeCredentials
from execution import Engine as LiveEngine
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

cred = ExchangeCredentials(exchange="alpaca", api_key="...", api_secret="...", testnet=True)
config = LiveConfig(
    exchange="alpaca", use_testnet=True, exchanges=[cred],
    symbol="SPY",
    bar_interval_s=60,        # bar size in seconds
    warmup_bars=300,          # bars to accumulate before first trade
    max_position_pct=0.10,    # max 10% of equity in one position
    leverage=1.0,
    max_daily_loss_pct=3.0,   # kill switch
    trade_log_csv="trades.csv",
)
engine = LiveEngine(
    strategy=MyStrategy(symbol="SPY", window=best_window),
    config=config,
    sizer=FixedNotionalSizer(notional=10_000),
    stop_loss=NopStopLoss(),
)
engine.start()   # blocks; 'q' + Enter to flatten all positions and stop
```

---

## Anti-patterns to avoid

| Anti-pattern | Problem | Fix |
|---|---|---|
| Using `data.iloc[idx+1]` in `bar()` | Look-ahead bias — uses a future price | Only index `<= idx` |
| Fitting parameters on the full dataset | Overfitting — in-sample metrics are optimistic | Use `TrainTestValidateSplit` |
| Running validate more than once | Contaminates the holdout | Treat validate as a lock box until final report |
| Ignoring transaction costs | Strategies with many trades look great without fees | Always pass `cost_model` |
| Reporting Sharpe without DSR | Raw Sharpe is inflated by the number of parameters you tried | Always use `DeflatedSharpeRatio(n_trials=...)` |
| Sharpe > 3 from a simple rule | Almost certainly a bug (look-ahead, data leak) | Audit `setup_data` and `bar()` very carefully |
