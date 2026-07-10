# Trading Framework — Agent Context

Import this file as a system prompt or context document to give any AI assistant a complete working knowledge of this repository.

---

## What this repository is

A modular, production-ready Python framework for quantitative strategy development, backtesting, statistical validation, and live trading. Supports crypto (Hyperliquid, Binance) and US equities/ETFs (Alpaca). Python 3.10+.

Install: `pip install pandas pyarrow numpy streamlit plotly websockets scipy python-dotenv`

---

## Repository layout

```
src/
├── core/          # Stable contracts — all data types, protocols, universe, parser
├── strategy/      # Strategy framework — base class, indicators, sizers, stops
├── backtester/    # Vectorised backtest engine, cost models, stress tests
├── hypothesis/    # Statistical validation — TTV splits, hypothesis tests, WFA, DSR
├── execution/     # Live trading engines and exchange executors
│   ├── alpaca/
│   ├── hyperliquid/
│   └── binance/
└── data/          # Scrapers, historical downloaders, auxiliary data

app/               # Streamlit dashboard (data explorer, backtester, live trading)
trading/           # Runnable demo scripts
data/              # Collected market data (auto-created by scrapers, parquet)
```

Dependency DAG (no upward or sideways imports):
```
core → strategy / data → backtester / hypothesis / execution
```

---

## Writing a strategy

Subclass `SingleAssetStrategy`. Implement `setup_data` (precompute indicators once) and `bar` (signal per bar).

```python
from core.models import Allocation, Side
from strategy.built_in import SingleAssetStrategy
from strategy.indicators import ema, rsi, atr, bollinger

class MyStrategy(SingleAssetStrategy):
    def __init__(self, symbol: str, slow: int = 200, **kw):
        super().__init__(symbol=symbol, **kw)
        self.slow = slow

    @property
    def params(self) -> dict:
        return {"slow": self.slow}

    def setup_data(self, data, l2=None):
        data["ema"] = ema(data["close"], self.slow)
        data["rsi"] = rsi(data["close"], 14)

    def bar(self, data, idx: int) -> Allocation:
        if idx < self.slow:
            return Allocation()          # not enough bars — stay flat
        if data["close"].iat[idx] > data["ema"].iat[idx]:
            return Allocation(side=Side.LONG, weight=1.0, reason="above EMA")
        return Allocation()              # flat
```

**`Allocation` fields:** `side` (LONG/SHORT/FLAT) · `weight` (0–1) · `confidence` (0–1) · `reason` (str) · `stop_loss` (absolute price or None) · `take_profit` (absolute price or None)

**Indicators available** (`from strategy.indicators import ...`):

| Function | Returns |
|---|---|
| `ema(series, span)` | Exponential moving average |
| `sma(series, window)` | Simple moving average |
| `rsi(series, period=14)` | RSI 0–100 |
| `atr(high, low, close, period=14)` | Average True Range |
| `bollinger(series, window=20, num_std=2.0)` | `(mid, upper, lower)` tuple |
| `vwap_rolling(price, volume, window)` | Rolling VWAP |
| `order_flow_imbalance(bid_vol, ask_vol, window=20)` | OFI |

---

## Loading data

```python
from core.parser import trades_to_ohlcv, l2_to_orderbook, funding_to_snapshots, align_funding_to_ohlcv
from core.universe import Universe

ohlcv = trades_to_ohlcv("data/trades/HYPERLIQUID_PERPETUALS/ETH", timeframe="1h")
l2 = l2_to_orderbook("data/l2/HYPERLIQUID_PERPETUALS/ETH", ohlcv_data=ohlcv)
funding = align_funding_to_ohlcv(funding_to_snapshots("data/funding/HYPERLIQUID_PERPETUALS/ETH"), ohlcv)

universe = Universe(symbols=["ETH"])
universe.add_asset("ETH", ohlcv, l2=l2, funding=funding)
```

Supported timeframes: `1s 2s 5s 10s 15s 30s 1m 2m 3m 5m 10m 15m 30m 1h 2h 4h 6h 8h 12h 1d`

---

## Running a backtest

```python
from core.models import BacktestConfig
from backtester.engine import Backtester
from backtester.costs import CompositeCostModel, default_cost_stack
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

config = BacktestConfig(initial_capital=100_000.0, taker_fee_bps=5.0, slippage_bps=1.0, leverage=1.0)

result = Backtester(
    strategy=MyStrategy(symbol="ETH"),
    config=config,
    sizer=FixedNotionalSizer(notional=10_000),   # fixed $10k per trade
    stop_loss=NopStopLoss(),                      # no stop (enables vectorised fast path)
    cost_model=CompositeCostModel(default_cost_stack()),
).run(universe=universe, timeframe="1h")

print(result.summary())     # Sharpe, Sortino, max DD, win rate, …
result.save("my_run_v1")    # saves to backtest_runs/my_run_v1/
```

**Vectorised fast path** (10–50× faster) activates automatically when stop = `NopStopLoss` AND sizer = `FixedNotionalSizer(notional=N)`.

---

## Position sizing options

| Class | When to use |
|---|---|
| `FixedNotionalSizer(notional=10_000)` | Fixed dollar per trade — also enables vectorised path |
| `FixedNotionalSizer(equity_pct=0.10)` | 10% of equity per trade |
| `FixedFractionalSizer(risk_frac=0.02)` | Risk 2% of equity; uses `Allocation.stop_loss` distance |
| `VolatilityTargetSizer(target_vol=0.15)` | Scale to hit 15% annualised portfolio vol |

---

## Stop-loss options

| Class | Notes |
|---|---|
| `NopStopLoss()` | No stop — required for vectorised fast path |
| `FixedPercentStop(sl_pct=2.0, tp_pct=4.0)` | Fixed % from entry price |
| `ATRStop(atr_mult_sl=2.0, atr_mult_tp=3.0, atr_period=14)` | ATR-based dynamic SL/TP |
| `TrailingStop(trail_pct=1.5)` | Trails high-water mark |

---

## Statistical validation (Train → Test → Validate)

**Never optimise parameters on the same data you evaluate performance on.**

```python
from hypothesis import (
    TrainTestValidateSplit, HypothesisTests, WalkForwardAnalysis,
    PermutationTest, BootstrapCI, DeflatedSharpeRatio, report
)
from backtester.stress import ParamSweep

# 1. Split universe into three periods (60/20/20 with 10-bar embargo)
ttv = TrainTestValidateSplit.by_fractions(universe, train_frac=0.60, test_frac=0.20, embargo_bars=10)

# 2. TRAIN — develop strategy, check consistency across sub-periods
wfa = WalkForwardAnalysis(strategy_cls=MyStrategy, strategy_params={"slow": 200},
    fixed_params={"symbol": "ETH"}, config=config, cost_model=cost_model, sizer=sizer, stop_loss=stop_loss
).run(universe=ttv.train, timeframe="1h", n_splits=5, split_method="expanding")
print(f"Consistency: {wfa.consistency_score:.0%}   IS/OOS: {wfa.efficiency_ratio:.2f}")

# 3. TEST — optimise parameters, count trials
sweep = ParamSweep(strategy_cls=MyStrategy, param_grid={"slow": [100, 150, 200, 250]},
    config=config, cost_model=cost_model, sizer=sizer, stop_loss=stop_loss
).run(universe=ttv.test, timeframe="1h")
best = sweep.best("sharpe_ratio")
n_trials = 4   # number of parameter combinations tried

# 4. VALIDATE — run once, never look back
val = Backtester(strategy=MyStrategy(symbol="ETH", slow=int(best["slow"])), ...).run(universe=ttv.validate, timeframe="1h")

# 5. Full statistical report
print(report(HypothesisTests.run_all(val)))

# Permutation test — Sharpe vs random trade ordering
pt = PermutationTest(metric="sharpe_ratio", n_permutations=2_000).run(val)
print(f"p={pt.p_value:.4f}  {'Significant' if pt.reject_null else 'Not significant'}")

# Deflated Sharpe — corrects for number of parameter combos tested
dsr = DeflatedSharpeRatio().compute(val, n_trials=n_trials)
print(f"DSR={dsr.deflated_sharpe:.3f}  {'Genuine edge' if dsr.reject_null else 'Likely overfit'}")

# Bootstrap confidence intervals
cis = BootstrapCI(n_bootstrap=2_000, ci=0.95).run(val)
```

---

## Stress testing

```python
from backtester.stress import MonteCarloStress, ParamSweep, RegimeStressTest

# Monte Carlo — distribution of outcomes from trade resampling
mc = MonteCarloStress(n_simulations=1_000, method="bootstrap").run(result)
print(mc.meta)   # median_return, 5th_pctl_return, 95th_pctl_return, median_max_dd

# Regime stress — how does the strategy behave in high-vol vs low-vol regimes?
rst = RegimeStressTest(regime_fn=RegimeStressTest.trend_regime, config=config, cost_model=cost_model)
regime_df = rst.run(strategy=my_strategy, universe=universe).summary
```

---

## Live trading

Credentials go in `.env`:

```env
ALP_PAPER_KEY=your_alpaca_paper_key
ALP_PAPER_SECRET=your_alpaca_paper_secret
```

```python
from core.models import LiveConfig, ExchangeCredentials
from execution import Engine as LiveEngine
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

cred = ExchangeCredentials(exchange="alpaca", api_key="...", api_secret="...", testnet=True)
config = LiveConfig(
    exchange="alpaca", use_testnet=True, exchanges=[cred],
    symbol="SPY", bar_interval_s=60, warmup_bars=300,
    max_position_pct=0.10, leverage=1.0, max_daily_loss_pct=3.0,
    trade_log_csv="trades.csv",
)
engine = LiveEngine(strategy=MyStrategy(symbol="SPY"), config=config,
    sizer=FixedNotionalSizer(notional=10_000), stop_loss=NopStopLoss())
engine.start()   # press 'q' + Enter to flatten all and stop
```

Supported exchanges: `"alpaca"` · `"hyperliquid"` · `"binance"`

---

## Data collection

```bash
# Hyperliquid (crypto) — all streams
python -m src.data.feeds.hyperliquid --coin ETH --mode all

# Binance USD-M futures
python -m src.data.feeds.binance --coin ETHUSDT --market futures --streams trades l2 funding

# Alpaca (US equities)
python -m src.data.feeds.alpaca --symbol SPY --timeframe 1Min
```

Data is stored as Parquet in `data/trades/`, `data/l2/`, `data/funding/`.

---

## Streamlit dashboard

```bash
streamlit run app/main.py
```

Three pages: **Data Explorer** (OHLCV + L2 + indicators) · **Backtester** (run and sweep strategies) · **Live Trading** (deploy and monitor)

---

## Key data types (`from core.models import ...`)

| Type | Description |
|---|---|
| `Side` | `LONG / SHORT / FLAT` enum |
| `Allocation` | Signal output (side, weight, confidence, reason, stop_loss, take_profit) |
| `Position` | Current open position |
| `Trade` | Closed trade record with PnL, fees, slippage |
| `BacktestConfig` | Fees, slippage, leverage, capital |
| `LiveConfig` | Full live trading configuration |
| `ExchangeCredentials` | API key, secret, testnet flag |
| `OrderBookSnapshot` | L2 snapshot with `.mid`, `.spread`, `.vwap_fill_price()` |
| `FundingSnapshot` | Funding rate, rate_annualized, mark_price, oracle_price |
