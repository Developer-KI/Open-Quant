from __future__ import annotations

import pandas as pd
from dotenv import load_dotenv, dotenv_values

from core.models import Allocation, BacktestConfig, Side
from core.universe import Universe
from testing.backtester.engine import Backtester
from testing.backtester.costs import CompositeCostModel, default_cost_stack
from testing.backtester.stress import MonteCarloStress, ParamSweep, RegimeStressTest
from testing.hypothesis import (
    HypothesisTests,
    PermutationTest,
    BootstrapCI,
    WalkForwardAnalysis,
    DeflatedSharpeRatio,
    TrainTestValidateSplit,
    report as hypothesis_report,
)
from strategy.base import Strategy, StrategyContext, PortfolioTarget
from strategy.built_in import CompositeStrategy, SingleAssetStrategy
from strategy.indicators import bollinger, ema, rsi
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss
from testing.backtester.costs import ExchangeFeeCost, FixedSlippageCost


# ═══════════════════════════════════════════════════════════════════════════
#  Data fetching
# ═══════════════════════════════════════════════════════════════════════════

def fetch_lse_bars(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1s",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from London Strategic Edge for a single symbol.

    Parameters
    ----------
    symbol    : ticker exactly as in the LSE catalog, e.g. "AAPL", "BTC/USD"
    start     : ISO date string, e.g. "2005-01-01"
    end       : ISO date string, e.g. "2026-01-01"
    timeframe : 1s 5s 15s 30s 1m 3m 5m 15m 30m 1h 4h 1d 1w 1mo (default 1d)
    api_key   : LSE key; falls back to LSE_DATA env var

    Returns
    -------
    DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
    """
    try:
        from lse import LSE
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: lse-data. "
            "Install with: pip install 'lse-data[frames]'"
        ) from exc

    load_dotenv()
    _env = dotenv_values()
    key = api_key or _env.get("LSE_DATA", "")
    if not key:
        raise ValueError(
            "LSE API key required. Set LSE_DATA in your .env file "
            "or pass api_key directly."
        )

    client = LSE(api_key=key)
    rows = client.candles(symbol, timeframe, start=start, end=end)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Forex candles carry no volume — fill with zero so downstream code is uniform
    if "volume" not in df.columns:
        df["volume"] = 0

    return df[["open", "high", "low", "close", "volume"]]


# ═══════════════════════════════════════════════════════════════════════════
#  Strategies  (identical to alpaca_backtest_demo)
# ═══════════════════════════════════════════════════════════════════════════

class EmaRsiStrategy(SingleAssetStrategy):
    """
    Long-only trend strategy filtered by RSI and volatility regime.

    Entry:  close > slow EMA AND RSI < rsi_overbought AND vol regime = medium
    Exit:   close <= slow EMA OR RSI >= rsi_overbought OR vol regime != medium
    """

    def __init__(
        self,
        symbol: str,
        slow: int = 200,
        rsi_period: int = 14,
        rsi_overbought: float = 80.0,
        vol_window: int = 20,
        vol_q_low: float = 0.33,
        vol_q_high: float = 0.66,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.vol_window = vol_window
        self.vol_q_low = vol_q_low
        self.vol_q_high = vol_q_high

    @property
    def params(self) -> dict:
        return {
            "slow": self.slow,
            "rsi_period": self.rsi_period,
            "rsi_overbought": self.rsi_overbought,
            "vol_q_low": self.vol_q_low,
            "vol_q_high": self.vol_q_high,
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        data["ema_slow"] = ema(data["close"], self.slow)
        data["rsi"] = rsi(data["close"], self.rsi_period)
        rv = data["close"].pct_change().rolling(self.vol_window).std()
        data["_rv"]     = rv
        data["_rv_q_lo"] = rv.expanding(min_periods=self.vol_window).quantile(self.vol_q_low)
        data["_rv_q_hi"] = rv.expanding(min_periods=self.vol_window).quantile(self.vol_q_high)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.slow:
            return Allocation()

        close   = data["close"].iat[idx]
        es      = data["ema_slow"].iat[idx]
        rsi_val = data["rsi"].iat[idx]
        rv      = data["_rv"].iat[idx]
        q_lo    = data["_rv_q_lo"].iat[idx]
        q_hi    = data["_rv_q_hi"].iat[idx]

        if any(v != v for v in (close, es, rsi_val)):
            return Allocation()

        if rv != rv or q_lo != q_lo or q_hi != q_hi:
            return Allocation(reason="vol_warmup")

        if rv <= q_lo:
            return Allocation(reason=f"vol_low rv={rv:.4f} q_lo={q_lo:.4f}")
        if rv >= q_hi:
            return Allocation(reason=f"vol_high rv={rv:.4f} q_hi={q_hi:.4f}")

        above_ema = close > es
        rsi_ok    = rsi_val < self.rsi_overbought

        if above_ema and rsi_ok:
            return Allocation(
                side=Side.LONG,
                weight=1.0,
                confidence=1.0,
                reason=f"above EMA{self.slow} | RSI={rsi_val:.0f} | vol=medium",
            )

        return Allocation(reason=f"no signal | above_ema={above_ema} | RSI={rsi_val:.0f} | vol=medium")


class BollingerMeanReversionStrategy(SingleAssetStrategy):
    """
    Long/short Bollinger Band mean reversion.

    Long entry:  close crosses below the lower band (oversold)
    Long exit:   close crosses back above the midline
    Short entry: close crosses above the upper band (overbought)
    Short exit:  close crosses back below the midline
    """

    def __init__(
        self,
        symbol: str,
        window: int = 20,
        num_std: float = 2.0,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.window = window
        self.num_std = num_std
        self._current_side: Side = Side.FLAT

    @property
    def params(self) -> dict:
        return {"window": self.window, "num_std": self.num_std}

    def setup(self, universe):
        self._current_side = Side.FLAT
        super().setup(universe)

    def setup_data(self, data: pd.DataFrame, l2=None):
        data["bb_mid"], data["bb_upper"], data["bb_lower"] = bollinger(
            data["close"], self.window, self.num_std
        )

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.window:
            return Allocation()

        close = data["close"].iat[idx]
        mid   = data["bb_mid"].iat[idx]
        upper = data["bb_upper"].iat[idx]
        lower = data["bb_lower"].iat[idx]

        if any(v != v for v in (close, mid, upper, lower)):
            return Allocation()

        # New entries take priority over holding/exit logic
        if close < lower:
            self._current_side = Side.LONG
            return Allocation(
                side=Side.LONG,
                weight=1.0,
                confidence=1.0,
                reason=f"BB oversold | close={close:.2f} < lower={lower:.2f}",
            )
        if close > upper:
            self._current_side = Side.SHORT
            return Allocation(
                side=Side.SHORT,
                weight=1.0,
                confidence=1.0,
                reason=f"BB overbought | close={close:.2f} > upper={upper:.2f}",
            )

        # Exit at midline (hold until then)
        if self._current_side == Side.LONG:
            if close >= mid:
                self._current_side = Side.FLAT
                return Allocation(reason=f"BB long exit at midline | close={close:.2f} mid={mid:.2f}")
            return Allocation(
                side=Side.LONG, weight=1.0, confidence=1.0,
                reason=f"BB holding long | close={close:.2f} mid={mid:.2f}",
            )
        if self._current_side == Side.SHORT:
            if close <= mid:
                self._current_side = Side.FLAT
                return Allocation(reason=f"BB short exit at midline | close={close:.2f} mid={mid:.2f}")
            return Allocation(
                side=Side.SHORT, weight=1.0, confidence=1.0,
                reason=f"BB holding short | close={close:.2f} mid={mid:.2f}",
            )

        return Allocation(reason=f"BB no signal | close={close:.2f} mid={mid:.2f}")


class VolFilteredCompositeStrategy(CompositeStrategy):
    """
    CompositeStrategy that sits out when volatility is elevated.

    Rolling vol is compared to a longer rolling median; if current vol exceeds
    vol_multiplier × median the bar is skipped.
    """

    def __init__(
        self,
        vol_window: int = 20,
        vol_multiplier: float = 1.5,
        **kw,
    ):
        super().__init__(**kw)
        self.vol_window = vol_window
        self.vol_multiplier = vol_multiplier

    @property
    def params(self) -> dict:
        return {
            **super().params,
            "vol_window": self.vol_window,
            "vol_multiplier": self.vol_multiplier,
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        super().setup_data(data, l2)
        rv = data["close"].pct_change().rolling(self.vol_window).std()
        data["_rv"] = rv
        data["_rv_median"] = rv.rolling(self.vol_window * 3).median()

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.vol_window * 4:
            return Allocation()

        rv     = data["_rv"].iat[idx]
        rv_med = data["_rv_median"].iat[idx]

        if rv != rv or rv_med != rv_med or rv_med == 0:
            return Allocation()

        if rv > self.vol_multiplier * rv_med:
            return Allocation(reason=f"vol filter | rv={rv:.4f} > {self.vol_multiplier}x med={rv_med:.4f}")

        return super().bar(data, idx)


class BuyAndHoldStrategy(SingleAssetStrategy):
    """Always long at full weight from bar 0."""

    def bar(self, _data: pd.DataFrame, _idx: int) -> Allocation:
        return Allocation(side=Side.LONG, weight=1.0, reason="buy and hold")


class MultiExchangeMomentumStrategy(Strategy):
    """
    Two independent signals routed to two separate exchanges.

    trend_exchange   — EMA trend-following (long-only): go long when close > EMA
    reversal_exchange — Bollinger mean-reversion (long + short): fade extreme moves

    Demonstrates multi-exchange Strategy usage:
      • setup() receives a dict[exchange, Universe] and precomputes per-exchange indicators
      • generate() returns PortfolioTarget.exchange_allocations keyed by (exchange, symbol)
    """

    def __init__(
        self,
        symbol: str,
        trend_exchange: str,
        reversal_exchange: str,
        ema_slow: int = 200,
        bb_window: int = 20,
        bb_std: float = 2.0,
        **kw,
    ):
        super().__init__(**kw)
        self.symbol = symbol
        self.trend_exchange = trend_exchange
        self.reversal_exchange = reversal_exchange
        self.ema_slow = ema_slow
        self.bb_window = bb_window
        self.bb_std = bb_std
        self._ind: dict[str, pd.DataFrame] = {}

    @property
    def params(self) -> dict:
        return {"ema_slow": self.ema_slow, "bb_window": self.bb_window, "bb_std": self.bb_std}

    def setup(self, universes):
        if isinstance(universes, Universe):
            universes = {self.trend_exchange: universes}
        self._ind = {}
        for ex, uni in universes.items():
            if self.symbol not in uni.symbols:
                continue
            df = uni.ohlcv(self.symbol).copy()
            df["ema_slow"] = ema(df["close"], self.ema_slow)
            df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger(
                df["close"], self.bb_window, self.bb_std
            )
            self._ind[ex] = df

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget(timestamp=ctx.timestamp)
        i = ctx.bar_idx
        if i < max(self.ema_slow, self.bb_window):
            return target

        # Trend exchange — EMA trend-following (long-only)
        df = self._ind.get(self.trend_exchange)
        if df is not None and i < len(df):
            close   = df["close"].iat[i]
            ema_val = df["ema_slow"].iat[i]
            if not (pd.isna(close) or pd.isna(ema_val)):
                if close > ema_val:
                    target[(self.trend_exchange, self.symbol)] = Allocation(
                        side=Side.LONG, weight=1.0, confidence=1.0,
                        reason=f"above EMA{self.ema_slow}",
                    )
                else:
                    target[(self.trend_exchange, self.symbol)] = Allocation(
                        side=Side.FLAT, reason="below EMA",
                    )

        # Reversal exchange — Bollinger mean-reversion (long + short)
        df = self._ind.get(self.reversal_exchange)
        if df is not None and i < len(df):
            close    = df["close"].iat[i]
            bb_upper = df["bb_upper"].iat[i]
            bb_lower = df["bb_lower"].iat[i]
            if not (pd.isna(close) or pd.isna(bb_upper)):
                if close < bb_lower:
                    target[(self.reversal_exchange, self.symbol)] = Allocation(
                        side=Side.LONG, weight=1.0, confidence=1.0,
                        reason=f"BB oversold | lower={bb_lower:.2f}",
                    )
                elif close > bb_upper:
                    target[(self.reversal_exchange, self.symbol)] = Allocation(
                        side=Side.SHORT, weight=1.0, confidence=1.0,
                        reason=f"BB overbought | upper={bb_upper:.2f}",
                    )
                else:
                    target[(self.reversal_exchange, self.symbol)] = Allocation(
                        side=Side.FLAT, reason="BB neutral",
                    )

        return target


# ═══════════════════════════════════════════════════════════════════════════
#  Demo runner
# ═══════════════════════════════════════════════════════════════════════════

_METRICS = [
    ("Total return %",    "total_return_pct"),
    ("Ann. return %",     "annualised_return_pct"),
    ("Ann. volatility %", "annualised_volatility_pct"),
    ("Sharpe",            "sharpe_ratio"),
    ("Sortino",           "sortino_ratio"),
    ("Max drawdown %",    "max_drawdown_pct"),
    ("Num trades",        "num_trades"),
    ("Win rate %",        "win_rate_pct"),
]


def _print_metrics_table(summaries: list[tuple[str, dict]]) -> None:
    col = 24
    headers = "".join(f"{label:>12}" for label, _ in summaries)
    print(f"\n{'Metric':<{col}} {headers}")
    print("-" * (col + 12 * len(summaries) + 1))
    for label, key in _METRICS:
        row = "".join(f"{s[key]:>12}" for _, s in summaries)
        print(f"{label:<{col}} {row}")


def _demo_multi_exchange(symbol: str, data: pd.DataFrame, timeframe: str) -> None:
    """
    Multi-exchange backtest section.

    Two virtual exchanges are created from the same OHLCV data:
      'nyse'  — low-cost venue, EMA trend-following (2 bps maker / 3 bps taker + 1 bps slip)
      'bats'  — higher-cost venue, Bollinger mean-reversion (10 bps maker / 15 bps taker + 5 bps slip)

    Capital is split $50k / $50k across the two exchanges.
    """
    print("\n\n" + "═" * 70)
    print("  MULTI-EXCHANGE DEMO")
    print("  'nyse': EMA trend-following (long-only, low-cost)")
    print("  'bats': Bollinger mean-reversion (long+short, higher-cost)")
    print("═" * 70)

    u_nyse = Universe(symbols=[symbol])
    u_nyse.add_asset(symbol, data)

    u_bats = Universe(symbols=[symbol])
    u_bats.add_asset(symbol, data)

    strategy = MultiExchangeMomentumStrategy(
        symbol=symbol,
        trend_exchange="nyse",
        reversal_exchange="bats",
    )

    nyse_cost = CompositeCostModel([
        ExchangeFeeCost(maker_bps=2, taker_bps=3),
        FixedSlippageCost(slippage_bps=1),
    ])
    bats_cost = CompositeCostModel([
        ExchangeFeeCost(maker_bps=10, taker_bps=15),
        FixedSlippageCost(slippage_bps=5),
    ])

    config = BacktestConfig(initial_capital=100_000.0, max_position_pct=1.0, leverage=1.0)
    sizer  = FixedNotionalSizer(notional=50_000)

    bt = Backtester(
        strategy=strategy,
        config=config,
        sizer=sizer,
        stop_loss=NopStopLoss(),
        exchange_costs={"nyse": nyse_cost, "bats": bats_cost},
        capital_by_exchange={"nyse": 50_000.0, "bats": 50_000.0},
    )
    result = bt.run(
        universes={"nyse": u_nyse, "bats": u_bats},
        timeframe=timeframe,
    )

    s = result.summary()
    print(f"\nCombined result  ({s['num_trades']} trades  |  run in {s['run_time_s']:.2f}s)")
    for label, key in _METRICS:
        print(f"  {label:<24} {s[key]:>10}")

    if result.equity_curves_by_exchange:
        print("\nPer-exchange breakdown:")
        print(f"  {'Exchange':<16} {'Start $':>12} {'End $':>12} {'Return %':>10} {'Trades':>8} {'Fees':>10}")
        print("  " + "-" * 70)
        for ex, curve in result.equity_curves_by_exchange.items():
            ex_trades = [t for t in result.trades if t.meta.get("exchange") == ex]
            total_fees = sum(t.fees for t in ex_trades)
            ret_pct = (curve.iloc[-1] / curve.iloc[0] - 1) * 100
            print(
                f"  {ex:<16} {curve.iloc[0]:>12,.0f} {curve.iloc[-1]:>12,.0f}"
                f" {ret_pct:>9.2f}% {len(ex_trades):>8} {total_fees:>10,.2f}"
            )

    run_dir = result.save("multi_exchange_demo")
    print(f"\n  Results saved to: {run_dir}")
    print(f"  Files: log.json  trades.csv  equity_curve.png  equity_curves_by_exchange.csv")


def demo(
    symbol: str = "NVDA",
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    timeframe: str = "1d",
):
    load_dotenv()
    _env = dotenv_values()
    api_key = _env.get("LSE_DATA", "")

    print(f"\nFetching {symbol} {timeframe} bars from LSE ({start} → {end})...")
    data = fetch_lse_bars(symbol, start=start, end=end, timeframe=timeframe, api_key=api_key)
    print(f"  {len(data)} bars loaded  |  {data.index[0].date()} → {data.index[-1].date()}")

    universe = Universe(symbols=[symbol])
    universe.add_asset(symbol, data)

    # ── Train / Test / Validate split (60 / 20 / 20, 10-bar embargo) ──────
    ttv = TrainTestValidateSplit.by_fractions(
        universe, train_frac=0.60, test_frac=0.20, embargo_bars=10
    )
    print(f"\n{ttv}")

    config     = BacktestConfig(initial_capital=100_000.0, max_position_pct=1.0, leverage=1.0)
    cost_model = CompositeCostModel(default_cost_stack())
    sizer      = FixedNotionalSizer(notional=100_000)
    stoploss   = NopStopLoss()

    def run_on(strategy, univ):
        return Backtester(
            strategy=strategy, config=config,
            sizer=sizer, stop_loss=stoploss, cost_model=cost_model,
        ).run(universe=univ, timeframe=timeframe)

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 1 — TRAIN  (strategy design, IS exploration)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 1 — TRAIN  (strategy design / IS exploration)")
    print(f"  {ttv.train_start.date()} → {ttv.train_end.date()}")
    print("═" * 70)

    print("\nRunning candidate strategies on TRAIN data...")
    train_ema  = run_on(EmaRsiStrategy(symbol=symbol, slow=200), ttv.train)
    train_bb   = run_on(BollingerMeanReversionStrategy(symbol=symbol), ttv.train)
    train_comp = run_on(
        VolFilteredCompositeStrategy(
            symbol=symbol,
            strategies=[
                EmaRsiStrategy(symbol=symbol, slow=200),
                BollingerMeanReversionStrategy(symbol=symbol, window=20, num_std=2.0),
            ],
            weights=[1, 0], threshold=0.4,
        ),
        ttv.train,
    )
    train_bah = run_on(BuyAndHoldStrategy(symbol=symbol), ttv.train)

    _print_metrics_table([
        ("EMA/RSI",    train_ema.summary()),
        ("Mean Rev",   train_bb.summary()),
        ("Composite",  train_comp.summary()),
        ("Buy & Hold", train_bah.summary()),
    ])

    print("\n--- Walk-Forward Analysis on TRAIN data (5 expanding folds) ---")
    wfa = WalkForwardAnalysis(
        strategy_cls=EmaRsiStrategy,
        strategy_params={"slow": 200},
        fixed_params={"symbol": symbol},
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=stoploss,
    )
    wf = wfa.run(universe=ttv.train, timeframe=timeframe, n_splits=5, split_method="expanding")
    print(f"  Consistency score : {wf.consistency_score:.0%}  (fraction of IS sub-folds profitable)")
    print(f"  IS/OOS efficiency : {wf.efficiency_ratio:.2f}  (sub-OOS Sharpe / IS Sharpe)")
    tbl = wf.summary_table()
    for fold, row in tbl.iterrows():
        is_sr   = row.get("is_sharpe_ratio",      float("nan"))
        oos_sr  = row.get("oos_sharpe_ratio",     float("nan"))
        is_ret  = row.get("is_total_return_pct",  float("nan"))
        oos_ret = row.get("oos_total_return_pct", float("nan"))
        print(
            f"    Fold {fold}  "
            f"IS  ret={is_ret:>7.2f}%  SR={is_sr:>6.3f}  │  "
            f"OOS ret={oos_ret:>7.2f}%  SR={oos_sr:>6.3f}"
        )

    print("\n  Decision: proceeding with EMA/RSI + Bollinger composite candidate.")

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 2 — TEST  (parameter optimisation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 2 — TEST  (parameter optimisation)")
    print(f"  {ttv.test_start.date()} → {ttv.test_end.date()}")
    print("═" * 70)

    ema_grid = {"slow": [100, 150, 200], "rsi_overbought": [65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0]}
    bb_grid  = {"window": [10, 15, 20, 30], "num_std": [1.5, 2.0, 2.5]}
    n_ema_trials = 3 * 7
    n_bb_trials  = 4 * 3
    n_trials     = n_ema_trials + n_bb_trials

    print(f"\nSweeping EmaRsiStrategy ({n_ema_trials} combos) on TEST data...")
    ema_sweep = ParamSweep(
        strategy_cls=EmaRsiStrategy,
        param_grid=ema_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=stoploss,
    ).run(universe=ttv.test, timeframe=timeframe)
    best_ema_row    = ema_sweep.best("sharpe_ratio")
    best_ema_params = {k: best_ema_row[k] for k in ema_grid}
    best_ema_params["slow"] = int(best_ema_params["slow"])
    print(f"  Best EMA params : {best_ema_params}  →  SR={best_ema_row['sharpe_ratio']:.3f}")

    print(f"\nSweeping BollingerMeanReversionStrategy ({n_bb_trials} combos) on TEST data...")
    bb_sweep = ParamSweep(
        strategy_cls=BollingerMeanReversionStrategy,
        param_grid=bb_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=stoploss,
    ).run(universe=ttv.test, timeframe=timeframe)
    best_bb_row    = bb_sweep.best("sharpe_ratio")
    best_bb_params = {k: best_bb_row[k] for k in bb_grid}
    best_bb_params["window"] = int(best_bb_params["window"])
    print(f"  Best BB params  : {best_bb_params}  →  SR={best_bb_row['sharpe_ratio']:.3f}")

    print(f"\n  Total trials tracked for DSR: {n_trials}")

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 3 — VALIDATE  (blind final evaluation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 3 — VALIDATE  (blind final evaluation)")
    print(f"  {ttv.validate_start.date()} → {ttv.validate_end.date()}")
    print("═" * 70)

    best_ema_strat = EmaRsiStrategy(symbol=symbol, **best_ema_params)
    best_bb_strat  = BollingerMeanReversionStrategy(symbol=symbol, **best_bb_params)
    final_composite = VolFilteredCompositeStrategy(
        symbol=symbol,
        strategies=[best_ema_strat, best_bb_strat],
        weights=[0.5, 0.5], threshold=0.4, vol_window=20, vol_multiplier=1.5,
    )

    print("\nRunning final strategies on VALIDATE data...")
    val_comp = run_on(final_composite, ttv.validate)
    val_bah  = run_on(BuyAndHoldStrategy(symbol=symbol), ttv.validate)

    run_dir = val_comp.save("ttv_validate_lse")
    print(f"  Result saved to: {run_dir}")

    _print_metrics_table([
        ("Composite (tuned)", val_comp.summary()),
        ("Buy & Hold",        val_bah.summary()),
    ])

    # ── Hypothesis test battery ───────────────────────────────────────────
    print("\n\n=== Hypothesis Tests — VALIDATE (final composite) ===")
    tests = HypothesisTests.run_all(val_comp)
    print(hypothesis_report(tests))

    # ── Strategy comparison ───────────────────────────────────────────────
    print("\n=== Strategy Comparison: Composite vs Buy & Hold (VALIDATE) ===")
    for metric in ("sharpe_ratio", "total_return_pct"):
        t = HypothesisTests.compare(val_comp, val_bah, metric=metric)
        verdict = "✓ Composite wins" if t.reject_null else "✗ No significant edge"
        print(f"  {metric:<22} p={t.p_value:.4f}  {verdict}")

    # ── Permutation test ──────────────────────────────────────────────────
    print("\n=== Permutation Test — VALIDATE composite (sharpe_ratio, 2000 perms) ===")
    pt = PermutationTest(metric="sharpe_ratio", n_permutations=2_000)
    pt_result = pt.run(val_comp)
    print(
        f"  Observed SR={pt_result.statistic:.3f}  "
        f"null median={pt_result.meta['null_mean']:.3f}  "
        f"p={pt_result.p_value:.4f}  "
        f"{'✓ Significant' if pt_result.reject_null else '✗ Not significant'}"
    )

    # ── Bootstrap confidence intervals ────────────────────────────────────
    print("\n=== Bootstrap 95% CIs — VALIDATE composite (2000 samples) ===")
    ci = BootstrapCI(n_bootstrap=2_000, ci=0.95)
    cis = ci.run(val_comp)
    ci_col = 22
    print(f"  {'Metric':<{ci_col}} {'Observed':>10} {'Lower 95%':>10} {'Upper 95%':>10}")
    print("  " + "-" * (ci_col + 32))
    for metric, vals in cis.items():
        print(
            f"  {metric:<{ci_col}}"
            f" {vals['observed']:>10.3f}"
            f" {vals['lower']:>10.3f}"
            f" {vals['upper']:>10.3f}"
        )

    # ── Deflated Sharpe Ratio (accounts for all test-phase trials) ────────
    print(f"\n=== Deflated Sharpe Ratio — VALIDATE (n_trials={n_trials}) ===")
    dsr = DeflatedSharpeRatio()
    d = dsr.compute(val_comp, n_trials=n_trials)
    verdict = "✓ Genuine edge" if d.reject_null else "✗ Likely overfit"
    print(
        f"  Composite (tuned)  SR={d.observed_sharpe:.4f}  "
        f"deflated_SR={d.deflated_sharpe:.4f}  "
        f"p={d.p_value:.4f}  {verdict}"
    )

    # ── Monte Carlo simulations ───────────────────────────────────────────
    print("\n=== Monte Carlo Simulations — VALIDATE (1000 bootstrap runs) ===")
    mc = MonteCarloStress(n_simulations=1_000, method="bootstrap")
    mc_res = mc.run(val_comp)
    m = mc_res.meta
    print(
        f"  Median ret={m['median_return']:.2f}%  "
        f"5th={m['5th_pctl_return']:.2f}%  "
        f"95th={m['95th_pctl_return']:.2f}%  "
        f"Median DD={m['median_max_dd']:.2f}%"
    )

    # ── Regime stress tests on validate universe ──────────────────────────
    print("\n=== Regime Stress Tests — VALIDATE ===")
    stress_cfg  = BacktestConfig(initial_capital=100_000.0, max_position_pct=1.0, leverage=1.0)
    stress_cols = ["regime", "n_bars", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]
    for regime_label, regime_fn in [
        ("Volatility", None),
        ("Trend",      RegimeStressTest.trend_regime),
    ]:
        rst = RegimeStressTest(regime_fn=regime_fn, config=stress_cfg, cost_model=cost_model)
        sr  = rst.run(strategy=final_composite, universe=ttv.validate)
        df  = sr.summary.sort_values("regime")[stress_cols].reset_index(drop=True)
        print(f"\n  {regime_label} regimes:")
        print(df.to_string(index=False))


    _demo_multi_exchange(symbol, data, timeframe)


if __name__ == "__main__":
    demo()
