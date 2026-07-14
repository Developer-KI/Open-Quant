"""
Test: buy-and-hold return is consistent across timeframes.

For a zero-cost buy-and-hold the backtester must report:
    total_return_pct == (last_close / first_close - 1) * 100

for every resampling of the same underlying daily price series.

Run:
    python trading/test_bah_timeframes.py
"""

from __future__ import annotations

import sys
import os

# Allow 'src/' imports when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from core.models import Allocation, BacktestConfig, Side
from core.universe import Universe
from testing.backtester.engine import Backtester
from testing.backtester.costs import NullCostModel
from strategy.base import Strategy, StrategyContext, PortfolioTarget
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss


# ── Minimal buy-and-hold strategy ────────────────────────────────────────────

class BuyAndHold(Strategy):
    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    @property
    def params(self) -> dict:
        return {}

    def setup(self, universe):
        pass

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget(timestamp=ctx.timestamp)
        target[self.symbol] = Allocation(side=Side.LONG, weight=1.0, reason="bah")
        return target


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_daily(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Daily OHLCV with a random-walk close, OHLC consistent."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, freq="B")  # business days only
    n = len(dates)

    returns = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    noise = rng.uniform(0.002, 0.008, n)
    high = np.maximum(open_, close) * (1 + noise)
    low  = np.minimum(open_, close) * (1 - noise)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000},
        index=pd.DatetimeIndex(dates, tz="UTC"),
    )
    df.index.name = "timestamp"
    return df


def _resample(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily OHLCV to the given rule (e.g. 'W', 'ME')."""
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    resampled = daily.resample(rule).agg(agg).dropna()
    return resampled


# ── Test runner ───────────────────────────────────────────────────────────────

TOLERANCE = 1e-3  # percentage points — summary() rounds to 4 dp, so max error ~5e-5


def _run_bah(symbol: str, df: pd.DataFrame, tf_label: str) -> dict:
    universe = Universe(symbols=[symbol])
    universe.add_asset(symbol, df)

    bt = Backtester(
        strategy=BuyAndHold(symbol),
        config=BacktestConfig(initial_capital=100_000.0, max_position_pct=1.0, leverage=1.0),
        cost_model=NullCostModel(),
        sizer=FixedNotionalSizer(notional=100_000.0),
        stop_loss=NopStopLoss(),
    )
    result = bt.run(universe=universe, timeframe=tf_label)
    s = result.summary()

    expected_return_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    return {
        "timeframe":          tf_label,
        "bars":               len(df),
        "first_close":        round(df["close"].iloc[0], 6),
        "last_close":         round(df["close"].iloc[-1], 6),
        "expected_return_pct": round(expected_return_pct, 6),
        "reported_return_pct": round(s["total_return_pct"], 6),
        "delta_pct":          round(s["total_return_pct"] - expected_return_pct, 8),
        "num_trades":         s["num_trades"],
    }


def test_bah_timeframe_consistency():
    symbol = "TEST"
    daily   = _make_daily("2018-01-01", "2023-12-31")
    weekly  = _resample(daily, "W")
    monthly = _resample(daily, "ME")

    print(f"\n{'-'*70}")
    print("  Buy-and-Hold: total_return must equal (last_close/first_close - 1)")
    print("  for each resampling of the same underlying price series.")
    print(f"{'-'*70}\n")

    timeframes = [
        ("1d", daily),
        ("1w", weekly),
        ("1mo", monthly),
    ]

    results = []
    for tf_label, df in timeframes:
        r = _run_bah(symbol, df, tf_label)
        results.append(r)

    # Print table
    col = 10
    header = (
        f"{'TF':<6} {'Bars':>6} {'First close':>12} {'Last close':>12} "
        f"{'Expected %':>12} {'Reported %':>12} {'Delta':>12} {'Trades':>7}"
    )
    print(header)
    print("-" * len(header))

    all_pass = True
    for r in results:
        ok = abs(r["delta_pct"]) < TOLERANCE
        flag = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(
            f"{r['timeframe']:<6} {r['bars']:>6} {r['first_close']:>12.4f} "
            f"{r['last_close']:>12.4f} {r['expected_return_pct']:>12.4f} "
            f"{r['reported_return_pct']:>12.4f} {r['delta_pct']:>12.2e}  [{flag}]"
        )

    print()

    # Additional check: all three timeframes share the same last close
    # (since weekly/monthly closes are derived from daily closes, the last
    # weekly/monthly close equals the last daily close).
    last_closes = {r["timeframe"]: r["last_close"] for r in results}
    for tf, lc in last_closes.items():
        match = abs(lc - last_closes["1d"]) < 0.01
        if not match:
            print(f"  WARNING: last close for {tf} ({lc}) differs from 1d ({last_closes['1d']})")

    # Assert per-timeframe correctness
    for r in results:
        delta = abs(r["delta_pct"])
        assert delta < TOLERANCE, (
            f"[{r['timeframe']}] reported={r['reported_return_pct']:.6f}%  "
            f"expected={r['expected_return_pct']:.6f}%  delta={delta:.2e}"
        )

    print("All assertions passed: reported total_return matches expected for every timeframe.")

    # Cross-timeframe note
    daily_ret   = results[0]["reported_return_pct"]
    weekly_ret  = results[1]["reported_return_pct"]
    monthly_ret = results[2]["reported_return_pct"]

    print(f"\nCross-timeframe returns (different because first bars differ):")
    print(f"  daily  first_close={results[0]['first_close']:.4f}  return={daily_ret:.4f}%")
    print(f"  weekly first_close={results[1]['first_close']:.4f}  return={weekly_ret:.4f}%")
    print(f"  monthly first_close={results[2]['first_close']:.4f}  return={monthly_ret:.4f}%")
    print()
    print("  Note: the last close is identical across all timeframes (last daily close")
    print("  == last weekly close == last monthly close for same end date).")
    print("  The first close differs because weekly/monthly aggregation opens on the")
    print("  first bar of each period and closes on the last — so weekly bar 0 close")
    print("  != daily bar 0 close.")
    print()

    # Verify all timeframes share the same last close (within float rounding)
    assert abs(results[0]["last_close"] - results[1]["last_close"]) < 0.01, \
        "daily and weekly last close should match"
    assert abs(results[0]["last_close"] - results[2]["last_close"]) < 0.01, \
        "daily and monthly last close should match"
    print("Last-close parity confirmed: daily == weekly == monthly last bar close.")


if __name__ == "__main__":
    test_bah_timeframe_consistency()
