"""
Test that BacktestResult.summary() computes Sharpe on daily-aggregated PnL
regardless of the bar frequency of the backtest.

Verifies:
  1. Daily bars  — Sharpe = mean(daily_ret)/std(daily_ret)*sqrt(ann_factor)
  2. Intraday bars (8h) — after daily aggregation, Sharpe matches daily bars
  3. Sub-year   — annualized Sharpe, no blow-up (was 23x before the fix)
  4. Multi-year — consistent formula across durations
  5. BAH sanity — realistic equity curve gives Sharpe < 5
  6. RangeIndex — no AttributeError crash (falls back to bar-level)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.models import BacktestConfig
from testing.backtester.engine import BacktestResult

# ── Constants ────────────────────────────────────────────────────────────────

RNG  = np.random.default_rng(42)
INIT = 100_000.0

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_result(eq: pd.Series) -> BacktestResult:
    return BacktestResult(
        trades=[],
        equity_curve=eq,
        positions=pd.Series(0, index=eq.index),
        config=BacktestConfig(initial_capital=INIT),
    )


def _daily_equity(daily_rets: np.ndarray, start: str = "2020-01-02") -> pd.Series:
    """
    Build a daily equity series from an array of daily returns.
    equity[0] = INIT, equity[i] = INIT * prod(1+rets[0:i]).
    Length = len(daily_rets) + 1 bars (matching summary()'s pct_change usage).
    """
    n = len(daily_rets) + 1
    equity = np.empty(n)
    equity[0] = INIT
    np.cumprod(1.0 + daily_rets, out=equity[1:])
    equity[1:] *= INIT
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.Series(equity, index=dates)


def _intraday_equity(
    daily_rets: np.ndarray,
    bars_per_day: int,
    start: str = "2020-01-02",
) -> pd.Series:
    """
    Build an intraday equity series splitting each daily return equally across
    bars_per_day hourly bars.  An anchor bar at t=0 provides INIT.
    """
    daily_dates = pd.date_range(start, periods=len(daily_rets) + 1, freq="B")
    timestamps: list[pd.Timestamp] = [daily_dates[0]]
    values: list[float] = [INIT]
    eq = INIT
    for d, r_d in enumerate(daily_rets):
        bar_ret = r_d / bars_per_day
        day_ts = daily_dates[d + 1]
        for h in range(bars_per_day):
            eq *= 1.0 + bar_ret
            timestamps.append(day_ts + pd.Timedelta(hours=h + 1))
            values.append(eq)
    return pd.Series(values, index=pd.DatetimeIndex(timestamps))


_MIN_DAILY_RETS = 5  # must match engine.py constant


def _expected_sharpe(eq: pd.Series) -> float:
    """
    Ground-truth reference that mirrors summary()'s exact algorithm (same
    _MIN_DAILY_RETS guard, same ann_factor, pandas ddof=1 std).
    """
    n_bars = len(eq)
    returns = eq.pct_change().dropna()

    ann_factor: float = float(n_bars)
    if isinstance(eq.index, pd.DatetimeIndex) and n_bars > 1:
        _span = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600)
        if _span > 0:
            ann_factor = int(n_bars / _span)

    if isinstance(eq.index, pd.DatetimeIndex) and n_bars > 1:
        spans = eq.index.to_series().diff().dropna()
        med_secs = spans.dt.total_seconds().median() if not spans.empty else 86400.0
        if med_secs < 86400.0:
            daily_eq = eq.resample("D").last().dropna()
            drets_daily = (
                daily_eq.pct_change().dropna() if len(daily_eq) > 1
                else pd.Series(dtype=float)
            )
            if len(drets_daily) >= _MIN_DAILY_RETS:
                drets = drets_daily
                s_af = 252.0
            else:
                drets = returns
                s_af = ann_factor
        else:
            drets = returns
            s_af = ann_factor
    else:
        drets = returns
        s_af = ann_factor

    std = float(drets.std())
    return float(drets.mean() / std * math.sqrt(s_af)) if std > 0 else 0.0


# ── Tests ────────────────────────────────────────────────────────────────────

def test_daily_bars_sharpe():
    """
    Daily bars: summary() Sharpe matches mean/std*sqrt(ann_factor) applied
    directly to the equity's bar returns (with pandas ddof=1 std).
    """
    rets = RNG.normal(0.001, 0.005, 251)
    eq   = _daily_equity(rets)
    s    = _make_result(eq).summary()
    expected = _expected_sharpe(eq)

    diff = abs(s["sharpe_ratio"] - expected)
    assert diff < 1e-4, (
        f"Daily Sharpe {s['sharpe_ratio']:.6f} vs expected {expected:.6f} (diff={diff:.2e})"
    )
    print(
        f"  [PASS] daily bars  | sharpe={s['sharpe_ratio']:.4f}  "
        f"ann_ret={s['annualised_return_pct']:.2f}%  "
        f"ann_vol={s['annualised_volatility_pct']:.2f}%"
    )


def test_intraday_matches_daily():
    """
    8 intraday bars per day (1h each, all within the same calendar day):
      - summary() aggregates to daily → uses sqrt(252) as ann_factor
      - daily bars use calendar-span ann_factor (~261 for business-day dates)

    We check two things separately:
      (a) Intraday Sharpe matches _expected_sharpe() exactly (formula correct)
      (b) The mean/std RATIO of the aggregated daily returns matches bar-level
          daily returns (same signal, only ann_factor differs)
    """
    rets = RNG.normal(0.001, 0.005, 251)
    eq_daily = _daily_equity(rets)
    eq_intra = _intraday_equity(rets, bars_per_day=8)   # 8×1h bars per day

    s_intra = _make_result(eq_intra).summary()
    expected_intra = _expected_sharpe(eq_intra)

    # (a) formula correctness
    diff_formula = abs(s_intra["sharpe_ratio"] - expected_intra)
    assert diff_formula < 1e-4, (
        f"Intraday Sharpe {s_intra['sharpe_ratio']:.6f} vs "
        f"expected {expected_intra:.6f} (diff={diff_formula:.2e})"
    )

    # (b) mean/std ratio is the same across frequencies (ann_factor only scales)
    bar_rets  = eq_daily.pct_change().dropna()
    daily_agg = eq_intra.resample("D").last().dropna().pct_change().dropna()
    ratio_bar   = float(bar_rets.mean()  / bar_rets.std())
    ratio_intra = float(daily_agg.mean() / daily_agg.std())
    ratio_diff  = abs(ratio_intra - ratio_bar)
    assert ratio_diff < 0.005, (
        f"mean/std ratio mismatch: intraday={ratio_intra:.6f} daily={ratio_bar:.6f} "
        f"(diff={ratio_diff:.6f})"
    )

    print(
        f"  [PASS] intraday 8h | sharpe={s_intra['sharpe_ratio']:.4f}  "
        f"mean/std ratio intra={ratio_intra:.4f} bar={ratio_bar:.4f}"
    )


def test_sub_year_no_blowup():
    """
    Sub-year backtest (63 days ≈ 3 months).
    Old formula gave Sharpe ≈ 23 for buy-and-hold; daily-return formula
    must give an annualized Sharpe that matches the reference directly.
    """
    rets = RNG.normal(0.001, 0.005, 62)
    eq   = _daily_equity(rets)
    s    = _make_result(eq).summary()
    expected = _expected_sharpe(eq)

    diff = abs(s["sharpe_ratio"] - expected)
    assert diff < 1e-4, f"Sub-year Sharpe mismatch: {s['sharpe_ratio']:.4f} vs {expected:.4f}"
    assert abs(s["sharpe_ratio"]) < 15, (
        f"Sub-year Sharpe blow-up: {s['sharpe_ratio']:.2f} (old bug was ~23)"
    )
    print(
        f"  [PASS] sub-year 3m | sharpe={s['sharpe_ratio']:.4f}  "
        f"expected={expected:.4f}  (would have been ~23 before fix)"
    )


def test_multi_year_consistent():
    """
    1-year and 2-year of the same i.i.d. daily process each return the
    Sharpe matching their own reference formula (no cross-period distortion).
    """
    rets_1yr = RNG.normal(0.001, 0.005, 251)
    rets_2yr = RNG.normal(0.001, 0.005, 503)

    eq_1yr = _daily_equity(rets_1yr)
    eq_2yr = _daily_equity(rets_2yr)

    s1 = _make_result(eq_1yr).summary()
    s2 = _make_result(eq_2yr).summary()

    for label, s, eq in [("1yr", s1, eq_1yr), ("2yr", s2, eq_2yr)]:
        expected = _expected_sharpe(eq)
        diff = abs(s["sharpe_ratio"] - expected)
        assert diff < 1e-4, f"{label}: {s['sharpe_ratio']:.4f} vs expected {expected:.4f}"

    print(
        f"  [PASS] multi-year  | 1yr sharpe={s1['sharpe_ratio']:.4f}  "
        f"2yr sharpe={s2['sharpe_ratio']:.4f}"
    )


def test_bah_sharpe_reasonable():
    """
    Buy-and-hold on realistic equity process: 10% annual drift, 15% annual vol.
    Theoretical Sharpe ≈ 0.67.  Must be < 5 and match the reference formula.
    """
    daily_drift = 0.10 / 252
    daily_vol   = 0.15 / math.sqrt(252)
    rets = RNG.normal(daily_drift, daily_vol, 251)
    eq   = _daily_equity(rets)
    s    = _make_result(eq).summary()
    expected = _expected_sharpe(eq)

    diff = abs(s["sharpe_ratio"] - expected)
    assert diff < 1e-4, f"BAH Sharpe mismatch: {s['sharpe_ratio']:.4f} vs {expected:.4f}"
    assert abs(s["sharpe_ratio"]) < 5, f"BAH Sharpe unreasonably large: {s['sharpe_ratio']:.3f}"
    print(
        f"  [PASS] BAH sanity  | sharpe={s['sharpe_ratio']:.4f}  "
        f"ann_ret={s['annualised_return_pct']:.2f}%  "
        f"ann_vol={s['annualised_volatility_pct']:.2f}%"
    )


def test_integer_index_no_crash():
    """
    Equity curve with RangeIndex (no DatetimeIndex) must not crash.
    Falls back to bar-level returns with ann_factor = n_bars (1-year assumption).
    """
    rets = RNG.normal(0.001, 0.005, 251)
    equity = np.empty(252)
    equity[0] = INIT
    np.cumprod(1 + rets, out=equity[1:])
    equity[1:] *= INIT
    eq = pd.Series(equity)  # integer RangeIndex
    s  = _make_result(eq).summary()

    assert "sharpe_ratio" in s
    assert not math.isnan(s["sharpe_ratio"])
    assert not math.isinf(s["sharpe_ratio"])
    expected = _expected_sharpe(eq)
    assert abs(s["sharpe_ratio"] - expected) < 1e-4
    print(f"  [PASS] integer idx | sharpe={s['sharpe_ratio']:.4f} (bar-level fallback)")


def test_short_intraday_fold_no_blowup():
    """
    WFA folds on intraday data can be very short (2-4 trading days).
    Daily aggregation of such a fold produces 2-3 return observations;
    if those daily returns are near-identical (consistent strategy P&L),
    std → 0 and Sharpe explodes (observed SR=-1035 in real WFA output).

    Fix: fall back to bar-level returns when fewer than 5 daily observations.
    With realistic intraday noise the bar-level std is non-trivial, giving a
    bounded Sharpe.  We verify it stays within ±100.
    """
    bars_per_day = 60      # 1-min bars
    n_days = 3             # only 3 trading days → 2 daily returns after pct_change
    daily_loss = -0.02     # -2% per day, very consistent trend

    # Add realistic intraday noise (0.2% bar vol) so bar-level std is non-zero
    bar_mean = daily_loss / bars_per_day
    bar_noise = 0.002      # 0.2% per bar — typical for liquid intraday
    bar_rets = RNG.normal(bar_mean, bar_noise, n_days * bars_per_day)

    timestamps: list[pd.Timestamp] = []
    values: list[float] = []
    eq_val = 100_000.0
    base = pd.Timestamp("2020-01-06")
    for i, r in enumerate(bar_rets):
        d, b = divmod(i, bars_per_day)
        eq_val *= 1.0 + r
        timestamps.append(base + pd.Timedelta(days=d) + pd.Timedelta(minutes=b + 1))
        values.append(eq_val)

    eq_series = pd.Series(values, index=pd.DatetimeIndex(timestamps))
    s = _make_result(eq_series).summary()
    expected = _expected_sharpe(eq_series)

    # Formula must be consistent with reference
    assert abs(s["sharpe_ratio"] - expected) < 1e-4, (
        f"Sharpe {s['sharpe_ratio']:.4f} vs expected {expected:.4f}"
    )
    # Must be bounded — daily aggregation of 2 near-identical returns was SR≈-1035
    assert abs(s["sharpe_ratio"]) < 100, (
        f"Sharpe blew up on short intraday fold: {s['sharpe_ratio']:.2f} "
        f"(should use bar-level fallback with realistic noise)"
    )
    assert not math.isnan(s["sharpe_ratio"]) and not math.isinf(s["sharpe_ratio"])
    print(
        f"  [PASS] short fold  | sharpe={s['sharpe_ratio']:.4f}  "
        f"(bar-level fallback, daily-only would have given SR~-1035)"
    )


def test_sortino_no_negative_returns():
    """
    When all daily returns are positive (e.g. BAH over 2 good days),
    Sortino used to return 0 (wrong).  Should now equal Sharpe as a lower bound.
    """
    # 2 trading days, both positive — mirrors BAH in a 2-day validate window
    rets = np.array([0.008, 0.006])   # +0.8%, +0.6% — purely positive
    eq = _daily_equity(rets)
    s = _make_result(eq).summary()

    assert s["sortino_ratio"] != 0.0, "Sortino should not be 0 when all returns are positive"
    # Must equal Sharpe (lower-bound convention when no downside observed)
    assert abs(s["sortino_ratio"] - s["sharpe_ratio"]) < 1e-4, (
        f"Sortino {s['sortino_ratio']:.4f} should equal Sharpe {s['sharpe_ratio']:.4f} "
        f"when there are no negative returns"
    )
    print(
        f"  [PASS] sortino pos | sortino={s['sortino_ratio']:.4f}  "
        f"sharpe={s['sharpe_ratio']:.4f}  (was 0.0 before fix)"
    )


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_daily_bars_sharpe,
        test_intraday_matches_daily,
        test_sub_year_no_blowup,
        test_multi_year_consistent,
        test_bah_sharpe_reasonable,
        test_integer_index_no_crash,
        test_short_intraday_fold_no_blowup,
        test_sortino_no_negative_returns,
    ]
    print(f"\nDaily-PnL Sharpe refactor --- {len(tests)} tests\n")
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(t.__name__)
    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
