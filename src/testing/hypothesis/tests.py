"""
tests.py — Statistical hypothesis tests for strategy validation.

Classes:
    TestResult       — result container with statistic, p-value, and interpretation
    HypothesisTests  — battery of tests on BacktestResult (Sharpe, mean return,
                       win rate, normality, autocorrelation, stationarity, comparison)
    PermutationTest  — non-parametric permutation test on trade sequence
    BootstrapCI      — bootstrap confidence intervals for any metric

Functions:
    report()         — format a list of TestResult objects into a readable report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from testing.backtester.engine import BacktestResult


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    null_hypothesis: str
    interpretation: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        verdict = "REJECT H0" if self.reject_null else "FAIL TO REJECT H0"
        return (
            f"[{self.name}]\n"
            f"  H0: {self.null_hypothesis}\n"
            f"  stat={self.statistic:.4f}  p={self.p_value:.4f}  α={self.alpha}\n"
            f"  → {verdict}\n"
            f"  {self.interpretation}"
        )


def report(tests: list[TestResult]) -> str:
    """Format a list of TestResult objects into a readable text report."""
    sep = "=" * 62
    lines = [sep, "  HYPOTHESIS TEST REPORT", sep]
    for t in tests:
        verdict = "✓ REJECT H0" if t.reject_null else "✗ FAIL TO REJECT H0"
        lines += [
            f"\n[{t.name}]",
            f"  H0 : {t.null_hypothesis}",
            f"  stat={t.statistic:.4f}  p={t.p_value:.4f}  α={t.alpha}",
            f"  {verdict}",
            f"  → {t.interpretation}",
        ]
    lines.append("\n" + sep)
    return "\n".join(lines)


# ── Annualization helper ──────────────────────────────────────────────────────

def _ann_factor(result: BacktestResult) -> float:
    """Infer per-bar annualization factor from the equity curve's time frequency."""
    eq = result.equity_curve
    if len(eq) < 2 or not isinstance(eq.index, pd.DatetimeIndex):
        return 252.0
    deltas = eq.index.to_series().diff().dropna()
    if deltas.empty:
        return 252.0
    median_secs = deltas.dt.total_seconds().median()
    if median_secs <= 0:
        return 252.0
    secs_per_year = 365.25 * 24 * 3600
    return secs_per_year / median_secs


# ═══════════════════════════════════════════════════════════════════════════
# Main test class
# ═══════════════════════════════════════════════════════════════════════════

class HypothesisTests:
    """
    Statistical hypothesis tests on BacktestResult objects.

    All methods are static — no instantiation needed.

    Usage:
        tests = HypothesisTests.run_all(result)
        print(report(tests))

        t = HypothesisTests.sharpe_significance(result, benchmark_sharpe=1.0)
        if t.reject_null:
            print("Strategy has significantly positive Sharpe vs benchmark.")
    """

    @staticmethod
    def sharpe_significance(
        result: BacktestResult,
        benchmark_sharpe: float = 0.0,
        alpha: float = 0.05,
    ) -> TestResult:
        """
        One-sided t-test (Jobson-Korkie): strategy Sharpe > benchmark_sharpe.

        Uses the variance formula: Var(SR) ≈ (1 + SR²/2) / n, which accounts
        for estimation error in both mean and variance of returns.
        """
        returns = result.equity_curve.pct_change().dropna()
        if len(returns) < 10:
            raise ValueError("Need ≥ 10 return observations for Sharpe significance test")

        n = len(returns)
        ann = _ann_factor(result)
        r_std = returns.std(ddof=1)
        observed_sr = (returns.mean() / r_std * np.sqrt(ann)) if r_std > 0 else 0.0

        se = np.sqrt((1 + 0.5 * observed_sr ** 2) / n)
        t_stat = (observed_sr - benchmark_sharpe) / se if se > 0 else 0.0
        p_value = float(stats.t.sf(t_stat, df=n - 1))

        return TestResult(
            name="sharpe_significance",
            statistic=float(t_stat),
            p_value=p_value,
            alpha=alpha,
            reject_null=p_value < alpha,
            null_hypothesis=f"Annualized Sharpe ≤ {benchmark_sharpe}",
            interpretation=(
                f"Observed annualized Sharpe = {observed_sr:.3f}. "
                + (
                    "Strategy Sharpe is significantly above benchmark."
                    if p_value < alpha
                    else "Cannot confirm Sharpe significantly exceeds benchmark."
                )
            ),
            meta={"observed_sharpe": observed_sr, "n_bars": n, "ann_factor": ann},
        )

    @staticmethod
    def mean_return(result: BacktestResult, alpha: float = 0.05) -> TestResult:
        """Two-sided t-test: mean trade PnL ≠ 0."""
        tdf = result.trades_df()
        if tdf.empty or "pnl" not in tdf.columns:
            raise ValueError("No trades available")
        pnl = tdf["pnl"].values
        t_stat, p_value = stats.ttest_1samp(pnl, popmean=0.0)
        mean_pnl = float(pnl.mean())
        return TestResult(
            name="mean_return",
            statistic=float(t_stat),
            p_value=float(p_value),
            alpha=alpha,
            reject_null=p_value < alpha,
            null_hypothesis="Mean trade PnL = 0",
            interpretation=(
                f"Mean trade PnL = {mean_pnl:.4f} ({'+' if mean_pnl >= 0 else ''}{mean_pnl:.4f}). "
                + (
                    f"Significantly {'positive' if mean_pnl > 0 else 'negative'} (p={p_value:.4f})."
                    if p_value < alpha
                    else f"Cannot reject zero mean (p={p_value:.4f})."
                )
            ),
            meta={"mean_pnl": mean_pnl, "std_pnl": float(pnl.std()), "n_trades": len(pnl)},
        )

    @staticmethod
    def win_rate(
        result: BacktestResult,
        expected_rate: float = 0.5,
        alpha: float = 0.05,
    ) -> TestResult:
        """
        One-sided binomial test: win rate > expected_rate.

        Use expected_rate=0.5 to test above-coin-flip performance.
        """
        tdf = result.trades_df()
        if tdf.empty or "pnl" not in tdf.columns:
            raise ValueError("No trades available")
        wins = int((tdf["pnl"] > 0).sum())
        n = len(tdf)
        observed_wr = wins / n
        binom = stats.binomtest(wins, n, expected_rate, alternative="greater")
        p_value = float(binom.pvalue)
        return TestResult(
            name="win_rate",
            statistic=observed_wr,
            p_value=p_value,
            alpha=alpha,
            reject_null=p_value < alpha,
            null_hypothesis=f"Win rate ≤ {expected_rate:.0%}",
            interpretation=(
                f"Win rate = {observed_wr:.1%} ({wins}/{n} trades). "
                + (
                    f"Significantly above {expected_rate:.0%} (p={p_value:.4f})."
                    if p_value < alpha
                    else f"Cannot confirm win rate exceeds {expected_rate:.0%} (p={p_value:.4f})."
                )
            ),
            meta={"wins": wins, "n_trades": n, "observed_win_rate": observed_wr},
        )

    @staticmethod
    def normality(result: BacktestResult, alpha: float = 0.05) -> TestResult:
        """
        Jarque-Bera normality test on per-trade return percentages.

        Non-normal returns invalidate Sharpe ratio as a summary statistic —
        use Sortino or Calmar when H0 is rejected.
        """
        tdf = result.trades_df()
        if tdf.empty or "pnl_pct" not in tdf.columns:
            raise ValueError("No trades with pnl_pct available")
        returns = tdf["pnl_pct"].dropna().values
        if len(returns) < 8:
            raise ValueError("Need ≥ 8 trades for normality test")
        jb_stat, p_value = stats.jarque_bera(returns)
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))
        return TestResult(
            name="normality",
            statistic=float(jb_stat),
            p_value=float(p_value),
            alpha=alpha,
            reject_null=p_value < alpha,
            null_hypothesis="Trade returns are normally distributed",
            interpretation=(
                f"Skewness={skew:.3f}, excess kurtosis={kurt:.3f}. "
                + (
                    "Returns are NOT normal — Sharpe ratio may be unreliable; prefer Sortino/Calmar."
                    if p_value < alpha
                    else "Cannot reject normality of trade returns."
                )
            ),
            meta={"skewness": skew, "excess_kurtosis": kurt, "n_trades": len(returns)},
        )

    @staticmethod
    def autocorrelation(
        result: BacktestResult,
        lags: int = 10,
        alpha: float = 0.05,
    ) -> TestResult:
        """
        Ljung-Box test for autocorrelation in strategy bar returns.

        Significant autocorrelation in a backtest is a red flag for look-ahead
        bias or signal smoothing that doesn't hold in live trading.
        """
        returns = result.equity_curve.pct_change().dropna()
        if len(returns) < lags + 5:
            raise ValueError(f"Need ≥ {lags + 5} bars for autocorrelation test")

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(returns, lags=[lags], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            p_value = float(lb["lb_pvalue"].iloc[0])
        except ImportError:
            # Manual Ljung-Box fallback
            n = len(returns)
            r = returns.values
            acf_vals = np.array([
                float(pd.Series(r).autocorr(lag=k)) for k in range(1, lags + 1)
            ])
            lb_stat = float(n * (n + 2) * np.sum(acf_vals ** 2 / (n - np.arange(1, lags + 1))))
            p_value = float(1 - stats.chi2.cdf(lb_stat, df=lags))

        return TestResult(
            name="autocorrelation",
            statistic=lb_stat,
            p_value=p_value,
            alpha=alpha,
            reject_null=p_value < alpha,
            null_hypothesis=f"No autocorrelation in returns up to lag {lags}",
            interpretation=(
                "Significant autocorrelation detected — check for look-ahead bias."
                if p_value < alpha
                else "No significant autocorrelation in bar returns."
            ),
            meta={"lags": lags},
        )

    @staticmethod
    def stationarity(result: BacktestResult, alpha: float = 0.05) -> TestResult:
        """
        Augmented Dickey-Fuller test on the equity curve.

        Rejecting the unit-root null (random walk) is consistent with genuine alpha —
        the equity curve has a deterministic drift, not just random variation.

        Requires statsmodels: pip install statsmodels
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            raise ImportError("statsmodels required: pip install statsmodels")

        equity = result.equity_curve.dropna()
        if len(equity) < 20:
            raise ValueError("Need ≥ 20 bars for ADF test")

        adf_stat, p_value, _, _, critical_values, _ = adfuller(equity, autolag="AIC")
        return TestResult(
            name="stationarity",
            statistic=float(adf_stat),
            p_value=float(p_value),
            alpha=alpha,
            reject_null=p_value < alpha,
            null_hypothesis="Equity curve has a unit root (random walk)",
            interpretation=(
                "Equity curve is stationary — consistent with genuine, persistent alpha."
                if p_value < alpha
                else "Cannot reject random walk — equity curve may not reflect genuine edge."
            ),
            meta={"critical_values": critical_values},
        )

    @staticmethod
    def compare(
        result_a: BacktestResult,
        result_b: BacktestResult,
        metric: str = "sharpe_ratio",
        alpha: float = 0.05,
        n_bootstrap: int = 2000,
        seed: int = 42,
    ) -> TestResult:
        """
        Bootstrap test: is result_a's metric significantly greater than result_b's?

        Uses IID bootstrap on bar returns. For highly autocorrelated strategies,
        consider using PermutationTest on trade PnL instead.

        Supported metrics: "sharpe_ratio", "total_return_pct", "sortino_ratio"
        """
        rng = np.random.default_rng(seed)
        ann_a = _ann_factor(result_a)
        ann_b = _ann_factor(result_b)
        ret_a = result_a.equity_curve.pct_change().dropna().values
        ret_b = result_b.equity_curve.pct_change().dropna().values

        def _compute(returns: np.ndarray, ann: float) -> float:
            if metric == "sharpe_ratio":
                std = returns.std()
                return float(returns.mean() / std * np.sqrt(ann)) if std > 0 else 0.0
            elif metric == "total_return_pct":
                return float(np.prod(1 + returns) - 1) * 100
            elif metric == "sortino_ratio":
                neg = returns[returns < 0]
                dstd = neg.std() if len(neg) > 0 else 1e-9
                return float(returns.mean() / dstd * np.sqrt(ann)) if dstd > 0 else 0.0
            else:
                raise ValueError(f"Unsupported metric for compare(): {metric!r}")

        obs_a = _compute(ret_a, ann_a)
        obs_b = _compute(ret_b, ann_b)
        observed_diff = obs_a - obs_b

        null_diffs = np.array([
            _compute(rng.choice(ret_a, size=len(ret_a), replace=True), ann_a)
            - _compute(rng.choice(ret_b, size=len(ret_b), replace=True), ann_b)
            for _ in range(n_bootstrap)
        ])
        # One-sided: P(A ≤ B) under the null
        p_value = float(np.mean(null_diffs <= 0))

        return TestResult(
            name=f"strategy_comparison_{metric}",
            statistic=float(observed_diff),
            p_value=p_value,
            alpha=alpha,
            reject_null=p_value < alpha,
            null_hypothesis=f"Strategy A {metric} ≤ Strategy B {metric}",
            interpretation=(
                f"A={obs_a:.3f}, B={obs_b:.3f}, diff={observed_diff:+.3f}. "
                + (
                    f"A is significantly better (p={p_value:.4f})."
                    if p_value < alpha
                    else f"Cannot confirm A outperforms B (p={p_value:.4f})."
                )
            ),
            meta={
                "metric": metric,
                "metric_a": obs_a,
                "metric_b": obs_b,
                "n_bootstrap": n_bootstrap,
            },
        )

    @staticmethod
    def run_all(result: BacktestResult, alpha: float = 0.05) -> list[TestResult]:
        """
        Run the standard battery of tests.

        Skips any test that raises (e.g. not enough data, missing statsmodels).
        """
        candidates = [
            lambda r: HypothesisTests.sharpe_significance(r, alpha=alpha),
            lambda r: HypothesisTests.mean_return(r, alpha=alpha),
            lambda r: HypothesisTests.win_rate(r, alpha=alpha),
            lambda r: HypothesisTests.normality(r, alpha=alpha),
            lambda r: HypothesisTests.autocorrelation(r, alpha=alpha),
            lambda r: HypothesisTests.stationarity(r, alpha=alpha),
        ]
        results = []
        for fn in candidates:
            try:
                results.append(fn(result))
            except Exception:
                pass
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Permutation test
# ═══════════════════════════════════════════════════════════════════════════

class PermutationTest:
    """
    Non-parametric permutation test on the trade sequence.

    Shuffles the order of trade PnLs to build a null distribution for a metric,
    then tests whether the observed value is significantly better than random.

    This is more robust than parametric tests when the number of trades is small
    or return distributions are heavy-tailed.

    Usage:
        pt = PermutationTest(metric="sharpe_ratio", n_permutations=2000)
        result = pt.run(bt_result)
        print(result)
    """

    _SUPPORTED = ("sharpe_ratio", "total_return_pct", "profit_factor")

    def __init__(
        self,
        metric: str = "sharpe_ratio",
        n_permutations: int = 2000,
        seed: int = 42,
    ):
        if metric not in self._SUPPORTED:
            raise ValueError(f"metric must be one of {self._SUPPORTED}")
        self.metric = metric
        self.n_permutations = n_permutations
        self.seed = seed

    def _compute(self, pnl: np.ndarray, initial: float, ann_factor: float = 252.0) -> float:
        equity = initial + np.cumsum(pnl)
        returns = np.diff(equity) / equity[:-1]
        if self.metric == "sharpe_ratio":
            std = returns.std()
            return float(returns.mean() / std * np.sqrt(ann_factor)) if std > 0 else 0.0
        elif self.metric == "total_return_pct":
            return float((equity[-1] / initial - 1) * 100)
        elif self.metric == "profit_factor":
            gross_win = pnl[pnl > 0].sum()
            gross_loss = abs(pnl[pnl < 0].sum())
            return float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
        return 0.0

    def run(self, result: BacktestResult, alpha: float = 0.05) -> TestResult:
        rng = np.random.default_rng(self.seed)
        tdf = result.trades_df()
        if tdf.empty or "pnl" not in tdf.columns:
            raise ValueError("No trades available")

        pnl = tdf["pnl"].values
        initial = result.config.initial_capital
        af = _ann_factor(result)
        observed = self._compute(pnl, initial, af)

        null_dist = np.array([
            self._compute(rng.permutation(pnl), initial, af)
            for _ in range(self.n_permutations)
        ])
        p_value = float(np.mean(null_dist >= observed))

        return TestResult(
            name=f"permutation_{self.metric}",
            statistic=observed,
            p_value=p_value,
            alpha=alpha,
            reject_null=p_value < alpha,
            null_hypothesis=f"{self.metric} ≤ random permutation of trade order",
            interpretation=(
                f"Observed {self.metric} = {observed:.3f} "
                f"(null median={np.median(null_dist):.3f}). "
                + (
                    f"Significantly outperforms random order (p={p_value:.4f})."
                    if p_value < alpha
                    else f"Cannot reject random-order null (p={p_value:.4f})."
                )
            ),
            meta={
                "observed": observed,
                "null_mean": float(null_dist.mean()),
                "null_p5": float(np.percentile(null_dist, 5)),
                "null_p95": float(np.percentile(null_dist, 95)),
                "n_permutations": self.n_permutations,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════════

class BootstrapCI:
    """
    Bootstrap confidence intervals for BacktestResult performance metrics.

    Samples trade PnL with replacement to quantify estimation uncertainty —
    useful when the number of trades is small (< 100).

    Usage:
        ci = BootstrapCI(ci=0.95)
        intervals = ci.run(result, metrics=["sharpe_ratio", "total_return_pct"])
        # → {"sharpe_ratio": {"observed": 1.2, "lower": 0.8, "upper": 1.6, ...}, ...}
    """

    def __init__(self, n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self.seed = seed

    def run(
        self,
        result: BacktestResult,
        metrics: list[str] | None = None,
    ) -> dict[str, dict]:
        rng = np.random.default_rng(self.seed)
        tdf = result.trades_df()
        if tdf.empty or "pnl" not in tdf.columns:
            raise ValueError("No trades available")

        pnl = tdf["pnl"].values
        initial = result.config.initial_capital
        target = metrics or ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]

        af = _ann_factor(result)

        def _all_metrics(pnl_seq: np.ndarray) -> dict:
            equity = initial + np.cumsum(pnl_seq)
            ret = np.diff(equity) / equity[:-1]
            std = float(ret.std()) if len(ret) > 1 else 0.0
            total_ret = float(equity[-1] / initial - 1)
            # Trade-based Sharpe: each trade is one period, annualize by trades/year (af).
            # Note: this is a trade-frequency Sharpe, not a daily-return Sharpe — it is
            # internally consistent across bootstrap samples but differs from summary().
            sr = float(ret.mean() / std * np.sqrt(af)) if std > 0 else 0.0
            peak = np.maximum.accumulate(equity)
            max_dd = float(((equity - peak) / peak).min() * 100)
            return {
                "total_return_pct": float(total_ret * 100),
                "sharpe_ratio": sr,
                "max_drawdown_pct": max_dd,
                "win_rate_pct": float((pnl_seq > 0).mean() * 100),
            }

        observed = _all_metrics(pnl)
        samples = [_all_metrics(rng.choice(pnl, size=len(pnl), replace=True)) for _ in range(self.n_bootstrap)]

        lo = (1 - self.ci) / 2 * 100
        hi = (1 + self.ci) / 2 * 100
        output = {}
        for m in target:
            if m not in observed:
                continue
            vals = np.array([s[m] for s in samples])
            output[m] = {
                "observed": observed[m],
                "lower": float(np.percentile(vals, lo)),
                "upper": float(np.percentile(vals, hi)),
                "std_error": float(vals.std()),
                "ci": self.ci,
            }
        return output
