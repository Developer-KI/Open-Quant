"""
overfitting.py — Quantitative overfitting detection for strategy research.

Classes:
    DeflatedSharpeRatio          — DSR (Bailey & López de Prado 2014): adjusts Sharpe for
                                   non-normality and multiple testing across trial variants
    MultipleComparisonCorrection — Bonferroni and Benjamini-Hochberg correction for ParamSweep
    ProbabilityOfBacktestOverfitting — PBO via Combinatorial Purged Cross-Validation (CPCV)
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from testing.backtester.engine import Backtester, BacktestResult
from testing.backtester.stress import StressResult


# ═══════════════════════════════════════════════════════════════════════════
# 1. Deflated Sharpe Ratio
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DSRResult:
    observed_sharpe: float        # raw per-bar SR (not annualized)
    deflated_sharpe: float        # threshold SR needed to claim significance
    p_value: float                # P(SR > deflated_SR | H0)
    reject_null: bool
    alpha: float
    n_trials: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        verdict = "REJECT H0 (genuine edge)" if self.reject_null else "FAIL TO REJECT (likely overfit)"
        return (
            f"DSRResult: observed_SR={self.observed_sharpe:.3f}, "
            f"deflated_SR={self.deflated_sharpe:.3f}, "
            f"p={self.p_value:.4f}, n_trials={self.n_trials}\n"
            f"  → {verdict}"
        )


def _expected_max_sr(n_trials: int) -> float:
    """
    Expected maximum SR from n_trials iid N(0,1) samples.
    Uses the Euler-Mascheroni approximation (Bailey & López de Prado, eq. 2).
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float((1 - gamma) * z1 + gamma * z2)


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio — Bailey & López de Prado (2014).

    Corrects the observed Sharpe ratio for three biases that cause overfitting:
      1. Multiple testing: if you tried N param combinations, one will look good by chance.
      2. Non-normality: fat tails and negative skew inflate apparent SR.
      3. Short track record: small-sample estimation noise.

    A strategy passes DSR only if its Sharpe is significantly better than the best
    Sharpe expected from random search across the same N trials.

    Usage:
        dsr = DeflatedSharpeRatio()

        # Single backtest, tested against 20 parameter variants you looked at
        result = dsr.compute(bt_result, n_trials=20)
        print(result)

        # Directly from a ParamSweep result (n_trials inferred automatically)
        result = dsr.from_sweep(sweep_result)
        print(result)
    """

    def compute(
        self,
        result: BacktestResult,
        n_trials: int = 1,
        trials_sharpes: list[float] | None = None,
        alpha: float = 0.05,
    ) -> DSRResult:
        """
        Args:
            n_trials:        Total number of strategy/parameter variants evaluated.
            trials_sharpes:  Sharpe ratios of all other trials (overrides n_trials benchmark).
            alpha:           Significance level.
        """
        returns = result.equity_curve.pct_change().dropna().values
        n = len(returns)
        if n < 10:
            raise ValueError("Need ≥ 10 return observations for DSR")

        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))   # excess kurtosis
        r_std = returns.std(ddof=1)
        sr = float(returns.mean() / r_std) if r_std > 0 else 0.0  # per-bar SR

        # Benchmark: best SR expected under the null from n_trials independent trials
        if trials_sharpes is not None and len(trials_sharpes) > 0:
            sr_benchmark = float(max(trials_sharpes))
        else:
            sr_benchmark = _expected_max_sr(n_trials)

        # Variance of SR estimator (corrected for skewness and kurtosis)
        sr_var = max(1e-12, (1 - skew * sr + (kurt + 3) / 4 * sr ** 2) / (n - 1))

        # Probability that observed SR exceeds benchmark SR
        z_stat = (sr - sr_benchmark) / np.sqrt(sr_var)
        p_value = float(stats.norm.sf(z_stat))

        # Minimum SR that would be significant at alpha (the "deflated" threshold)
        deflated_sr = sr_benchmark + np.sqrt(sr_var) * stats.norm.ppf(1 - alpha)

        return DSRResult(
            observed_sharpe=sr,
            deflated_sharpe=deflated_sr,
            p_value=p_value,
            reject_null=p_value < alpha,
            alpha=alpha,
            n_trials=n_trials,
            meta={
                "skewness": skew,
                "excess_kurtosis": kurt,
                "sr_benchmark": sr_benchmark,
                "n_returns": n,
                "z_stat": float(z_stat),
            },
        )

    def from_sweep(
        self,
        sweep_result: StressResult,
        target_result: BacktestResult | None = None,
        metric: str = "sharpe_ratio",
        alpha: float = 0.05,
    ) -> DSRResult:
        """
        Apply DSR to the best result in a ParamSweep.

        n_trials is inferred from the number of rows in sweep_result.summary,
        and sr_benchmark is set to the maximum Sharpe among all trials.

        Args:
            target_result: Backtest to evaluate. Defaults to the one with the highest metric.
        """
        sr_vals = sweep_result.summary[metric].dropna().tolist()
        n_trials = len(sr_vals)

        if target_result is None:
            if not sweep_result.results:
                raise ValueError("sweep_result has no BacktestResult objects")
            target_result = max(
                sweep_result.results.values(),
                key=lambda r: r.summary().get(metric, float("-inf")),
            )

        return self.compute(
            target_result,
            n_trials=n_trials,
            trials_sharpes=sr_vals,
            alpha=alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Multiple comparison correction
# ═══════════════════════════════════════════════════════════════════════════

class MultipleComparisonCorrection:
    """
    Correct for multiple testing bias in parameter sweeps.

    When you test many parameter combinations some will appear significant by
    chance. These corrections control either the family-wise error rate (Bonferroni)
    or the false discovery rate (Benjamini-Hochberg).

    Usage:
        mcc = MultipleComparisonCorrection()
        corrected = mcc.benjamini_hochberg(sweep_result, alpha=0.05)
        print(corrected[["sharpe_ratio", "p_value", "adjusted_p", "significant"]])
    """

    @staticmethod
    def _raw_pvalues(sr_vals: np.ndarray) -> np.ndarray:
        """Approximate two-sided p-values via normal distribution on Sharpe ratios."""
        return np.array([float(2 * stats.norm.sf(abs(sr))) for sr in sr_vals])

    def bonferroni(
        self,
        sweep_result: StressResult,
        alpha: float = 0.05,
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """
        Bonferroni correction: divide alpha by the number of tests.

        Most conservative — controls family-wise error rate. Use when a single
        false positive would be very costly.
        """
        df = sweep_result.summary.copy()
        if metric not in df.columns:
            raise ValueError(f"Metric {metric!r} not found in sweep summary")
        n = len(df)
        p_vals = self._raw_pvalues(df[metric].fillna(0).values)
        df["p_value"] = p_vals
        df["adjusted_alpha"] = alpha / n
        df["significant"] = p_vals < (alpha / n)
        return df

    def benjamini_hochberg(
        self,
        sweep_result: StressResult,
        alpha: float = 0.05,
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """
        Benjamini-Hochberg FDR correction.

        Less conservative than Bonferroni — controls the expected fraction of
        false discoveries among all claimed discoveries. Recommended for
        exploratory parameter searches.
        """
        df = sweep_result.summary.copy()
        if metric not in df.columns:
            raise ValueError(f"Metric {metric!r} not found in sweep summary")

        p_vals = self._raw_pvalues(df[metric].fillna(0).values)
        n = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        sorted_p = p_vals[sorted_idx]

        # Step-up BH: find the largest k where p_(k) ≤ k/n * alpha
        thresholds = (np.arange(1, n + 1) / n) * alpha
        below = sorted_p <= thresholds
        bh_threshold = thresholds[np.where(below)[0].max()] if below.any() else 0.0

        # Adjusted p-values (step-up)
        adj = np.minimum(1.0, sorted_p * n / np.arange(1, n + 1))
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        adj_original = np.empty(n)
        adj_original[sorted_idx] = adj

        df["p_value"] = p_vals
        df["adjusted_p"] = adj_original
        df["bh_threshold"] = bh_threshold
        df["significant"] = adj_original < alpha
        return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Probability of Backtest Overfitting
# ═══════════════════════════════════════════════════════════════════════════

class ProbabilityOfBacktestOverfitting:
    """
    Probability of Backtest Overfitting (PBO) via Combinatorial Purged Cross-Validation.
    Bailey, Borwein, López de Prado & Zhu (2014).

    Partitions the data into n_splits subperiods, then for each random IS/OOS partition:
      - Selects the best param combo on IS
      - Computes how that combo ranks among all combos on OOS
      - PBO = fraction of paths where best-IS underperforms median OOS (logit < 0)

    PBO < 0.5 is healthy (IS selection predicts OOS leadership).
    PBO > 0.5 means your IS optimization is likely to pick overfit parameters.

    WARNING: This runs n_paths × n_param_combos × 2 full backtests.
    With large grids or slow strategies, prefer n_paths=50 and n_splits=4.

    Usage:
        pbo = ProbabilityOfBacktestOverfitting(n_splits=6, n_paths=100)
        result = pbo.run(
            strategy_cls=EMACross,
            param_grid={"fast": [10, 20, 50], "slow": [50, 100, 200]},
            fixed_params={"symbol": "SPY"},
            universe=universe,
        )
        print(result["interpretation"])   # PBO = 32% — low overfitting risk
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_paths: int = 100,
        seed: int = 42,
        config=None,
        cost_model=None,
        sizer=None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be ≥ 2")
        self.n_splits = n_splits
        self.n_paths = n_paths
        self.seed = seed
        self.config = config
        self.cost_model = cost_model
        self.sizer = sizer

    def run(
        self,
        strategy_cls,
        param_grid: dict,
        universe,
        fixed_params: dict | None = None,
        timeframe: str | None = None,
        metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        from testing.hypothesis.splits import _slice_universe
        from strategy.built_in import SingleAssetStrategy

        fixed = fixed_params or {}
        rng = np.random.default_rng(self.seed)
        ref_sym = universe.symbols[0]
        idx = universe.ohlcv(ref_sym).index
        n = len(idx)

        keys = list(param_grid.keys())
        combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
        n_combos = len(combos)
        if n_combos < 2:
            raise ValueError("Need ≥ 2 parameter combinations for PBO")

        split_size = n // self.n_splits
        subperiods = [
            idx[i * split_size: (i + 1) * split_size] for i in range(self.n_splits - 1)
        ]
        subperiods.append(idx[(self.n_splits - 1) * split_size:])  # last absorbs remainder

        n_train = self.n_splits // 2

        def _metric_for(universe_subset, combo: dict) -> float:
            params = {**fixed, **combo}
            if issubclass(strategy_cls, SingleAssetStrategy) and "symbol" not in params:
                params["symbol"] = ref_sym
            strategy = strategy_cls(**params)
            bt = Backtester(
                strategy=strategy,
                config=self.config,
                cost_model=self.cost_model,
                sizer=self.sizer,
            )
            try:
                r = bt.run(universe=copy.deepcopy(universe_subset), timeframe=timeframe)
                return float(r.summary().get(metric, 0.0) or 0.0)
            except Exception:
                return 0.0

        lambda_values: list[float] = []

        for _ in range(self.n_paths):
            perm = rng.permutation(self.n_splits)
            is_idx_list = [subperiods[i] for i in perm[:n_train]]
            oos_idx_list = [subperiods[i] for i in perm[n_train:]]

            is_idx = pd.concat(is_idx_list).sort_values()
            oos_idx = pd.concat(oos_idx_list).sort_values()

            is_u = _slice_universe(universe, is_idx)
            oos_u = _slice_universe(universe, oos_idx)

            is_metrics = [_metric_for(is_u, c) for c in combos]
            oos_metrics = [_metric_for(oos_u, c) for c in combos]

            best_is_idx = int(np.argmax(is_metrics))
            oos_of_best = oos_metrics[best_is_idx]
            # Relative OOS rank of the IS-best combo (0..1)
            rank = float(np.sum(np.array(oos_metrics) <= oos_of_best)) / n_combos
            rank = max(1e-6, min(1 - 1e-6, rank))
            lambda_values.append(np.log(rank / (1 - rank)))

        lambda_arr = np.array(lambda_values)
        pbo = float(np.mean(lambda_arr < 0))

        return {
            "pbo": pbo,
            "interpretation": (
                f"PBO = {pbo:.1%}. "
                + (
                    "High overfitting risk — IS parameter selection unlikely to generalize to OOS."
                    if pbo > 0.5
                    else "Low overfitting risk — IS parameter selection tends to generalize."
                )
            ),
            "n_paths": self.n_paths,
            "n_splits": self.n_splits,
            "n_param_combos": n_combos,
            "lambda_mean": float(lambda_arr.mean()),
            "lambda_values": lambda_values,
        }
