"""
stress.py — Modular stress testing framework.

Four types of stress tests:
  1. ParamSweep       — sweep any Strategy parameters (grid or random); works for
                        both single-asset and multi-asset strategies
  2. RegimeStressTest — test across market regime subsets (vol, trend, etc.)
  3. MonteCarloStress — bootstrap / shuffle trades to build confidence intervals


All tests return a StressResult with a summary DataFrame and optional plots.
"""

from __future__ import annotations

import copy
import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.models import BacktestConfig
from .costs import CostModel
from .engine import Backtester, BacktestResult

from strategy.base import Strategy
from strategy.built_in import SingleAssetStrategy
from core.universe import Universe


# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class StressResult:
    name: str
    summary: pd.DataFrame  # one row per scenario
    results: dict[str, BacktestResult] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_csv(self, path: str | None = None):
        p = path or f"stress_{self.name}.csv"
        self.summary.to_csv(p, index=False)
        return p

    def best(self, metric: str = "sharpe_ratio") -> pd.Series:
        return self.summary.loc[self.summary[metric].idxmax()]

    def worst(self, metric: str = "sharpe_ratio") -> pd.Series:
        return self.summary.loc[self.summary[metric].idxmin()]

    def plot_heatmap(
        self, x: str, y: str, z: str = "sharpe_ratio", save_path: str | None = None
    ):
        """Pivot two sweep params into a heatmap of a target metric."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pivot = self.summary.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v}" for v in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v}" for v in pivot.index])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{z} heatmap")
        fig.colorbar(im)
        fig.tight_layout()
        path = save_path or f"heatmap_{self.name}_{z}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# ═══════════════════════════════════════════════════════════════════════════
# 1. Parameter sweep (any Strategy — single- or multi-asset)
# ═══════════════════════════════════════════════════════════════════════════


class ParamSweep:
    """
    Grid/random sweep over Strategy constructor parameters.

    Works for both SingleAssetStrategy and multi-asset Strategy subclasses.
    For SingleAssetStrategy, the first symbol in the universe is injected
    automatically as the ``symbol`` argument.

    Usage (single-asset):
        sweep = ParamSweep(
            strategy_cls=EMACrossStrategy,
            param_grid={"fast": [5, 8, 12, 20], "slow": [21, 26, 50]},
        )
        result = sweep.run(universe=universe, timeframe="1h")
        result.plot_heatmap("fast", "slow")

    Usage (multi-asset):
        sweep = ParamSweep(
            strategy_cls=ZPairsSpreadStrategy,
            param_grid={"lookback": [30, 60, 120], "entry_z": [1.5, 2.0, 2.5]},
            fixed_params={"asset_a": "BTC", "asset_b": "ETH"},
        )
        result = sweep.run(universe=universe)

    Set ``n_jobs=-1`` (default) to use all CPU cores, or ``n_jobs=1`` for
    sequential execution (useful when debugging exceptions).
    """

    def __init__(
        self,
        strategy_cls: type[Strategy],
        param_grid: dict[str, list],
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        sizer=None,
        stop_loss=None,
        fixed_params: dict | None = None,
        n_random: int | None = None,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.config = config or BacktestConfig()
        self.cost_model = cost_model
        self.sizer = sizer
        self.stop_loss = stop_loss
        self.fixed_params = fixed_params or {}
        self.n_random = n_random
        self.seed = seed
        self.n_jobs = n_jobs

    def _build_combos(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        vals = list(self.param_grid.values())
        combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        if self.n_random and self.n_random < len(combos):
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(combos), size=self.n_random, replace=False)
            combos = [combos[i] for i in idx]
        return combos

    def _run_one(
        self, combo: dict, universe: Universe, timeframe: str | None
    ) -> tuple[dict, BacktestResult | None, str | None]:
        params = {**self.fixed_params, **combo}
        if issubclass(self.strategy_cls, SingleAssetStrategy):
            sym = universe.symbols[0] if universe.symbols else "ASSET"
            strategy = self.strategy_cls(symbol=sym, **params)
        else:
            strategy = self.strategy_cls(**params)

        bt = Backtester(
            strategy=strategy,
            config=self.config,
            cost_model=self.cost_model,
            sizer=self.sizer,
            stop_loss=self.stop_loss,
        )
        try:
            res = bt.run(universe=copy.deepcopy(universe), timeframe=timeframe)
            return combo, res, None
        except Exception as e:
            return combo, None, str(e)

    def run(
        self,
        universe: Universe,
        timeframe: str | None = None,
    ) -> StressResult:
        combos = self._build_combos()
        rows: list[dict] = []
        results: dict[str, BacktestResult] = {}
        n_workers = os.cpu_count() if self.n_jobs == -1 else self.n_jobs

        def _call(combo: dict):
            return self._run_one(combo, universe, timeframe)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for combo, res, err in ex.map(_call, combos):
                if res is not None:
                    rows.append({**combo, **res.summary()})
                    results[str(combo)] = res
                else:
                    rows.append({**combo, "error": err})

        return StressResult(
            name="param_sweep",
            summary=pd.DataFrame(rows),
            results=results,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Regime-based stress test
# ═══════════════════════════════════════════════════════════════════════════


class RegimeStressTest:
    """
    Split data by market regime and run backtest on each subset.

    Usage:
        rst = RegimeStressTest(regime_fn=my_classifier, regime_symbol="BTC")
        result = rst.run(strategy=my_strategy, universe=my_universe)

    Set ``n_jobs=-1`` (default) to use all CPU cores, or ``n_jobs=1`` for
    sequential execution.
    """

    def __init__(
        self,
        regime_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        regime_symbol: str | None = None,
        n_jobs: int = -1,
    ):
        self.regime_fn = regime_fn or self._default_vol_regime
        self.config = config or BacktestConfig()
        self.cost_model = cost_model
        self.regime_symbol = regime_symbol
        self.n_jobs = n_jobs

    @staticmethod
    def _default_vol_regime(data: pd.DataFrame) -> pd.Series:
        """Classify bars into low/medium/high volatility regimes."""
        returns = data["close"].pct_change()
        rolling_vol = returns.rolling(20).std()
        # Expanding quantiles: threshold at bar t uses only bars 0..t, no future data.
        q33 = rolling_vol.expanding(min_periods=20).quantile(0.33)
        q66 = rolling_vol.expanding(min_periods=20).quantile(0.66)
        labels = pd.Series("medium", index=data.index)
        labels[rolling_vol <= q33] = "low_vol"
        labels[rolling_vol >= q66] = "high_vol"
        return labels

    @staticmethod
    def trend_regime(data: pd.DataFrame, sma_window: int = 50) -> pd.Series:
        """Classify by trend: above/below SMA."""
        sma = data["close"].rolling(sma_window).mean()
        labels = pd.Series("range", index=data.index)
        labels[data["close"] > sma * 1.02] = "uptrend"
        labels[data["close"] < sma * 0.98] = "downtrend"
        return labels

    @staticmethod
    def volume_regime(data: pd.DataFrame) -> pd.Series:
        """Classify by volume percentile."""
        vol_pct = (
            data["volume"]
            .rolling(50)
            .apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
        )
        labels = pd.Series("normal_volume", index=data.index)
        labels[vol_pct > 1] = "high_volume"
        labels[vol_pct < -1] = "low_volume"
        return labels

    def _build_subset_universe(
        self,
        universe: Universe,
        subset_idx: pd.Index,
    ) -> Universe:
        """
        Build a Universe containing only bars whose timestamps fall
        within ``subset_idx`` (derived from the reference asset's regime mask).
        """
        sub_universe = Universe(symbols=universe.symbols)

        for sym in universe.symbols:
            full_ohlcv = universe.ohlcv(sym)
            common = full_ohlcv.index.intersection(subset_idx)
            sub_ohlcv = full_ohlcv.loc[common].copy()
            if sub_ohlcv.empty:
                continue

            # Compute positional indices once for both L2 and funding.
            # get_indexer (not get_loc) always returns an integer array, safe
            # for non-unique indices where get_loc() would return slice/bool-array.
            raw_locs = full_ohlcv.index.get_indexer(common)
            idx_positions = [int(j) for j in raw_locs if j >= 0]

            full_l2 = universe.l2(sym)
            sub_l2 = None
            if full_l2 is not None:
                sub_l2 = [
                    full_l2[j] for j in idx_positions if j < len(full_l2)
                ]

            full_funding = universe.funding(sym)
            sub_funding = None
            if full_funding is not None:
                sub_funding = [
                    full_funding[j] for j in idx_positions if j < len(full_funding)
                ]

            sub_universe.add_asset(sym, sub_ohlcv, l2=sub_l2, funding=sub_funding)

        # Copy auxiliary data sources (they filter themselves by index)
        for src_name in universe.list_sources():
            sub_universe.add_data_source(universe._aux_sources[src_name])

        return sub_universe

    def _run_regime(
        self,
        strategy: Strategy,
        regime: str,
        sub_universe: Universe,
    ) -> tuple[str, int, BacktestResult | None, str | None]:
        bt = Backtester(
            strategy=copy.deepcopy(strategy),
            config=self.config,
            cost_model=self.cost_model,
        )
        n_bars = sub_universe.bar_count()
        try:
            res = bt.run(universe=sub_universe)
            return regime, n_bars, res, None
        except Exception as e:
            return regime, n_bars, None, str(e)

    def run(
        self,
        strategy: Strategy,
        universe: Universe,
    ) -> StressResult:
        ref_sym = self.regime_symbol or universe.symbols[0]
        ref_data = universe.ohlcv(ref_sym)

        regimes = self.regime_fn(ref_data)
        unique_regimes = regimes.dropna().unique()

        # Build subset universes up front (sequential — they share universe reads)
        regime_tasks: list[tuple[str, Universe]] = []
        for regime in unique_regimes:
            mask = regimes == regime
            subset_idx = ref_data.index[mask]
            if len(subset_idx) < 50:
                continue
            sub_universe = self._build_subset_universe(universe, subset_idx)
            if sub_universe.bar_count() < 50:
                continue
            regime_tasks.append((regime, sub_universe))

        rows: list[dict] = []
        results: dict[str, BacktestResult] = {}
        n_workers = os.cpu_count() if self.n_jobs == -1 else self.n_jobs

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(self._run_regime, strategy, regime, sub_u)
                for regime, sub_u in regime_tasks
            ]
            for future in futures:
                regime, n_bars, res, err = future.result()
                if res is not None:
                    rows.append({"regime": regime, "n_bars": n_bars, **res.summary()})
                    results[str(regime)] = res
                else:
                    rows.append({"regime": regime, "n_bars": n_bars, "error": err})

        return StressResult(
            name="regime_stress",
            summary=pd.DataFrame(rows),
            results=results,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Monte Carlo stress test
# ═══════════════════════════════════════════════════════════════════════════


class MonteCarloStress:
    """
    Bootstrap or shuffle trade returns to build confidence intervals.

    Usage:
        mc = MonteCarloStress(n_simulations=1000)
        result = mc.run(backtest_result)
    """

    def __init__(
        self, n_simulations: int = 1000, seed: int = 42, method: str = "bootstrap"
    ):
        self.n_simulations = n_simulations
        self.seed = seed
        self.method = method  # bootstrap | shuffle | block_bootstrap

    def run(self, bt_result: BacktestResult) -> StressResult:
        rng = np.random.default_rng(self.seed)
        tdf = bt_result.trades_df()
        if tdf.empty or "pnl" not in tdf.columns:
            return StressResult(name="monte_carlo", summary=pd.DataFrame())

        trade_pnls = tdf["pnl"].values
        initial = bt_result.config.initial_capital
        rows = []

        for i in range(self.n_simulations):
            if self.method == "bootstrap":
                sampled = rng.choice(trade_pnls, size=len(trade_pnls), replace=True)
            elif self.method == "shuffle":
                sampled = rng.permutation(trade_pnls)
            elif self.method == "block_bootstrap":
                block_size = max(5, len(trade_pnls) // 10)
                n_blocks = len(trade_pnls) // block_size + 1
                blocks = [
                    trade_pnls[j : j + block_size]
                    for j in range(0, len(trade_pnls), block_size)
                ]
                chosen = [blocks[rng.integers(len(blocks))] for _ in range(n_blocks)]
                sampled = np.concatenate(chosen)[: len(trade_pnls)]
            else:
                sampled = rng.choice(trade_pnls, size=len(trade_pnls), replace=True)

            cum_pnl = np.cumsum(sampled)
            equity = initial + cum_pnl
            total_ret = (equity[-1] / initial) - 1
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            max_dd = dd.min()

            rows.append(
                {
                    "sim_id": i,
                    "total_return_pct": total_ret * 100,
                    "max_drawdown_pct": max_dd * 100,
                    "final_equity": equity[-1],
                    "n_trades": len(sampled),
                }
            )

        summary = pd.DataFrame(rows)
        return StressResult(
            name="monte_carlo",
            summary=summary,
            meta={
                "median_return": summary["total_return_pct"].median(),
                "5th_pctl_return": summary["total_return_pct"].quantile(0.05),
                "95th_pctl_return": summary["total_return_pct"].quantile(0.95),
                "median_max_dd": summary["max_drawdown_pct"].median(),
                "5th_pctl_max_dd": summary["max_drawdown_pct"].quantile(0.05),
            },
        )

    def plot_distribution(
        self,
        stress_result: StressResult,
        metric: str = "total_return_pct",
        save_path: str | None = None,
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = stress_result.summary[metric]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(vals, bins=50, alpha=0.7, color="#2563eb", edgecolor="white")
        ax.axvline(
            vals.median(),
            color="#dc2626",
            linestyle="--",
            label=f"Median: {vals.median():.2f}",
        )
        ax.axvline(
            vals.quantile(0.05),
            color="#f59e0b",
            linestyle="--",
            label=f"5th pctl: {vals.quantile(0.05):.2f}",
        )
        ax.axvline(
            vals.quantile(0.95),
            color="#10b981",
            linestyle="--",
            label=f"95th pctl: {vals.quantile(0.95):.2f}",
        )
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.set_title(f"Monte Carlo Distribution — {metric}")
        ax.legend()
        fig.tight_layout()
        path = save_path or f"mc_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
