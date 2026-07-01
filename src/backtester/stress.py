"""
stress.py — Modular stress testing framework.

Five types of stress tests:
  1. ParamSweep       — sweep SingleAssetStrategy parameters (grid or random)
  2. CostStressTest   — sweep cost assumptions (fees, slippage, impact)
  3. RegimeStressTest — test across market regime subsets (vol, trend, etc.)
  4. MonteCarloStress — bootstrap / shuffle trades to build confidence intervals
  5. StrategyStressTest — sweep Strategy parameters (multi-asset)

All tests return a StressResult with a summary DataFrame and optional plots.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.models import BacktestConfig
from .costs import CostModel, CompositeCostModel
from .engine import Backtester, BacktestResult

from strategy.base import SingleAssetStrategy, Strategy
from strategy.universe import Universe


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


# ── Helpers ──────────────────────────────────────────────────────────────────
def _run_backtester(
    bt: Backtester,
    *,
    data: pd.DataFrame | None = None,
    l2: list[OrderBookSnapshot] | None = None,
    universe: Universe | None = None,
) -> BacktestResult:
    """Run a Backtester against whichever data source was provided."""
    if universe is not None:
        return bt.run(universe=universe)
    return bt.run(data, l2)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Parameter sweep (single-asset strategy)
# ═══════════════════════════════════════════════════════════════════════════


class ParamSweep:
    """
    Grid/random sweep over SingleAssetStrategy parameters.

    Usage:
        sweep = ParamSweep(
            strategy_cls=EMACrossStrategy,
            param_grid={"fast": [5, 8, 12, 20], "slow": [21, 26, 50]},
        )
        result = sweep.run(universe=universe, timeframe="1h")
        result.plot_heatmap("fast", "slow")
    """

    def __init__(
        self,
        strategy_cls: type[SingleAssetStrategy],
        param_grid: dict[str, list],
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        fixed_params: dict | None = None,
        n_random: int | None = None,
        seed: int = 42,
    ):
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.config = config or BacktestConfig()
        self.cost_model = cost_model
        self.fixed_params = fixed_params or {}
        self.n_random = n_random
        self.seed = seed

    def _build_combos(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        vals = list(self.param_grid.values())
        combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        if self.n_random and self.n_random < len(combos):
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(combos), size=self.n_random, replace=False)
            combos = [combos[i] for i in idx]
        return combos

    def run(
        self,
        universe: Universe,
        timeframe: str | None = None,
    ) -> StressResult:
        combos = self._build_combos()
        rows = []
        results = {}
        sym = universe.symbols[0] if universe.symbols else "ASSET"

        for combo in combos:
            params = {**self.fixed_params, **combo}
            strategy = self.strategy_cls(symbol=sym, **params)
            bt = Backtester(strategy=strategy, config=self.config, cost_model=self.cost_model)

            try:
                res = bt.run(universe=universe, timeframe=timeframe)
                summary = res.summary()
                key = str(combo)
                rows.append({**combo, **summary})
                results[key] = res
            except Exception as e:
                rows.append({**combo, "error": str(e)})

        return StressResult(
            name="param_sweep",
            summary=pd.DataFrame(rows),
            results=results,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Transaction cost stress test
# ═══════════════════════════════════════════════════════════════════════════


class CostStressTest:
    """
    Sweep transaction cost parameters to find where alpha breaks down.

    Usage:
        cst = CostStressTest(cost_grid={...})
        result = cst.run(strategy=my_strategy, universe=my_universe)
    """

    def __init__(
        self,
        cost_grid: dict[str, dict[str, list]],
        base_cost_model: CompositeCostModel | None = None,
        config: BacktestConfig | None = None,
    ):
        self.cost_grid = cost_grid
        self.base_cost_model = base_cost_model or CompositeCostModel()
        self.config = config or BacktestConfig()

    def _build_cost_combos(self) -> list[dict[str, dict[str, Any]]]:
        """Build all combinations across cost model parameters."""
        all_keys = []
        all_vals = []
        key_map = []  # (model_name, param_name)

        for model_name, params in self.cost_grid.items():
            for param_name, values in params.items():
                all_keys.append(f"{model_name}.{param_name}")
                all_vals.append(values)
                key_map.append((model_name, param_name))

        combos = []
        for combo_vals in itertools.product(*all_vals):
            override = {}
            for (model_name, param_name), val in zip(key_map, combo_vals):
                if model_name not in override:
                    override[model_name] = {}
                override[model_name][param_name] = val
            combos.append(override)
        return combos

    def run(
        self,
        strategy: Strategy,
        universe: Universe,
    ) -> StressResult:
        combos = self._build_cost_combos()
        rows = []
        results = {}

        for combo in combos:
            cost_model = self.base_cost_model.with_overrides(**combo)
            bt = Backtester(
                strategy=copy.deepcopy(strategy),
                config=self.config,
                cost_model=cost_model,
            )

            try:
                res = bt.run(universe=universe)
                summary = res.summary()
                flat_combo = {}
                for mn, ps in combo.items():
                    for pn, pv in ps.items():
                        flat_combo[f"{mn}__{pn}"] = pv
                key = str(flat_combo)
                rows.append({**flat_combo, **summary})
                results[key] = res
            except Exception as e:
                flat_combo = {}
                for mn, ps in combo.items():
                    for pn, pv in ps.items():
                        flat_combo[f"{mn}__{pn}"] = pv
                rows.append({**flat_combo, "error": str(e)})

        return StressResult(
            name="cost_sweep",
            summary=pd.DataFrame(rows),
            results=results,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Regime-based stress test
# ═══════════════════════════════════════════════════════════════════════════


class RegimeStressTest:
    """
    Split data by market regime and run backtest on each subset.

    Usage:
        rst = RegimeStressTest(regime_fn=my_classifier, regime_symbol="BTC")
        result = rst.run(strategy=my_strategy, universe=my_universe)
    """

    def __init__(
        self,
        regime_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        regime_symbol: str | None = None,
    ):
        self.regime_fn = regime_fn or self._default_vol_regime
        self.config = config or BacktestConfig()
        self.cost_model = cost_model
        self.regime_symbol = regime_symbol

    @staticmethod
    def _default_vol_regime(data: pd.DataFrame) -> pd.Series:
        """Classify bars into low/medium/high volatility regimes."""
        returns = data["close"].pct_change()
        rolling_vol = returns.rolling(20).std()
        terciles = rolling_vol.quantile([0.33, 0.66])
        labels = pd.Series("medium", index=data.index)
        labels[rolling_vol <= terciles.iloc[0]] = "low_vol"
        labels[rolling_vol >= terciles.iloc[1]] = "high_vol"
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

            # Compute positional indices once for both L2 and funding
            idx_positions = [
                full_ohlcv.index.get_loc(ts)
                for ts in common
                if ts in full_ohlcv.index
            ]

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

    def run(
        self,
        strategy: Strategy,
        universe: Universe,
    ) -> StressResult:
        ref_sym = self.regime_symbol or universe.symbols[0]
        ref_data = universe.ohlcv(ref_sym)

        regimes = self.regime_fn(ref_data)
        unique_regimes = regimes.dropna().unique()
        rows = []
        results = {}

        for regime in unique_regimes:
            mask = regimes == regime
            subset_idx = ref_data.index[mask]
            if len(subset_idx) < 50:
                continue

            sub_universe = self._build_subset_universe(universe, subset_idx)
            if sub_universe.bar_count() < 50:
                continue

            bt = Backtester(
                strategy=copy.deepcopy(strategy),
                config=self.config,
                cost_model=self.cost_model,
            )

            try:
                res = bt.run(universe=sub_universe)
                summary = res.summary()
                rows.append({
                    "regime": regime,
                    "n_bars": sub_universe.bar_count(),
                    **summary,
                })
                results[str(regime)] = res
            except Exception as e:
                rows.append({
                    "regime": regime,
                    "n_bars": sub_universe.bar_count(),
                    "error": str(e),
                })

        return StressResult(
            name="regime_stress",
            summary=pd.DataFrame(rows),
            results=results,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Monte Carlo stress test
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


# ═══════════════════════════════════════════════════════════════════════════
# 5. Strategy parameter stress test
# ═══════════════════════════════════════════════════════════════════════════


class StrategyStressTest:
    """
    Grid/random sweep over Strategy parameters (multi-asset analogue of
    SignalStressTest).

    The strategy_cls is instantiated fresh for each parameter combo, so
    the sweep covers constructor arguments.  Use ``fixed_params`` for
    arguments that should stay constant across all runs.

    Usage:
        sst = StrategyStressTest(
            strategy_cls=ZPairsSpreadStrategy,
            param_grid={
                "lookback": [30, 60, 120],
                "entry_z": [1.5, 2.0, 2.5],
                "exit_z": [0.3, 0.5, 1.0],
            },
        )
        result = sst.run(universe=my_universe)
        result.plot_heatmap("lookback", "entry_z")

    For strategies that require non-serialisable constructor args (e.g.
    Signal instances), pass them in ``fixed_params``:

        sst = StrategyStressTest(
            strategy_cls=PerAssetSignalStrategy,
            param_grid={"some_threshold": [0.3, 0.5]},
            fixed_params={"signals": {"ETH": eth_sig, "BTC": btc_sig}},
        )
    """

    def __init__(
        self,
        strategy_cls: type,
        param_grid: dict[str, list],
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        sizer=None,
        stop_loss=None,
        fixed_params: dict | None = None,
        n_random: int | None = None,
        seed: int = 42,
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

    def _build_combos(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        vals = list(self.param_grid.values())
        combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        if self.n_random and self.n_random < len(combos):
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(combos), size=self.n_random, replace=False)
            combos = [combos[i] for i in idx]
        return combos

    def run(
        self,
        universe: Universe,
        bars_per_year: int | None = None,
    ) -> StressResult:
        """
        Run parameter sweep over the strategy.

        Args:
            universe:       Multi-asset Universe containing all OHLCV + aux data.
            bars_per_year:  Optional override for annualisation factor.

        Returns:
            StressResult with one row per parameter combination.
        """
        combos = self._build_combos()
        rows = []
        results = {}

        for combo in combos:
            params = {**self.fixed_params, **combo}
            strat = self.strategy_cls(**params)

            bt = Backtester(
                strategy=strat,
                config=self.config,
                cost_model=self.cost_model,
                sizer=self.sizer,
                stop_loss=self.stop_loss,
            )

            try:
                res = bt.run(
                    universe=copy.deepcopy(universe),
                    bars_per_year=bars_per_year,
                )
                summary = res.summary()
                key = str(combo)
                rows.append({**combo, **summary})
                results[key] = res
            except Exception as e:
                rows.append({**combo, "error": str(e)})

        return StressResult(
            name="strategy_sweep",
            summary=pd.DataFrame(rows),
            results=results,
        )