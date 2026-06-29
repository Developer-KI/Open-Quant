"""
stress.py — Modular stress testing framework.

Five types of stress tests:
  1. SignalStressTest      — sweep signal parameters, find fragile combos
  2. CostStressTest        — sweep cost assumptions (fees, slippage, impact)
  3. RegimeStressTest      — test across market regime subsets (vol, trend, etc.)
  4. MonteCarloStress      — bootstrap / shuffle trades to build confidence intervals
  5. StrategyStressTest    — sweep strategy parameters (multi-asset analogue of SignalStressTest)

All tests return a StressResult with a summary DataFrame and optional plots.

CostStressTest and RegimeStressTest accept either:
  • signal + data + l2    (old single-asset API, backwards compatible)
  • strategy + universe   (new multi-asset API)
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.models import BacktestConfig, OrderBookSnapshot
from .costs import CostModel, CompositeCostModel
from .engine import Backtester, BacktestResult

from strategy.base import Signal, Strategy
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
# 1. Signal parameter stress test
# ═══════════════════════════════════════════════════════════════════════════


class SignalStressTest:
    """
    Grid/random sweep over signal parameters.

    Usage:
        sst = SignalStressTest(
            signal_cls=EMACrossoverSignal,
            param_grid={"fast": [5, 8, 12, 20], "slow": [21, 26, 50]},
        )
        result = sst.run(data)
        result.plot_heatmap("fast", "slow")
    """

    def __init__(
        self,
        signal_cls: type[Signal],
        param_grid: dict[str, list],
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        fixed_params: dict | None = None,
        n_random: int | None = None,
        seed: int = 42,
    ):
        self.signal_cls = signal_cls
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
        data: pd.DataFrame,
        l2: list[OrderBookSnapshot] | None = None,
        timeframe: str | None = None,
    ) -> StressResult:
        combos = self._build_combos()
        rows = []
        results = {}

        for combo in combos:
            params = {**self.fixed_params, **combo}
            sig = self.signal_cls(**params)
            bt = Backtester(signal=sig, config=self.config, cost_model=self.cost_model)

            try:
                res = bt.run(data, l2, timeframe=timeframe)
                summary = res.summary()
                key = str(combo)
                rows.append({**combo, **summary})
                results[key] = res
            except Exception as e:
                rows.append({**combo, "error": str(e)})

        return StressResult(
            name="signal_sweep",
            summary=pd.DataFrame(rows),
            results=results,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Transaction cost stress test
# ═══════════════════════════════════════════════════════════════════════════


class CostStressTest:
    """
    Sweep transaction cost parameters to find where alpha breaks down.

    Accepts either the old single-asset API or the new multi-asset API:

      Old API (positional, backwards compatible):
          cst = CostStressTest(cost_grid={...})
          result = cst.run(signal, data)
          result = cst.run(signal, data, l2)

      New API (keyword):
          result = cst.run(strategy=my_strategy, universe=my_universe)

      Hybrid (signal over universe):
          result = cst.run(signal=my_signal, universe=my_universe)
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
        signal: Signal | None = None,
        data: pd.DataFrame | None = None,
        l2: list[OrderBookSnapshot] | None = None,
        *,
        strategy: Strategy | None = None,
        universe: Universe | None = None,
    ) -> StressResult:
        """
        Run cost sweep.

        Backwards-compatible positional call:
            result = cst.run(signal, data)
            result = cst.run(signal, data, l2)

        New keyword API:
            result = cst.run(strategy=strat, universe=univ)

        Hybrid:
            result = cst.run(signal=sig, universe=univ)
        """
        if signal is None and strategy is None:
            raise ValueError("Provide either signal= or strategy=")

        combos = self._build_cost_combos()
        rows = []
        results = {}

        for combo in combos:
            cost_model = self.base_cost_model.with_overrides(**combo)

            if strategy is not None:
                bt = Backtester(
                    strategy=copy.deepcopy(strategy),
                    config=self.config,
                    cost_model=cost_model,
                )
            else:
                bt = Backtester(
                    signal=copy.deepcopy(signal),
                    config=self.config,
                    cost_model=cost_model,
                )

            try:
                res = _run_backtester(bt, data=data, l2=l2, universe=universe)
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

    Accepts either the old single-asset API or the new multi-asset API.

    For multi-asset (strategy + universe) mode the regime_fn receives
    a reference DataFrame.  By default the first asset's OHLCV is used;
    override with ``regime_symbol`` to pick a different asset.

      Old API (positional, backwards compatible):
          rst = RegimeStressTest(regime_fn=my_classifier)
          result = rst.run(signal, data)

      New API (keyword):
          rst = RegimeStressTest(regime_fn=my_classifier, regime_symbol="BTC")
          result = rst.run(strategy=my_strategy, universe=my_universe)

      Hybrid:
          result = rst.run(signal=sig, universe=univ)
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
        signal: Signal | None = None,
        data: pd.DataFrame | None = None,
        l2: list[OrderBookSnapshot] | None = None,
        *,
        strategy: Strategy | None = None,
        universe: Universe | None = None,
    ) -> StressResult:
        """
        Run regime stress test.

        Backwards-compatible positional call:
            result = rst.run(signal, data)
            result = rst.run(signal, data, l2)

        New keyword API:
            result = rst.run(strategy=strat, universe=univ)

        Hybrid:
            result = rst.run(signal=sig, universe=univ)
        """
        if signal is None and strategy is None:
            raise ValueError("Provide either signal= or strategy=")

        is_multi_asset = universe is not None

        # ── Determine the reference DataFrame for regime classification ──
        if is_multi_asset:
            ref_sym = self.regime_symbol or universe.symbols[0]
            ref_data = universe.ohlcv(ref_sym)
        else:
            ref_data = data

        regimes = self.regime_fn(ref_data)
        unique_regimes = regimes.dropna().unique()
        rows = []
        results = {}

        for regime in unique_regimes:
            mask = regimes == regime

            if is_multi_asset:
                subset_idx = ref_data.index[mask]
                if len(subset_idx) < 50:
                    continue

                sub_universe = self._build_subset_universe(universe, subset_idx)
                if sub_universe.bar_count() < 50:
                    continue

                if strategy is not None:
                    bt = Backtester(
                        strategy=copy.deepcopy(strategy),
                        config=self.config,
                        cost_model=self.cost_model,
                    )
                else:
                    bt = Backtester(
                        signal=copy.deepcopy(signal),
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

            else:
                # ── Old single-asset path (unchanged) ────────────────────
                subset = data[mask].copy()
                if len(subset) < 50:
                    continue

                l2_sub = None
                if l2 is not None:
                    l2_sub = [l2[i] for i, m in enumerate(mask) if m and i < len(l2)]

                sig = copy.deepcopy(signal)
                bt = Backtester(
                    signal=sig, config=self.config, cost_model=self.cost_model,
                )

                try:
                    res = bt.run(subset, l2_sub)
                    summary = res.summary()
                    rows.append({"regime": regime, "n_bars": len(subset), **summary})
                    results[str(regime)] = res
                except Exception as e:
                    rows.append({
                        "regime": regime, "n_bars": len(subset), "error": str(e),
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