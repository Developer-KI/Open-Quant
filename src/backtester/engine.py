"""
strategy/engine.py — Unified backtester engine.

Drop-in replacement for backtester.engine.Backtester.
Accepts EITHER the old API or the new multi-asset API:

  Old (single asset):
      bt = Backtester(signal=my_signal)
      result = bt.run(data=eth_df, l2=snapshots)

  New (multi asset):
      bt = Backtester(strategy=my_strategy)
      result = bt.run(universe=universe)

  Hybrid (signal auto-wrapped):
      bt = Backtester(signal=my_signal, symbol="ETH")
      result = bt.run(universe=universe)

The returned BacktestResult is identical in both cases — same .summary(),
.trades_df(), .plot_equity(), .to_csv() interface as the original engine.
Multi-asset runs get extra fields (positions_log, allocation_log,
trades_by_symbol) on top of the standard ones.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.models import (
    BacktestConfig,
    OrderBookSnapshot,
    Position,
    Side,
    Trade,
)
from core.parser import timeframe_to_bars_per_year, TIMEFRAMES
from backtester.costs import CostModel, CompositeCostModel
from risk.sizing import Sizer, SizingContext, default_sizer
from risk.stops import (
    StopLoss,
    StopContext,
    SignalStop,
    default_stop_loss,
)

from strategy.base import Signal, Strategy, StrategyContext
from strategy.built_in import SingleSignalStrategy
from strategy.universe import Universe


# ── Result container (superset of old BacktestResult) ────────────────────────


@dataclass
class BacktestResult:
    """
    Backward-compatible result container.

    Has every field the old BacktestResult had, plus optional multi-asset
    extras.  Code that used the old result object works unchanged.
    """

    trades: list[Trade]
    equity_curve: pd.Series
    positions: pd.Series                            # side per bar (single-asset compat)
    signal_log: pd.DataFrame                        # per-bar signal/allocation values
    config: BacktestConfig
    run_time_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    # ── multi-asset extras (None when single-asset) ──────────────────────
    positions_log: pd.DataFrame | None = None       # per-bar, per-asset positions
    allocation_log: pd.DataFrame | None = None      # per-bar strategy targets

    # ── export helpers ───────────────────────────────────────────────────

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def trades_by_symbol(self, symbol: str) -> pd.DataFrame:
        """Filter trades to one symbol (multi-asset runs)."""
        df = self.trades_df()
        if df.empty or "meta_symbol" not in df.columns:
            return df
        return df[df["meta_symbol"] == symbol]

    def to_csv(self, path: str = "trades.csv"):
        df = self.trades_df()
        df.to_csv(path, index=False)
        return path

    # ── analytics (identical to old engine) ───────────────────────────────

    def summary(self) -> dict[str, Any]:
        eq = self.equity_curve
        returns = eq.pct_change().dropna()
        tdf = self.trades_df()

        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1 if len(eq) > 1 else 0
        n_bars = len(eq)

        ann_factor = 365 * 24 if n_bars > 1 else 1
        if "bars_per_year" in self.meta:
            ann_factor = self.meta["bars_per_year"]

        ann_return = (1 + total_return) ** (ann_factor / max(n_bars, 1)) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor) if len(returns) > 1 else 0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = dd.min()

        if len(tdf) > 0 and "pnl" in tdf.columns:
            wins = (tdf["pnl"] > 0).sum()
            win_rate = wins / len(tdf)
            avg_win = tdf.loc[tdf["pnl"] > 0, "pnl"].mean() if wins > 0 else 0
            avg_loss = (
                tdf.loc[tdf["pnl"] <= 0, "pnl"].mean()
                if (len(tdf) - wins) > 0
                else 0
            )
            profit_factor = (
                abs(avg_win * wins / (avg_loss * (len(tdf) - wins)))
                if avg_loss != 0
                else np.inf
            )
            total_fees = tdf["fees"].sum()
        else:
            win_rate = avg_win = avg_loss = profit_factor = total_fees = 0

        calmar = abs(ann_return / max_dd) if max_dd != 0 else 0
        sortino_vol = (
            returns[returns < 0].std() * np.sqrt(ann_factor)
            if (returns < 0).any()
            else 0
        )
        sortino = ann_return / sortino_vol if sortino_vol > 0 else 0

        result = {
            "total_return_pct": round(total_return * 100, 4),
            "annualised_return_pct": round(ann_return * 100, 4),
            "annualised_volatility_pct": round(ann_vol * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "num_trades": len(tdf),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "total_fees": round(total_fees, 4),
            "run_time_s": round(self.run_time_s, 3),
        }

        # Extra field for multi-asset
        symbols = self.meta.get("symbols")
        if symbols and len(symbols) > 1:
            result["symbols_traded"] = list(
                tdf["meta_symbol"].unique()
            ) if "meta_symbol" in tdf.columns else symbols

        return result

    def plot_equity(self, save_path: str | None = None, show: bool = False):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        eq = self.equity_curve
        peak = eq.cummax()
        dd = (eq - peak) / peak

        ax1.plot(eq.index, eq.values, linewidth=1.2, color="#2563eb", label="Equity")
        ax1.fill_between(eq.index, eq.values, alpha=0.08, color="#2563eb")
        ax1.set_ylabel("Equity")
        ax1.set_title("Equity Curve")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(dd.index, dd.values, 0, color="#dc2626", alpha=0.4)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = save_path or "equity_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def save(self, run_name: str, base_dir: str = "logs/test") -> str:
        """Save log.json, trades.csv, equity_curve.png, and signal_log.csv to logs/test/<run_name>/."""
        import dataclasses
        from datetime import datetime, timezone

        run_dir = os.path.join(base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        summary = self.summary()
        config_dict = dataclasses.asdict(self.config)

        eq = self.equity_curve
        data_range: dict[str, Any] = {"bars": len(eq)}
        if isinstance(eq.index, pd.DatetimeIndex) and len(eq) > 0:
            data_range["start"] = str(eq.index[0])
            data_range["end"] = str(eq.index[-1])

        meta_out: dict[str, Any] = {}
        for k, v in self.meta.items():
            if isinstance(v, (list, tuple)):
                meta_out[k] = list(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                meta_out[k] = v
            else:
                meta_out[k] = str(v)

        log_data = {
            "run_name": run_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "data_range": data_range,
            "summary": summary,
            "config": config_dict,
            "meta": meta_out,
        }
        with open(os.path.join(run_dir, "log.json"), "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        self.to_csv(os.path.join(run_dir, "trades.csv"))
        self.plot_equity(save_path=os.path.join(run_dir, "equity_curve.png"))

        if self.signal_log is not None and not self.signal_log.empty:
            self.signal_log.to_csv(os.path.join(run_dir, "signal_log.csv"), index=False)

        return run_dir


# ── Per-asset state ──────────────────────────────────────────────────────────


@dataclass
class _AssetState:
    position: Position = field(default_factory=Position)
    stop_loss: StopLoss | None = None
    open_trade: Trade | None = None


# ── Unified Backtester ───────────────────────────────────────────────────────


class Backtester:
    """
    Drop-in replacement for backtester.engine.Backtester.

    Accepts EITHER:
      • signal + (optional symbol) — old single-asset API
      • strategy                   — new multi-asset API

    The run() method accepts EITHER:
      • data + l2                  — old API (single DataFrame)
      • universe                   — new API (multi-asset Universe)

    Components (sizer, stop_loss, cost_model) can be:
      • A single instance           — shared across all assets
      • A dict[symbol, instance]    — per-asset overrides
    """

    def __init__(
        self,
        signal: Signal | None = None,
        strategy: Strategy | None = None,
        config: BacktestConfig | None = None,
        cost_model: CostModel | dict[str, CostModel] | None = None,
        sizer: Sizer | dict[str, Sizer] | None = None,
        stop_loss: StopLoss | dict[str, StopLoss] | None = None,
        symbol: str | None = None,
    ):
        if signal is None and strategy is None:
            raise ValueError("Provide either signal= or strategy=")
        if signal is not None and strategy is not None:
            raise ValueError("Provide signal= or strategy=, not both")

        self.config = config or BacktestConfig()
        self._cost_model_spec = cost_model
        self._sizer_spec = sizer
        self._stop_loss_spec = stop_loss

        # If given a Signal, store it so run() can auto-wrap it
        self._signal = signal
        self._strategy = strategy
        self._default_symbol = symbol or "ASSET"

    # ── Component resolution ─────────────────────────────────────────────

    def _resolve(self, spec, symbol, default_fn):
        if isinstance(spec, dict):
            return copy.deepcopy(spec.get(symbol, default_fn()))
        elif spec is not None:
            return copy.deepcopy(spec)
        return default_fn()

    # ── Public API ───────────────────────────────────────────────────────

    def run(
        self,
        data: pd.DataFrame | None = None,
        l2: list[OrderBookSnapshot] | None = None,
        universe: Universe | None = None,
        timeframe: str | None = None,
        bars_per_year: int | None = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Old API:  result = bt.run(data=df, l2=snapshots)
        New API:  result = bt.run(universe=universe, timeframe="1h")

        timeframe     — bar size label (e.g. "1m", "5m", "1h", "1d").
                        Used to compute bars_per_year for annualisation.
                        Takes precedence over bars_per_year when both supplied.
        bars_per_year — explicit override; ignored when timeframe is set.
        """
        # ── Resolve strategy + universe from whatever was provided ────────

        if data is not None and universe is not None:
            raise ValueError("Provide data= or universe=, not both")

        if data is not None:
            # Old API: single DataFrame → wrap into Universe
            sym = self._default_symbol
            universe = Universe(symbols=[sym])
            universe.add_asset(sym, data, l2=l2)

            if self._signal is not None:
                strategy = SingleSignalStrategy(
                    signal=self._signal, symbol=sym,
                )
            else:
                strategy = self._strategy
        elif universe is not None:
            if self._signal is not None:
                # Signal + Universe → wrap signal for the first/specified symbol
                sym = self._default_symbol
                if sym == "ASSET" and universe.symbols:
                    sym = universe.symbols[0]
                strategy = SingleSignalStrategy(
                    signal=self._signal, symbol=sym,
                )
            else:
                strategy = self._strategy
        else:
            raise ValueError("Provide either data= or universe=")

        symbols = universe.symbols
        is_single_asset = len(symbols) == 1

        # timeframe takes precedence over the legacy bars_per_year param
        resolved_bpy = bars_per_year
        if timeframe is not None:
            resolved_bpy = timeframe_to_bars_per_year(timeframe)

        return self._run_loop(
            strategy=strategy,
            universe=universe,
            symbols=symbols,
            is_single_asset=is_single_asset,
            bars_per_year=resolved_bpy,
            timeframe=timeframe,
        )

    # ── Core loop ────────────────────────────────────────────────────────

    def _run_loop(
        self,
        strategy: Strategy,
        universe: Universe,
        symbols: list[str],
        is_single_asset: bool,
        bars_per_year: int | None,
        timeframe: str | None = None,
    ) -> BacktestResult:
        t0 = time.perf_counter()
        n_bars = universe.bar_count()
        if n_bars == 0:
            raise ValueError("No data in universe")

        # Use common index for multi-asset alignment
        if is_single_asset:
            index = universe.ohlcv(symbols[0]).index
        else:
            index = universe.common_index()
            if len(index) == 0:
                index = universe.ohlcv(symbols[0]).index
        n_bars = len(index)

        # Infer annualisation
        if bars_per_year is None:
            if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
                median_delta = index.to_series().diff().median()
                secs = median_delta.total_seconds()
                bars_per_year = int(365 * 24 * 3600 / secs) if secs > 0 else 8760
            else:
                bars_per_year = 8760

        # Setup strategy (vectorised indicators)
        strategy.setup(universe)

        # Per-asset state
        states: dict[str, _AssetState] = {}
        sizers: dict[str, Sizer] = {}
        cost_models: dict[str, CostModel] = {}
        for sym in symbols:
            states[sym] = _AssetState(
                stop_loss=self._resolve(
                    self._stop_loss_spec, sym, default_stop_loss,
                ),
            )
            sizers[sym] = self._resolve(self._sizer_spec, sym, default_sizer)
            cost_models[sym] = self._resolve(
                self._cost_model_spec, sym, CompositeCostModel,
            )

        # Pre-allocate
        equity_arr = np.full(n_bars, np.nan)
        equity = self.config.initial_capital
        equity_arr[0] = equity

        # For single-asset backward compat: track position side per bar
        pos_side_arr = np.zeros(n_bars, dtype=int) if is_single_asset else None

        all_trades: list[Trade] = []
        closed_trades: list[Trade] = []
        signal_log_rows: list[dict] = []
        alloc_log_rows: list[dict] = []
        pos_log_rows: list[dict] = []

        # ── Bar loop ─────────────────────────────────────────────────────

        for i in range(n_bars):
            ts = index[i]

            # Resolve bar-level data for each asset
            prices: dict[str, float] = {}
            bar_dicts: dict[str, dict] = {}
            bar_locs: dict[str, int] = {}  # asset-local index

            for sym in symbols:
                ohlcv = universe.ohlcv(sym)
                if ts in ohlcv.index:
                    loc = ohlcv.index.get_loc(ts)
                    # get_loc can return a slice for duplicates; take first
                    if isinstance(loc, slice):
                        loc = loc.start
                    prices[sym] = ohlcv["close"].iat[loc]
                    bar_dicts[sym] = {
                        c: ohlcv[c].iat[loc] for c in ohlcv.columns
                        if np.isscalar(ohlcv[c].iat[loc])
                    }
                    bar_locs[sym] = loc

                    # Inject funding rate if available
                    funding_snap = universe.funding_at(sym, loc)
                    if funding_snap is not None:
                        bar_dicts[sym]["funding_rate"] = funding_snap.rate
                        bar_dicts[sym]["funding_rate_ann_bps"] = funding_snap.rate_annualized
                        if funding_snap.oracle_price > 0:
                            bar_dicts[sym]["oracle_price"] = funding_snap.oracle_price
                        if funding_snap.mark_price > 0:
                            bar_dicts[sym]["mark_price"] = funding_snap.mark_price

            # ── Mark-to-market ───────────────────────────────────────────
            for sym in symbols:
                st = states[sym]
                pos = st.position
                if pos.side != Side.FLAT and pos.size > 0 and sym in prices:
                    direction = 1 if pos.side == Side.LONG else -1
                    pos.unrealized_pnl = (
                        (prices[sym] - pos.entry_price) * pos.size * direction
                    )

            # ── Stop-loss checks ─────────────────────────────────────────
            for sym in symbols:
                st = states[sym]
                pos = st.position
                if pos.side == Side.FLAT or sym not in prices:
                    continue

                loc = bar_locs.get(sym)
                if loc is None:
                    continue

                ohlcv = universe.ohlcv(sym)
                l2_list = universe.l2(sym)
                l2_snap = l2_list[loc] if l2_list and loc < len(l2_list) else None

                stop_ctx = StopContext(
                    position=pos,
                    bar_idx=loc,
                    open=ohlcv["open"].iat[loc],
                    high=ohlcv["high"].iat[loc],
                    low=ohlcv["low"].iat[loc],
                    close=prices[sym],
                    data=ohlcv,
                    l2=l2_snap,
                    bar_data=bar_dicts.get(sym, {}),
                )
                st.stop_loss.update(stop_ctx)
                stop_result = st.stop_loss.check(stop_ctx)

                # SignalStop backward compat (single-asset only)
                if (
                    not stop_result.triggered
                    and is_single_asset
                    and isinstance(st.stop_loss, SignalStop)
                    and self._signal is not None
                ):
                    sig_peek = self._signal.generate(ohlcv, loc)
                    st.stop_loss.set_levels(sig_peek.stop_loss, sig_peek.take_profit)
                    stop_result = st.stop_loss.check_with_levels(stop_ctx)

                if stop_result.triggered:
                    exit_p = stop_result.exit_price
                    cost = cost_models[sym].compute(
                        exit_p, pos.size, pos.side, self.config,
                        l2_snap, bar_dicts.get(sym, {}),
                    )
                    raw_pnl = (
                        (exit_p - pos.entry_price) * pos.size
                        if pos.side == Side.LONG
                        else (pos.entry_price - exit_p) * pos.size
                    )
                    pnl = raw_pnl - cost
                    equity += pnl

                    if st.open_trade is not None:
                        st.open_trade.exit_price = exit_p
                        st.open_trade.exit_timestamp = ts
                        st.open_trade.pnl = pnl
                        st.open_trade.pnl_pct = (
                            pnl / (pos.entry_price * pos.size)
                            if pos.entry_price * pos.size > 0 else 0
                        )
                        st.open_trade.fees += cost
                        st.open_trade.reason_exit = stop_result.reason
                        st.open_trade.meta.update(stop_result.meta)
                        if not is_single_asset:
                            st.open_trade.meta["symbol"] = sym
                        closed_trades.append(st.open_trade)

                    st.position = Position()
                    st.open_trade = None
                    st.stop_loss.reset()

            # ── Generate strategy targets ────────────────────────────────
            ctx = StrategyContext(
                universe=universe,
                bar_idx=i,
                timestamp=ts,
                equity=equity,
                positions={sym: states[sym].position for sym in symbols},
                trade_history=closed_trades,
            )
            target = strategy.generate(ctx)

            # Build signal log row (backward-compat for single asset)
            if is_single_asset:
                sym0 = symbols[0]
                alloc = target[sym0]
                signal_log_rows.append({
                    "timestamp": ts,
                    "side": alloc.side.name,
                    "weight": alloc.weight,
                    "confidence": alloc.confidence,
                    "reason": alloc.reason,
                })
            else:
                row = {"timestamp": ts}
                for sym in symbols:
                    alloc = target[sym]
                    row[f"{sym}_side"] = alloc.side.name
                    row[f"{sym}_weight"] = alloc.weight
                    row[f"{sym}_confidence"] = alloc.confidence
                    row[f"{sym}_reason"] = alloc.reason
                alloc_log_rows.append(row)

            # ── Close positions that should be flat or flipped ────────────
            for sym in symbols:
                st = states[sym]
                pos = st.position
                if pos.side == Side.FLAT:
                    continue

                desired = target[sym]
                should_close = (
                    desired.side == Side.FLAT
                    or desired.side != pos.side
                )
                if not should_close or sym not in prices:
                    continue

                price = prices[sym]
                cost = cost_models[sym].compute(
                    price, pos.size, pos.side, self.config,
                    None, bar_dicts.get(sym, {}),
                )
                pnl = pos.unrealized_pnl - cost
                pnl_pct = (
                    pnl / (pos.entry_price * pos.size)
                    if pos.entry_price * pos.size > 0 else 0
                )

                trade_meta = {"symbol": sym} if not is_single_asset else {}
                trade = Trade(
                    timestamp=pos.entry_timestamp,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    exit_timestamp=ts,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    fees=cost,
                    reason_entry=(
                        st.open_trade.reason_entry if st.open_trade else ""
                    ),
                    reason_exit=desired.reason or "target_flat",
                    signal_values=desired.meta,
                    meta=trade_meta,
                )
                all_trades.append(trade)
                closed_trades.append(trade)
                equity += pnl

                st.position = Position()
                st.open_trade = None
                st.stop_loss.reset()

            # ── Open new positions ───────────────────────────────────────
            for sym in symbols:
                st = states[sym]
                if st.position.side != Side.FLAT:
                    continue

                alloc = target[sym]
                if alloc.side == Side.FLAT or alloc.weight <= 0:
                    continue
                if sym not in prices:
                    continue

                price = prices[sym]
                loc = bar_locs.get(sym, 0)
                ohlcv = universe.ohlcv(sym)
                sig_result = alloc.to_signal_result()

                l2_list = universe.l2(sym)
                l2_snap = l2_list[loc] if l2_list and loc < len(l2_list) else None

                sizing_ctx = SizingContext(
                    equity=equity,
                    price=price,
                    signal=sig_result,
                    config=self.config,
                    position=st.position,
                    data=ohlcv,
                    bar_idx=loc,
                    trade_history=closed_trades,
                    l2=l2_snap,
                    bar_data=bar_dicts.get(sym, {}),
                )
                size = sizers[sym].compute(sizing_ctx)

                # Cap at max position (use allocation weight for multi-asset)
                if is_single_asset:
                    max_notional = (
                        equity * self.config.max_position_pct * self.config.leverage
                    )
                else:
                    max_notional = (
                        equity * alloc.weight * self.config.leverage
                    )
                max_size = max_notional / price if price > 0 else 0
                size = min(size, max_size)

                if size <= 0:
                    continue

                cost = cost_models[sym].compute(
                    price, size, alloc.side, self.config,
                    l2_snap, bar_dicts.get(sym, {}),
                )

                st.position = Position(
                    side=alloc.side,
                    size=size,
                    entry_price=price,
                    entry_timestamp=ts,
                )
                equity -= cost

                # Initialize stop
                stop_ctx = StopContext(
                    position=st.position,
                    bar_idx=loc,
                    open=ohlcv["open"].iat[loc] if loc < len(ohlcv) else price,
                    high=ohlcv["high"].iat[loc] if loc < len(ohlcv) else price,
                    low=ohlcv["low"].iat[loc] if loc < len(ohlcv) else price,
                    close=price,
                    data=ohlcv,
                    l2=l2_snap,
                    bar_data=bar_dicts.get(sym, {}),
                )
                st.stop_loss.on_entry(st.position, stop_ctx)

                trade_meta = {"symbol": sym} if not is_single_asset else {}
                trade = Trade(
                    timestamp=ts,
                    side=alloc.side,
                    size=size,
                    entry_price=price,
                    fees=cost,
                    reason_entry=alloc.reason,
                    signal_values=alloc.meta,
                    meta=trade_meta,
                )
                all_trades.append(trade)
                st.open_trade = trade

            # ── Record state ─────────────────────────────────────────────
            unrealized = sum(
                st.position.unrealized_pnl
                for st in states.values()
                if st.position.side != Side.FLAT
            )
            equity_arr[i] = equity + unrealized

            if is_single_asset:
                pos_side_arr[i] = states[symbols[0]].position.side.value
            else:
                row = {"timestamp": ts}
                for sym in symbols:
                    st = states[sym]
                    row[f"{sym}_side"] = st.position.side.value
                    row[f"{sym}_size"] = st.position.size
                pos_log_rows.append(row)

        # ── Force-close remaining positions ──────────────────────────────
        last_ts = index[-1]
        for sym in symbols:
            st = states[sym]
            pos = st.position
            if pos.side == Side.FLAT:
                continue

            last_ohlcv = universe.ohlcv(sym)
            last_price = last_ohlcv["close"].iloc[-1]
            cost = cost_models[sym].compute(
                last_price, pos.size, pos.side, self.config, None, None,
            )
            raw_pnl = (
                (last_price - pos.entry_price) * pos.size
                if pos.side == Side.LONG
                else (pos.entry_price - last_price) * pos.size
            )
            pnl = raw_pnl - cost

            if st.open_trade is not None:
                st.open_trade.exit_price = last_price
                st.open_trade.exit_timestamp = last_ts
                st.open_trade.pnl = pnl
                st.open_trade.pnl_pct = (
                    pnl / (pos.entry_price * pos.size)
                    if pos.entry_price * pos.size > 0 else 0
                )
                st.open_trade.fees += cost
                st.open_trade.reason_exit = "End of data"
                if not is_single_asset:
                    st.open_trade.meta["symbol"] = sym
                closed_trades.append(st.open_trade)
            else:
                trade_meta = {"symbol": sym} if not is_single_asset else {}
                trade = Trade(
                    timestamp=pos.entry_timestamp,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    exit_price=last_price,
                    exit_timestamp=last_ts,
                    pnl=pnl,
                    pnl_pct=(
                        pnl / (pos.entry_price * pos.size)
                        if pos.entry_price * pos.size > 0 else 0
                    ),
                    fees=cost,
                    reason_exit="End of data",
                    meta=trade_meta,
                )
                all_trades.append(trade)
                closed_trades.append(trade)

        # ── Build result ─────────────────────────────────────────────────
        final_trades = [t for t in all_trades if t.exit_price is not None]
        elapsed = time.perf_counter() - t0

        eq_series = pd.Series(equity_arr, index=index, name="equity")

        # positions Series (backward compat for single-asset)
        if is_single_asset:
            pos_series = pd.Series(pos_side_arr, index=index, name="position")
        else:
            if pos_log_rows:
                pos_df = pd.DataFrame(pos_log_rows)
                side_cols = [f"{s}_side" for s in symbols]
                pos_series = pos_df[side_cols].sum(axis=1)
                pos_series.index = index
                pos_series.name = "position"
            else:
                pos_series = pd.Series(
                    np.zeros(n_bars, dtype=int), index=index, name="position",
                )

        # signal_log (backward compat: single-asset uses per-bar rows)
        if is_single_asset:
            sig_df = pd.DataFrame(signal_log_rows)
        else:
            sig_df = pd.DataFrame(alloc_log_rows) if alloc_log_rows else pd.DataFrame()

        meta: dict[str, Any] = {"bars_per_year": bars_per_year, "symbols": symbols}
        if timeframe is not None:
            meta["timeframe"] = timeframe

        return BacktestResult(
            trades=final_trades,
            equity_curve=eq_series,
            positions=pos_series,
            signal_log=sig_df,
            config=self.config,
            run_time_s=elapsed,
            meta=meta,
            positions_log=(
                pd.DataFrame(pos_log_rows) if pos_log_rows else None
            ),
            allocation_log=(
                pd.DataFrame(alloc_log_rows) if alloc_log_rows else None
            ),
        )