"""Backtest Runner — configure, run, and inspect vectorised backtests."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "src"), str(_ROOT), str(_ROOT / "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import candlestick_chart, equity_chart, trade_markers
from components.forms import backtest_config_form, signal_form, sizer_form, stop_form
from components.style import inject
from utils.plotting import plot_features

st.set_page_config(page_title="Backtester", page_icon="🔬", layout="wide")
inject()
st.title("Backtester")

DATA_DIR: Path = st.session_state.get("data_dir", _ROOT / "data")

# ── Sidebar: configuration ────────────────────────────────────────────────────

with st.sidebar:
    st.header("Signal")
    signal_cls, sig_params = signal_form(st.sidebar, key_prefix="bt_sig")

    st.divider()
    st.header("Data")
    exchange_folder = st.selectbox(
        "Exchange / market",
        ["HYPERLIQUID_PERPETUALS", "BINANCE_PERPETUALS", "BINANCE_SPOT"],
        key="bt_exch",
    )
    bt_symbol = st.text_input("Symbol", value="ETH", key="bt_sym").upper()
    from core.parser import TIMEFRAMES
    bt_timeframe = st.selectbox("Bar timeframe", TIMEFRAMES, index=0, key="bt_tf")

    st.divider()
    st.header("Config")
    config = backtest_config_form(st.sidebar, key_prefix="bt_cfg")

    st.divider()
    st.header("Sizer")
    sizer = sizer_form(st.sidebar, key_prefix="bt_sizer")

    st.divider()
    st.header("Stop loss")
    stop = stop_form(st.sidebar, key_prefix="bt_stop")

    st.divider()
    run_bt = st.button("Run Backtest", type="primary", use_container_width=True)

# ── Run backtest ──────────────────────────────────────────────────────────────

if run_bt:
    if signal_cls is None:
        st.error("No signal selected.")
    else:
        data_path = DATA_DIR / "trades" / exchange_folder / bt_symbol
        if not data_path.exists():
            st.error(f"No data at `{data_path}`. Collect data first.")
        else:
            with st.spinner("Running backtest…"):
                try:
                    from core.parser import trades_to_ohlcv
                    from backtester.engine import Backtester
                    from backtester.costs import CompositeCostModel, aggressive_cost_stack

                    signal = signal_cls(**sig_params)
                    df = trades_to_ohlcv(data_path, timeframe=bt_timeframe)
                    cost = CompositeCostModel(models=aggressive_cost_stack())
                    bt = Backtester(signal=signal, config=config, cost_model=cost,
                                   sizer=sizer, stop_loss=stop)
                    result = bt.run(data=df, timeframe=bt_timeframe)
                    st.session_state["bt_result"] = result
                    st.session_state["bt_ohlcv"] = df
                    st.session_state["bt_timeframe"] = bt_timeframe
                    st.success("Backtest complete.")
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    st.exception(e)

# ── Results ───────────────────────────────────────────────────────────────────

result = st.session_state.get("bt_result")
ohlcv_df: pd.DataFrame | None = st.session_state.get("bt_ohlcv")

if result is None:
    st.info("Configure a signal in the sidebar and click **Run Backtest**.")
    st.stop()

summary = result.summary()

# Metric rows
st.subheader("Summary")
cols = st.columns(4)
cols[0].metric("Total Return", f"{summary.get('total_return_pct', 0):.2f}%")
cols[1].metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.3f}")
cols[2].metric("Max Drawdown", f"{summary.get('max_drawdown_pct', 0):.2f}%")
cols[3].metric("Win Rate", f"{summary.get('win_rate_pct', 0):.1f}%")

cols2 = st.columns(5)
cols2[0].metric("Calmar", f"{summary.get('calmar_ratio', 0):.3f}")
cols2[1].metric("Sortino", f"{summary.get('sortino_ratio', 0):.3f}")
cols2[2].metric("Profit Factor", f"{summary.get('profit_factor', 0):.3f}")
cols2[3].metric("Trades", f"{summary.get('num_trades', 0)}")
cols2[4].metric("Total Fees", f"${summary.get('total_fees', 0):,.2f}")

st.divider()

# Equity curve + drawdown
st.subheader("Equity Curve")
eq = result.equity_curve
dd = (eq - eq.cummax()) / eq.cummax()
st.plotly_chart(equity_chart(eq, dd), use_container_width=True)

# Price chart with trade markers
if ohlcv_df is not None and not ohlcv_df.empty:
    st.subheader("Trades on Price Chart")
    price_fig = candlestick_chart(ohlcv_df, title=f"{bt_symbol} — trades")
    trades_df = result.trades_df()
    if not trades_df.empty:
        price_fig = trade_markers(price_fig, trades_df)
    st.plotly_chart(price_fig, use_container_width=True)

# Signal log
if result.signal_log is not None and not result.signal_log.empty:
    sig_cols = [c for c in result.signal_log.columns if result.signal_log[c].dtype != object]
    if sig_cols:
        st.subheader("Signal Log")
        fig_sig = plot_features(result.signal_log, sig_cols[:6], title="Signal Log", height=250)
        st.plotly_chart(fig_sig, use_container_width=True)

# Trade table
st.subheader("Trade Log")
trades_df = result.trades_df()
if trades_df.empty:
    st.info("No completed trades.")
else:
    # Colour PnL column
    def _color_pnl(val):
        if isinstance(val, (int, float)):
            color = "#26a69a" if val > 0 else "#ef5350" if val < 0 else "inherit"
            return f"color: {color}"
        return ""

    display_cols = [c for c in [
        "timestamp", "side", "size", "entry_price", "exit_price",
        "pnl", "pnl_pct", "fees", "reason_entry", "reason_exit",
    ] if c in trades_df.columns]

    styled = trades_df[display_cols].style.applymap(_color_pnl, subset=["pnl"] if "pnl" in display_cols else [])
    st.dataframe(styled, use_container_width=True)

# ── Stress Test ───────────────────────────────────────────────────────────────

with st.expander("Parameter Sweep"):
    if signal_cls is None:
        st.info("Select a signal first.")
    else:
        import inspect
        sig_sig = inspect.signature(signal_cls.__init__)
        sweep_params = [
            k for k, v in sig_sig.parameters.items()
            if k != "self" and v.default is not inspect.Parameter.empty
            and isinstance(v.default, (int, float))
        ]

        if not sweep_params:
            st.info("This signal has no numeric parameters to sweep.")
        else:
            sc1, sc2 = st.columns(2)
            p1 = sc1.selectbox("Parameter 1", sweep_params, key="stress_p1")
            p1_vals = sc1.text_input("Values (comma-sep)", value="", key="stress_p1_vals",
                                      placeholder="e.g. 5,10,15,20")
            p2_options = ["(none)"] + [p for p in sweep_params if p != p1]
            p2 = sc2.selectbox("Parameter 2 (optional)", p2_options, key="stress_p2")
            p2_vals = sc2.text_input("Values (comma-sep)", value="", key="stress_p2_vals",
                                      placeholder="e.g. 20,30,40") if p2 != "(none)" else ""
            metric_pick = st.selectbox("Metric", [
                "sharpe_ratio", "total_return_pct", "max_drawdown_pct",
                "calmar_ratio", "win_rate_pct", "profit_factor",
            ], key="stress_metric")

            run_stress = st.button("Run sweep", key="run_stress")
            if run_stress:
                try:
                    vals1 = [float(v.strip()) for v in p1_vals.split(",") if v.strip()]
                    vals2 = [float(v.strip()) for v in p2_vals.split(",") if v.strip()] if p2_vals else []

                    if not vals1:
                        st.warning("Enter at least one value for Parameter 1.")
                    else:
                        data_path = DATA_DIR / "trades" / exchange_folder / bt_symbol
                        from core.parser import trades_to_ohlcv
                        from backtester.costs import CompositeCostModel, aggressive_cost_stack
                        from backtester.stress import SignalStressTest

                        df_stress = trades_to_ohlcv(data_path, timeframe=bt_timeframe)
                        cost = CompositeCostModel(models=aggressive_cost_stack())

                        param_grid = {p1: vals1}
                        if vals2:
                            param_grid[p2] = vals2

                        with st.spinner("Running parameter sweep…"):
                            stress_test = SignalStressTest(
                                signal_cls=signal_cls,
                                param_grid=param_grid,
                                cost_model=cost,
                            )
                            sweep = stress_test.run(data=df_stress)
                        st.session_state["stress_result"] = (sweep, p1, p2, metric_pick, vals1, vals2)
                        st.success("Sweep complete.")
                except Exception as e:
                    st.error(f"Sweep failed: {e}")
                    st.exception(e)

    sweep_data = st.session_state.get("stress_result")
    if sweep_data:
        sweep, p1, p2, metric, vals1, vals2 = sweep_data
        summary_df: pd.DataFrame = sweep.summary

        if p2 and p2 != "(none)" and vals2 and metric in summary_df.columns:
            # 2D heatmap
            pivot = summary_df.pivot(index=p2, columns=p1, values=metric)
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdYlGn",
                text=[[f"{v:.3f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(
                template="plotly_dark", height=400,
                title=f"{metric} heatmap — {p1} vs {p2}",
                xaxis_title=p1, yaxis_title=p2,
                margin=dict(l=40, r=40, t=50, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        elif metric in summary_df.columns and p1 in summary_df.columns:
            # 1D bar chart
            fig_bar = go.Figure(go.Bar(
                x=summary_df[p1].astype(str),
                y=summary_df[metric],
                marker_color="#2196F3",
            ))
            fig_bar.update_layout(
                template="plotly_dark", height=300,
                title=f"{metric} vs {p1}",
                xaxis_title=p1, yaxis_title=metric,
                margin=dict(l=40, r=40, t=50, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(summary_df, use_container_width=True)
