"""Backtester — run backtests and stress tests on LSE data."""
import inspect
import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "src"), str(_ROOT), str(_ROOT / "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.lse_data import BACKTEST_TIMEFRAMES, build_universe, get_api_key, load_bars_cached
from components.charts import candlestick_chart, equity_chart, signal_log_chart, trade_markers
from components.forms import backtest_config_form, signal_form, sizer_form, stop_form
from components.style import inject

st.set_page_config(page_title="Backtester", page_icon="🔬", layout="wide")
inject()
st.title("Backtester")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    with st.expander("LSE API Key", expanded=False):
        env_key = get_api_key()
        api_key = st.text_input(
            "API Key", value=env_key, type="password", key="bt_api_key",
            help="Leave blank to use LSE_DATA from .env",
        )

    st.divider()
    st.header("Signal")
    signal_cls, sig_params = signal_form(st.sidebar, key_prefix="bt_sig")

    st.divider()
    st.header("Data")
    bt_symbol    = st.text_input("Symbol", value="AAPL", key="bt_sym").upper()
    bt_timeframe = st.selectbox("Timeframe", BACKTEST_TIMEFRAMES, index=6, key="bt_tf")
    col_s, col_e = st.columns(2)
    bt_start = col_s.date_input("From", value=date.today() - timedelta(days=365*5), key="bt_start")
    bt_end   = col_e.date_input("To",   value=date.today(), key="bt_end")

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
        with st.spinner(f"Fetching {bt_symbol} data and running backtest…"):
            try:
                df = load_bars_cached(
                    bt_symbol, bt_timeframe,
                    bt_start.strftime("%Y-%m-%d"),
                    bt_end.strftime("%Y-%m-%d"),
                    api_key,
                    cache_key_prefix="bt",
                )
                if df is None:
                    st.stop()

                from strategy.built_in import SingleAssetStrategy
                from backtester.engine import Backtester
                from backtester.costs import CompositeCostModel, aggressive_cost_stack

                strategy = signal_cls(symbol=bt_symbol, **sig_params) \
                    if issubclass(signal_cls, SingleAssetStrategy) \
                    else signal_cls(**sig_params)

                uni  = build_universe(bt_symbol, df)
                cost = CompositeCostModel(models=aggressive_cost_stack())
                bt   = Backtester(strategy=strategy, config=config, cost_model=cost,
                                  sizer=sizer, stop_loss=stop)
                result = bt.run(universe=uni, timeframe=bt_timeframe)

                st.session_state["bt_result"]    = result
                st.session_state["bt_ohlcv"]     = df
                st.session_state["bt_timeframe"] = bt_timeframe
                st.session_state["bt_symbol"]    = bt_symbol
                st.success("Backtest complete.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.exception(e)

# ── Results guard ─────────────────────────────────────────────────────────────

result       = st.session_state.get("bt_result")
ohlcv_df     = st.session_state.get("bt_ohlcv")
stored_symbol = st.session_state.get("bt_symbol", bt_symbol)
stored_tf     = st.session_state.get("bt_timeframe", bt_timeframe)

if result is None:
    st.info("Configure a signal in the sidebar and click **Run Backtest**.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_results, tab_hypothesis, tab_sweep, tab_regime, tab_mc = st.tabs([
    "Results", "Hypothesis Tests", "Param Sweep", "Regime Test", "Monte Carlo",
])

# ═══════════════════════════════════════════════════════════════ Results tab

with tab_results:
    summary = result.summary()

    st.subheader("Summary")
    c1 = st.columns(4)
    c1[0].metric("Total Return",   f"{summary.get('total_return_pct', 0):.2f}%")
    c1[1].metric("Sharpe Ratio",   f"{summary.get('sharpe_ratio', 0):.3f}")
    c1[2].metric("Max Drawdown",   f"{summary.get('max_drawdown_pct', 0):.2f}%")
    c1[3].metric("Win Rate",       f"{summary.get('win_rate_pct', 0):.1f}%")

    c2 = st.columns(5)
    c2[0].metric("Calmar",         f"{summary.get('calmar_ratio', 0):.3f}")
    c2[1].metric("Sortino",        f"{summary.get('sortino_ratio', 0):.3f}")
    c2[2].metric("Profit Factor",  f"{summary.get('profit_factor', 0):.3f}")
    c2[3].metric("Trades",         f"{summary.get('num_trades', 0)}")
    c2[4].metric("Total Fees",     f"${summary.get('total_fees', 0):,.2f}")

    st.divider()

    st.subheader("Equity Curve")
    eq = result.equity_curve
    dd = (eq - eq.cummax()) / eq.cummax()
    st.plotly_chart(equity_chart(eq, dd), use_container_width=True)

    if ohlcv_df is not None and not ohlcv_df.empty:
        st.subheader("Trades on Price Chart")
        price_fig = candlestick_chart(ohlcv_df, title=f"{stored_symbol} — {stored_tf}")
        trades_df = result.trades_df()
        if not trades_df.empty:
            price_fig = trade_markers(price_fig, trades_df)
        st.plotly_chart(price_fig, use_container_width=True)

    sig_log = getattr(result, "signal_log", None)
    if sig_log is not None and not sig_log.empty:
        st.subheader("Signal Log")
        st.plotly_chart(signal_log_chart(sig_log, height=320), use_container_width=True)
        with st.expander("Signal Log Table", expanded=False):
            display_log = sig_log.copy()
            if "timestamp" in display_log.columns:
                ts_conv = pd.to_datetime(display_log["timestamp"], unit="ms", errors="coerce")
                if not ts_conv.isna().all():
                    display_log["timestamp"] = ts_conv

            def _color_side(val):
                return {"LONG": "color: #26a69a", "SHORT": "color: #ef5350"}.get(val, "color: #9ba3b8")

            st.dataframe(
                display_log.style.map(_color_side, subset=["side"] if "side" in display_log.columns else []),
                use_container_width=True,
            )

    st.subheader("Trade Log")
    trades_df = result.trades_df()
    if trades_df.empty:
        st.info("No completed trades.")
    else:
        def _color_pnl(val):
            if isinstance(val, (int, float)):
                return f"color: {'#26a69a' if val > 0 else '#ef5350' if val < 0 else 'inherit'}"
            return ""

        display_cols = [c for c in [
            "timestamp", "side", "size", "entry_price", "exit_price",
            "pnl", "pnl_pct", "fees", "reason_entry", "reason_exit",
        ] if c in trades_df.columns]
        styled = trades_df[display_cols].style.map(_color_pnl, subset=["pnl"] if "pnl" in display_cols else [])
        st.dataframe(styled, use_container_width=True)

        csv = trades_df[display_cols].to_csv(index=False).encode()
        st.download_button("Download trades CSV", csv, file_name=f"{stored_symbol}_trades.csv", mime="text/csv")

# ════════════════════════════════════════════════════════ Hypothesis Tests tab

with tab_hypothesis:
    st.info(
        "Run statistical tests on the current backtest result. "
        "For a proper TTV workflow, use the Param Sweep and Regime tabs on the **test** split only, "
        "then run hypothesis tests on the held-out **validate** split."
    )

    ht_col1, ht_col2 = st.columns(2)
    n_permutations = ht_col1.number_input("Permutation test samples", value=2_000, step=500, min_value=200, key="ht_perms")
    n_bootstrap    = ht_col2.number_input("Bootstrap CI samples",     value=2_000, step=500, min_value=200, key="ht_boot")
    n_trials_input = ht_col1.number_input(
        "Number of param combos tried (for DSR)", value=1, step=1, min_value=1, key="ht_ntrials",
        help="Total parameter combinations you tested on the test split — used by Deflated Sharpe Ratio.",
    )
    ci_level = ht_col2.slider("Bootstrap CI level", min_value=0.80, max_value=0.99, value=0.95, step=0.01, key="ht_ci")

    run_ht = st.button("Run Hypothesis Tests", type="primary", key="run_ht")

    if run_ht:
        try:
            from hypothesis import HypothesisTests, PermutationTest, BootstrapCI, DeflatedSharpeRatio, report

            with st.spinner("Running hypothesis tests…"):
                tests    = HypothesisTests.run_all(result)
                pt       = PermutationTest(metric="sharpe_ratio", n_permutations=int(n_permutations)).run(result)
                ci       = BootstrapCI(n_bootstrap=int(n_bootstrap), ci=ci_level).run(result)
                dsr      = DeflatedSharpeRatio().compute(result, n_trials=int(n_trials_input))

            st.session_state["ht_results"] = (tests, pt, ci, dsr)
            st.success("Done.")
        except Exception as e:
            st.error(f"Hypothesis tests failed: {e}")
            st.exception(e)

    ht_data = st.session_state.get("ht_results")
    if ht_data:
        tests, pt, ci, dsr = ht_data

        st.subheader("Statistical Test Battery")
        from hypothesis import report
        st.code(report(tests))

        st.subheader("Permutation Test  (Sharpe ratio)")
        verdict = "Significant" if pt.reject_null else "Not significant"
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Observed Sharpe", f"{pt.statistic:.4f}")
        col_b.metric("Null median", f"{pt.meta.get('null_mean', 0):.4f}")
        col_c.metric(f"p-value → {verdict}", f"{pt.p_value:.4f}")

        st.subheader(f"Bootstrap {int(ci_level*100)}% Confidence Intervals")
        ci_rows = []
        for metric, vals in ci.items():
            ci_rows.append({"Metric": metric, "Observed": vals["observed"],
                            "Lower": vals["lower"], "Upper": vals["upper"]})
        st.dataframe(pd.DataFrame(ci_rows), use_container_width=True)

        st.subheader(f"Deflated Sharpe Ratio  (n_trials={int(n_trials_input)})")
        dsr_verdict = "Genuine edge" if dsr.reject_null else "Likely overfit"
        da, db, dc, dd_ = st.columns(4)
        da.metric("Observed Sharpe",  f"{dsr.observed_sharpe:.4f}")
        db.metric("Deflated Sharpe",  f"{dsr.deflated_sharpe:.4f}")
        dc.metric("p-value",          f"{dsr.p_value:.4f}")
        dd_.metric("Verdict",         dsr_verdict)

# ═══════════════════════════════════════════════════════════ Param Sweep tab

with tab_sweep:
    if signal_cls is None:
        st.info("Select a signal in the sidebar first.")
    else:
        sweep_params = [
            k for k, v in inspect.signature(signal_cls.__init__).parameters.items()
            if k != "self"
            and v.default is not inspect.Parameter.empty
            and isinstance(v.default, (int, float))
        ]

        if not sweep_params:
            st.info("This strategy has no numeric parameters to sweep.")
        else:
            sc1, sc2 = st.columns(2)
            p1       = sc1.selectbox("Parameter 1", sweep_params, key="sweep_p1")
            p1_vals  = sc1.text_input("Values (comma-sep)", placeholder="e.g. 5,10,15,20", key="sweep_p1v")

            p2_opts  = ["(none)"] + [p for p in sweep_params if p != p1]
            p2       = sc2.selectbox("Parameter 2 (optional)", p2_opts, key="sweep_p2")
            p2_vals  = sc2.text_input("Values (comma-sep)", placeholder="e.g. 20,30,40", key="sweep_p2v") if p2 != "(none)" else ""

            metric_pick = st.selectbox("Optimise metric", [
                "sharpe_ratio", "total_return_pct", "max_drawdown_pct",
                "calmar_ratio", "win_rate_pct", "profit_factor",
            ], key="sweep_metric")

            run_sweep = st.button("Run Sweep", type="primary", key="run_sweep")

            if run_sweep:
                try:
                    vals1 = [float(v.strip()) for v in p1_vals.split(",") if v.strip()]
                    vals2 = [float(v.strip()) for v in p2_vals.split(",") if v.strip()] if p2_vals else []
                    if not vals1:
                        st.warning("Enter at least one value for Parameter 1.")
                    else:
                        df_sweep = load_bars_cached(
                            bt_symbol, bt_timeframe,
                            bt_start.strftime("%Y-%m-%d"),
                            bt_end.strftime("%Y-%m-%d"),
                            api_key,
                            cache_key_prefix="bt",
                        )
                        if df_sweep is not None:
                            from backtester.costs import CompositeCostModel, aggressive_cost_stack
                            from backtester.stress import ParamSweep

                            param_grid = {p1: vals1}
                            if vals2:
                                param_grid[p2] = vals2

                            uni  = build_universe(bt_symbol, df_sweep)
                            cost = CompositeCostModel(models=aggressive_cost_stack())
                            n_combos = len(vals1) * max(len(vals2), 1)

                            with st.spinner(f"Running {n_combos} backtest(s)…"):
                                sweep_result = ParamSweep(
                                    strategy_cls=signal_cls,
                                    param_grid=param_grid,
                                    config=config,
                                    cost_model=cost,
                                    sizer=sizer,
                                    stop_loss=stop,
                                ).run(universe=uni, timeframe=bt_timeframe)

                            st.session_state["sweep_result"] = (sweep_result, p1, p2, metric_pick, vals1, vals2)
                            st.info(f"Tracked {n_combos} trials — enter this in the Hypothesis Tests tab for DSR correction.")
                            st.success("Sweep complete.")
                except Exception as e:
                    st.error(f"Sweep failed: {e}")
                    st.exception(e)

    sweep_data = st.session_state.get("sweep_result")
    if sweep_data:
        sweep, p1, p2, metric, vals1, vals2 = sweep_data
        summary_df: pd.DataFrame = sweep.summary

        if p2 and p2 != "(none)" and vals2 and metric in summary_df.columns and p1 in summary_df.columns and p2 in summary_df.columns:
            pivot = summary_df.pivot(index=p2, columns=p1, values=metric)
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale="RdYlGn",
                text=[[f"{v:.3f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(
                template="plotly_dark", height=420,
                title=f"{metric} — {p1} vs {p2}",
                xaxis_title=p1, yaxis_title=p2,
                margin=dict(l=40, r=40, t=50, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        elif metric in summary_df.columns and p1 in summary_df.columns:
            fig_bar = go.Figure(go.Bar(
                x=summary_df[p1].astype(str), y=summary_df[metric], marker_color="#2196F3",
            ))
            fig_bar.update_layout(
                template="plotly_dark", height=300,
                title=f"{metric} vs {p1}",
                xaxis_title=p1, yaxis_title=metric,
                margin=dict(l=40, r=40, t=50, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if metric in summary_df.columns:
            best  = sweep.best(metric)
            worst = sweep.worst(metric)
            b1, b2 = st.columns(2)
            b1.success(f"Best  {metric}: {best[metric]:.4f} @ {best.drop(metric).to_dict()}")
            b2.error(  f"Worst {metric}: {worst[metric]:.4f} @ {worst.drop(metric).to_dict()}")

        st.dataframe(summary_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════ Regime Test tab

with tab_regime:
    if signal_cls is None:
        st.info("Select a signal in the sidebar first.")
    else:
        regime_choice = st.radio(
            "Regime classifier",
            ["Volatility (default)", "Trend (SMA)", "Volume"],
            key="regime_choice",
        )
        sma_win = st.number_input("SMA window", value=50, step=5, min_value=10, key="regime_sma") \
            if regime_choice == "Trend (SMA)" else 50

        run_regime = st.button("Run Regime Test", type="primary", key="run_regime")

        if run_regime:
            df_reg = load_bars_cached(
                bt_symbol, bt_timeframe,
                bt_start.strftime("%Y-%m-%d"),
                bt_end.strftime("%Y-%m-%d"),
                api_key,
                cache_key_prefix="bt",
            )
            if df_reg is not None:
                try:
                    from strategy.built_in import SingleAssetStrategy
                    from backtester.stress import RegimeStressTest
                    from backtester.costs import CompositeCostModel, aggressive_cost_stack

                    strategy_reg = signal_cls(symbol=bt_symbol, **sig_params) \
                        if issubclass(signal_cls, SingleAssetStrategy) \
                        else signal_cls(**sig_params)

                    if regime_choice == "Volatility (default)":
                        regime_fn = None
                    elif regime_choice == "Trend (SMA)":
                        _w = sma_win
                        regime_fn = lambda df_: RegimeStressTest.trend_regime(df_, sma_window=_w)
                    else:
                        regime_fn = RegimeStressTest.volume_regime

                    cost = CompositeCostModel(models=aggressive_cost_stack())
                    uni  = build_universe(bt_symbol, df_reg)

                    with st.spinner("Running regime stress test…"):
                        regime_result = RegimeStressTest(
                            regime_fn=regime_fn, config=config, cost_model=cost,
                        ).run(strategy=strategy_reg, universe=uni)

                    st.session_state["regime_result"] = regime_result
                    st.success("Regime test complete.")
                except Exception as e:
                    st.error(f"Regime test failed: {e}")
                    st.exception(e)

    regime_result = st.session_state.get("regime_result")
    if regime_result is not None:
        st.dataframe(regime_result.summary, use_container_width=True)

        for regime_name, res in regime_result.results.items():
            with st.expander(f"Regime: {regime_name}", expanded=False):
                eq_r = res.equity_curve
                dd_r = (eq_r - eq_r.cummax()) / eq_r.cummax()
                st.plotly_chart(equity_chart(eq_r, dd_r), use_container_width=True)
                rs = res.summary()
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Return", f"{rs.get('total_return_pct', 0):.2f}%")
                rc2.metric("Sharpe", f"{rs.get('sharpe_ratio', 0):.3f}")
                rc3.metric("Max DD", f"{rs.get('max_drawdown_pct', 0):.2f}%")
                rc4.metric("Trades", f"{rs.get('num_trades', 0)}")

# ═══════════════════════════════════════════════════════════ Monte Carlo tab

with tab_mc:
    if result is None:
        st.info("Run a backtest first (Results tab).")
    else:
        trades_df_mc = result.trades_df()
        if trades_df_mc.empty or "pnl" not in trades_df_mc.columns:
            st.warning("No trades with PnL data available for Monte Carlo simulation.")
        else:
            mc1, mc2, mc3 = st.columns(3)
            n_sims     = mc1.number_input("Simulations", value=1000, step=100, min_value=100, max_value=10000, key="mc_n")
            mc_method  = mc2.radio("Method", ["bootstrap", "shuffle", "block_bootstrap"], key="mc_method")
            mc_seed    = mc3.number_input("Seed", value=42, step=1, key="mc_seed")
            run_mc     = st.button("Run Monte Carlo", type="primary", key="run_mc")

            if run_mc:
                try:
                    from backtester.stress import MonteCarloStress
                    with st.spinner(f"Running {n_sims} simulations…"):
                        mc_result = MonteCarloStress(
                            n_simulations=int(n_sims), seed=int(mc_seed), method=mc_method,
                        ).run(result)
                    st.session_state["mc_result"] = mc_result
                    st.success("Monte Carlo complete.")
                except Exception as e:
                    st.error(f"Monte Carlo failed: {e}")
                    st.exception(e)

    mc_result = st.session_state.get("mc_result")
    if mc_result is not None and not mc_result.summary.empty:
        meta = mc_result.meta
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Median Return",    f"{meta.get('median_return', 0):.2f}%")
        mm2.metric("5th Pctl Return",  f"{meta.get('5th_pctl_return', 0):.2f}%")
        mm3.metric("95th Pctl Return", f"{meta.get('95th_pctl_return', 0):.2f}%")
        mm4.metric("Median Max DD",    f"{meta.get('median_max_dd', 0):.2f}%")

        mc_df = mc_result.summary

        def _mc_hist(col: str, title: str) -> go.Figure:
            vals = mc_df[col]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=vals, nbinsx=60, marker_color="#2196F3", opacity=0.75))
            fig.add_vline(x=vals.median(), line_dash="dash", line_color="#ef5350",
                          annotation_text=f"Median {vals.median():.2f}", annotation_position="top right")
            fig.add_vline(x=vals.quantile(0.05), line_dash="dot", line_color="#FF9800",
                          annotation_text=f"5th {vals.quantile(0.05):.2f}", annotation_position="top left")
            fig.add_vline(x=vals.quantile(0.95), line_dash="dot", line_color="#26a69a",
                          annotation_text=f"95th {vals.quantile(0.95):.2f}", annotation_position="top right")
            fig.update_layout(template="plotly_dark", height=300, title=title,
                              showlegend=False, margin=dict(l=40, r=20, t=50, b=20))
            return fig

        st.plotly_chart(_mc_hist("total_return_pct", "Return % Distribution"), use_container_width=True)
        st.plotly_chart(_mc_hist("max_drawdown_pct", "Max Drawdown % Distribution"), use_container_width=True)

        with st.expander("Full statistics"):
            st.dataframe(mc_df.describe(), use_container_width=True)

        st.download_button(
            "Download MC results CSV",
            mc_df.to_csv(index=False).encode(),
            file_name="monte_carlo_results.csv",
            mime="text/csv",
        )
