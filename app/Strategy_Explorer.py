"""
Strategy_Explorer — EDA dashboard for quantitative strategy research.

Launch from project root:
    streamlit run app/Strategy_Explorer.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
_SRC = _ROOT / "src"
for _p in [str(_SRC), str(_ROOT), str(_APP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from components.lse_data import TIMEFRAMES, build_universe, get_api_key, load_bars_cached
from components.charts import (
    atr_chart, bollinger_traces, candlestick_chart,
    macd_chart, rsi_chart, trade_markers, volume_bars,
)
from components.forms import signal_form
from components.style import inject

st.set_page_config(
    page_title="Strategy Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()
st.title("Strategy Explorer")


def _to_naive(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return dti.tz_convert("UTC").tz_localize(None) if dti.tz is not None else dti


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    with st.expander("LSE API Key", expanded=False):
        env_key = get_api_key()
        api_key = st.text_input(
            "API Key", value=env_key, type="password", key="viz_api_key",
            help="Leave blank to use LSE_DATA from .env",
        )

    st.divider()
    st.header("Data")
    symbol = st.text_input("Symbol", value="AAPL", key="viz_sym").upper()
    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=6, key="viz_tf")  # default 1d
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("From", value=date.today() - timedelta(days=730), key="viz_start")
    end_date   = col_e.date_input("To",   value=date.today(), key="viz_end")
    load_btn = st.button("Load Data", type="primary", use_container_width=True, key="viz_load")

    df_loaded: pd.DataFrame | None = st.session_state.get("viz_ohlcv")

    if df_loaded is not None:
        st.divider()
        st.header("Indicators")

        overlay_ema = st.checkbox("EMA", value=True, key="viz_ema")
        if overlay_ema:
            ema_fast = st.number_input("EMA fast", value=12, step=1, min_value=2, key="viz_ef")
            ema_slow = st.number_input("EMA slow", value=26, step=1, min_value=2, key="viz_es")
        else:
            ema_fast, ema_slow = 12, 26

        overlay_sma = st.checkbox("SMA", value=False, key="viz_sma")
        sma_period = st.number_input("SMA period", value=50, step=1, min_value=2, key="viz_sp") if overlay_sma else 50

        overlay_bb = st.checkbox("Bollinger Bands", value=False, key="viz_bb")
        if overlay_bb:
            bb_window = st.number_input("BB window", value=20, step=1, min_value=5, key="viz_bbw")
            bb_std    = st.number_input("BB std devs", value=2.0, step=0.5, min_value=0.5, key="viz_bbs")
        else:
            bb_window, bb_std = 20, 2.0

        show_rsi = st.checkbox("RSI", value=True, key="viz_rsi")
        rsi_period = st.number_input("RSI period", value=14, step=1, min_value=2, key="viz_rp") if show_rsi else 14

        show_atr = st.checkbox("ATR", value=False, key="viz_atr")
        atr_period = st.number_input("ATR period", value=14, step=1, min_value=2, key="viz_ap") if show_atr else 14

        show_macd = st.checkbox("MACD", value=False, key="viz_macd")
        if show_macd:
            macd_fast = st.number_input("MACD fast",   value=12, step=1, min_value=2, key="viz_mf")
            macd_slow = st.number_input("MACD slow",   value=26, step=1, min_value=2, key="viz_ms")
            macd_sig  = st.number_input("MACD signal", value=9,  step=1, min_value=2, key="viz_mg")
        else:
            macd_fast, macd_slow, macd_sig = 12, 26, 9

        show_vol_regime = st.checkbox("Volatility Regime", value=False, key="viz_vreg")
        vol_window = st.number_input("Vol window", value=20, step=1, min_value=5, key="viz_vw") if show_vol_regime else 20

        st.divider()
        with st.expander("Strategy Signal Overlay", expanded=False):
            enable_overlay = st.checkbox("Show signals on chart", value=False, key="viz_overlay")
            if enable_overlay:
                sig_cls, sig_params = signal_form(st.sidebar, key_prefix="viz_sig")
                run_overlay = st.button("Compute Signals", key="viz_run_overlay")
            else:
                sig_cls = sig_params = None
                run_overlay = False

# ── Data loading ──────────────────────────────────────────────────────────────

if load_btn:
    with st.spinner(f"Fetching {symbol} {timeframe} bars from LSE…"):
        df = load_bars_cached(
            symbol, timeframe,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            api_key,
            cache_key_prefix="viz",
        )
        if df is not None:
            st.session_state["viz_ohlcv"] = df
            st.session_state["viz_symbol"] = symbol
            st.session_state["viz_timeframe"] = timeframe
            st.session_state.pop("viz_trades_df", None)
            st.session_state.pop("viz_stats", None)
            st.rerun()

# ── Guard: nothing loaded yet ─────────────────────────────────────────────────

df: pd.DataFrame | None = st.session_state.get("viz_ohlcv")
viz_symbol: str   = st.session_state.get("viz_symbol", symbol)
viz_timeframe: str = st.session_state.get("viz_timeframe", timeframe)

if df is None:
    st.info("Enter a symbol and date range in the sidebar, then click **Load Data**.")
    st.stop()

# ── Strategy signal overlay ───────────────────────────────────────────────────

if "enable_overlay" not in dir() or not enable_overlay:
    enable_overlay = False
    run_overlay = False
    sig_cls = None

if enable_overlay and run_overlay and sig_cls is not None:
    with st.spinner("Running strategy on loaded data…"):
        try:
            from strategy.built_in import SingleAssetStrategy
            from backtester.engine import Backtester
            from core.models import BacktestConfig

            strategy = sig_cls(symbol=viz_symbol, **sig_params) \
                if issubclass(sig_cls, SingleAssetStrategy) \
                else sig_cls(**sig_params)

            uni = build_universe(viz_symbol, df)
            result = Backtester(strategy=strategy, config=BacktestConfig()).run(universe=uni)
            st.session_state["viz_trades_df"] = result.trades_df()
            st.success(f"Signal computed — {len(st.session_state['viz_trades_df'])} trades.")
        except Exception as e:
            st.error(f"Signal overlay failed: {e}")

trades_df_overlay: pd.DataFrame | None = st.session_state.get("viz_trades_df")

# ── View range filter ─────────────────────────────────────────────────────────

idx_naive = _to_naive(df.index)
min_date = idx_naive.min().date()
max_date = idx_naive.max().date()

vc1, vc2, _ = st.columns([2, 2, 6])
view_start = vc1.date_input("View from", value=min_date, min_value=min_date, max_value=max_date, key="viz_vs")
view_end   = vc2.date_input("View to",   value=max_date, min_value=min_date, max_value=max_date, key="viz_ve")

mask = (idx_naive >= pd.Timestamp(view_start)) & (idx_naive < pd.Timestamp(view_end) + pd.Timedelta(days=1))
df_view = df[mask]

if df_view.empty:
    st.warning("No data in the selected view range.")
    st.stop()

# ── Metrics strip ─────────────────────────────────────────────────────────────

returns = df_view["close"].pct_change().dropna()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Close", f"${df_view['close'].iloc[-1]:,.4f}")
m2.metric("Bars", f"{len(df_view):,}")
m3.metric("Ann. Vol", f"{returns.std() * (252 ** 0.5) * 100:.1f}%" if len(returns) > 1 else "—")
m4.metric("Total Return", f"{(df_view['close'].iloc[-1] / df_view['close'].iloc[0] - 1) * 100:.2f}%")
m5.metric("Range", f"{min_date} → {max_date}")

# ── Indicators ────────────────────────────────────────────────────────────────

from strategy.indicators import ema, sma, bollinger, rsi as _rsi, atr as _atr

overlays: dict = {}
if overlay_ema:
    overlays[f"EMA {ema_fast}"] = ema(df_view["close"], ema_fast)
    overlays[f"EMA {ema_slow}"] = ema(df_view["close"], ema_slow)
if overlay_sma:
    overlays[f"SMA {sma_period}"] = sma(df_view["close"], sma_period)

# ── Price chart ───────────────────────────────────────────────────────────────

fig = candlestick_chart(df_view, overlays=overlays, title=f"{viz_symbol} — {viz_timeframe}")

if overlay_bb:
    mid, upper, lower = bollinger(df_view["close"], window=bb_window, num_std=bb_std)
    for trace in bollinger_traces(mid, upper, lower):
        fig.add_trace(trace)

if trades_df_overlay is not None and not trades_df_overlay.empty:
    if "timestamp" in trades_df_overlay.columns:
        ts = pd.to_datetime(trades_df_overlay["timestamp"])
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None) if ts.dt.tz is not None else ts
        view_mask = (ts_naive >= pd.Timestamp(view_start)) & (ts_naive < pd.Timestamp(view_end) + pd.Timedelta(days=1))
        fig = trade_markers(fig, trades_df_overlay[view_mask])
    else:
        fig = trade_markers(fig, trades_df_overlay)

st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(volume_bars(df_view), use_container_width=True)

# ── Sub-charts ────────────────────────────────────────────────────────────────

if show_rsi:
    rsi_series = _rsi(df_view["close"], period=rsi_period)
    st.plotly_chart(rsi_chart(rsi_series, period=rsi_period), use_container_width=True)

if show_atr:
    atr_series = _atr(df_view["high"], df_view["low"], df_view["close"], period=atr_period)
    st.plotly_chart(atr_chart(atr_series, period=atr_period), use_container_width=True)

if show_macd:
    st.plotly_chart(
        macd_chart(df_view["close"], fast=macd_fast, slow=macd_slow, signal=macd_sig),
        use_container_width=True,
    )

# ── Volatility regime chart ───────────────────────────────────────────────────

if show_vol_regime:
    import plotly.graph_objects as go

    rv = df_view["close"].pct_change().rolling(vol_window).std() * (252 ** 0.5) * 100
    q_lo = rv.expanding(min_periods=vol_window).quantile(0.33)
    q_hi = rv.expanding(min_periods=vol_window).quantile(0.66)

    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(x=rv.index, y=rv, name=f"Ann. Vol ({vol_window})", line=dict(color="#2196F3", width=1.5)))
    fig_rv.add_trace(go.Scatter(x=q_lo.index, y=q_lo, name="33rd pctl", line=dict(color="#26a69a", dash="dot", width=1)))
    fig_rv.add_trace(go.Scatter(x=q_hi.index, y=q_hi, name="66th pctl", line=dict(color="#ef5350", dash="dot", width=1)))
    fig_rv.update_layout(
        template="plotly_dark", height=260,
        title=f"Volatility Regime — {vol_window}-bar rolling ann. vol",
        yaxis_title="Ann. vol %",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_rv, use_container_width=True)

# ── Returns analysis ──────────────────────────────────────────────────────────

with st.expander("Returns Analysis", expanded=False):
    import plotly.graph_objects as go
    import numpy as np

    col_dist, col_stats = st.columns([3, 2])

    with col_dist:
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=returns * 100, nbinsx=80, name="Returns",
            marker_color="#2196F3", opacity=0.75,
        ))
        fig_ret.add_vline(x=0, line_color="white", line_dash="dash", line_width=1)
        fig_ret.update_layout(
            template="plotly_dark", height=300,
            title="Return Distribution (%)",
            xaxis_title="Return %", yaxis_title="Count",
            showlegend=False, margin=dict(l=40, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with col_stats:
        from scipy import stats as sp_stats
        st.markdown("**Descriptive Statistics**")
        r = returns.dropna()
        ann = 252 ** 0.5
        sharpe = (r.mean() / r.std() * ann) if r.std() > 0 else 0
        skew = float(sp_stats.skew(r))
        kurt = float(sp_stats.kurtosis(r))
        _, pval_norm = sp_stats.normaltest(r)
        st.table(pd.DataFrame({
            "Metric": [
                "Mean daily ret %", "Std daily %", "Ann. vol %",
                "Ann. return %", "Sharpe (ann.)",
                "Skewness", "Excess kurtosis", "Normality p-val",
            ],
            "Value": [
                f"{r.mean()*100:.4f}", f"{r.std()*100:.4f}", f"{r.std()*ann*100:.2f}",
                f"{((1+r.mean())**252 - 1)*100:.2f}", f"{sharpe:.3f}",
                f"{skew:.3f}", f"{kurt:.3f}", f"{pval_norm:.4f}",
            ],
        }))

# ── Raw data ──────────────────────────────────────────────────────────────────

with st.expander("Raw OHLCV Data", expanded=False):
    st.dataframe(df_view.style.format({c: "{:,.4f}" for c in ["open","high","low","close"]}), use_container_width=True)
    csv = df_view.reset_index().to_csv(index=False).encode()
    st.download_button("Download CSV", csv, file_name=f"{viz_symbol}_{viz_timeframe}.csv", mime="text/csv")

st.caption(f"{len(df_view):,} bars  ·  {df_view.index[0]}  →  {df_view.index[-1]}")
