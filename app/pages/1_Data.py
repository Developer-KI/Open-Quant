"""Data Explorer — OHLCV, order book, funding, sentiment, and macro data."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "src"), str(_ROOT), str(_ROOT / "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import (
    atr_chart, bollinger_traces, candlestick_chart,
    depth_chart, funding_chart, funding_rate_mini, macro_chart, rsi_chart,
    sentiment_scatter, spread_chart, volume_bars,
)
from components.style import inject

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
inject()
st.title("Data Explorer")

DATA_DIR = st.session_state.get("data_dir", _ROOT / "data")

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data source")
    exchange = st.selectbox("Exchange", ["hyperliquid", "binance"], index=0)
    market_type = st.selectbox("Market", ["PERPETUALS", "SPOT"], index=0)
    symbol = st.text_input("Symbol", value="ETH").upper()

    exchange_folder = f"{exchange.upper()}_{market_type}"

    st.divider()
    st.subheader("OHLCV options")
    from core.parser import TIMEFRAMES
    ohlcv_timeframe = st.selectbox("Bar timeframe", TIMEFRAMES, index=0, key="data_tf")
    overlay_ema = st.checkbox("EMA", value=True)
    if overlay_ema:
        ema_fast = st.number_input("EMA fast", value=12, step=1, min_value=2)
        ema_slow = st.number_input("EMA slow", value=26, step=1, min_value=2)
    overlay_sma = st.checkbox("SMA", value=False)
    if overlay_sma:
        sma_period = st.number_input("SMA period", value=50, step=1, min_value=2)
    overlay_bb = st.checkbox("Bollinger Bands", value=False)
    if overlay_bb:
        bb_window = st.number_input("BB window", value=20, step=1, min_value=5)
        bb_std = st.number_input("BB std devs", value=2.0, step=0.5, min_value=0.5)
    show_rsi = st.checkbox("RSI", value=True)
    rsi_period = st.number_input("RSI period", value=14, step=1, min_value=2) if show_rsi else 14
    show_atr = st.checkbox("ATR", value=False)
    atr_period = st.number_input("ATR period", value=14, step=1, min_value=2) if show_atr else 14

    st.divider()
    st.subheader("Overlays")
    overlay_l2 = st.checkbox("L2 bid/ask band", value=False,
                              help="Show best bid and ask from order book snapshots on the price chart, plus a spread (bps) subplot.")
    overlay_funding_on_price = st.checkbox("Funding rate", value=False,
                                           help="Show funding rate (bps) as a subplot below the price chart.")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_price, tab_ob, tab_funding, tab_sentiment, tab_macro = st.tabs([
    "Price & Indicators", "Order Book", "Funding", "Sentiment", "Macro",
])

# ═══════════════════════════════════════════════════════════ Price & Indicators

with tab_price:
    trades_path = DATA_DIR / "trades" / exchange_folder / symbol

    if not trades_path.exists():
        st.info(f"No trade data found at `{trades_path}`. Start a data feed to collect data.")
    else:
        if st.button("Load data", key="load_ohlcv"):
            with st.spinner("Parsing trade data…"):
                try:
                    from core.parser import trades_to_ohlcv
                    _df = trades_to_ohlcv(trades_path, timeframe=ohlcv_timeframe)
                    st.session_state["ohlcv_df"] = _df
                    st.success(f"Loaded {len(_df):,} bars ({ohlcv_timeframe})")
                except Exception as e:
                    st.error(f"OHLCV load failed: {e}")

            if overlay_l2:
                _l2_path = DATA_DIR / "l2" / exchange_folder / symbol
                if _l2_path.exists():
                    with st.spinner("Loading L2 snapshots…"):
                        try:
                            from core.parser import l2_to_orderbook
                            st.session_state["l2_snapshots"] = l2_to_orderbook(_l2_path)
                        except Exception as e:
                            st.warning(f"L2 load failed: {e}")
                else:
                    st.caption(f"No L2 data at `{_l2_path}` — overlay skipped.")

            if overlay_funding_on_price:
                _fund_path = DATA_DIR / "funding" / exchange_folder / symbol
                if _fund_path.exists():
                    with st.spinner("Loading funding rates…"):
                        try:
                            from core.parser import funding_to_snapshots
                            _fund_snaps = funding_to_snapshots(_fund_path)
                            st.session_state["funding_snapshots"] = _fund_snaps
                            if _fund_snaps:
                                _df_f = pd.DataFrame([
                                    {"funding_rate": s.rate, "funding_rate_ann_bps": s.rate_annualized,
                                     "mark_price": s.mark_price, "oracle_price": s.oracle_price}
                                    for s in _fund_snaps
                                ], index=pd.DatetimeIndex([s.timestamp for s in _fund_snaps]))
                                st.session_state["funding_df"] = _df_f
                        except Exception as e:
                            st.warning(f"Funding load failed: {e}")
                else:
                    st.caption(f"No funding data at `{_fund_path}` — overlay skipped.")

    df: pd.DataFrame | None = st.session_state.get("ohlcv_df")

    if df is not None and not df.empty:
        from strategy.indicators import ema, sma, bollinger, rsi, atr

        # Date filter
        c1, c2 = st.columns(2)
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        start_date = c1.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
        end_date = c2.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
        df_view = df.loc[str(start_date):str(end_date)]

        if df_view.empty:
            st.warning("No data in selected range.")
        else:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            def _to_naive(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
                return dti.tz_convert("UTC").tz_localize(None) if dti.tz is not None else dti

            # ── Collect overlay data filtered to the selected date range ──
            ts_l2: list = []
            bid0: list = []
            ask0: list = []
            bps_series: list = []
            if overlay_l2:
                all_snaps = st.session_state.get("l2_snapshots") or []
                if all_snaps:
                    snap_ts = pd.DatetimeIndex([s.timestamp for s in all_snaps])
                    snap_ts_naive = _to_naive(snap_ts)
                    in_range = (snap_ts_naive >= start_ts) & (snap_ts_naive < end_ts)
                    snaps_view = [s for s, ok in zip(all_snaps, in_range) if ok]
                    ts_l2 = [s.timestamp for s in snaps_view]
                    bid0 = [s.bids[0].price if s.bids else None for s in snaps_view]
                    ask0 = [s.asks[0].price if s.asks else None for s in snaps_view]
                    bps_series = [s.spread_bps for s in snaps_view]

            df_fund_view: pd.DataFrame | None = None
            if overlay_funding_on_price:
                df_fund = st.session_state.get("funding_df")
                if df_fund is not None and not df_fund.empty:
                    idx_naive = _to_naive(df_fund.index)
                    fund_mask = (idx_naive >= start_ts) & (idx_naive < end_ts)
                    _fv = df_fund[fund_mask]
                    df_fund_view = _fv if not _fv.empty else None

            # ── Build main price chart ────────────────────────────────────
            overlays: dict = {}
            if overlay_ema:
                overlays[f"EMA {ema_fast}"] = ema(df_view["close"], ema_fast)
                overlays[f"EMA {ema_slow}"] = ema(df_view["close"], ema_slow)
            if overlay_sma:
                overlays[f"SMA {sma_period}"] = sma(df_view["close"], sma_period)

            fig = candlestick_chart(df_view, overlays=overlays, title=f"{symbol} {exchange_folder}")

            if overlay_bb:
                mid, upper, lower = bollinger(df_view["close"], window=bb_window, num_std=bb_std)
                for trace in bollinger_traces(mid, upper, lower):
                    fig.add_trace(trace)

            # L2 bid/ask band — two thin lines that form the spread ribbon
            if ts_l2:
                fig.add_trace(go.Scatter(
                    x=ts_l2, y=bid0, name="Best Bid",
                    line=dict(color="#26a69a", width=0.8), opacity=0.6,
                ))
                fig.add_trace(go.Scatter(
                    x=ts_l2, y=ask0, name="Best Ask",
                    line=dict(color="#ef5350", width=0.8), opacity=0.6,
                    fill="tonexty", fillcolor="rgba(128,128,128,0.06)",
                ))

            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(volume_bars(df_view), use_container_width=True)

            if show_rsi:
                rsi_series = rsi(df_view["close"], period=rsi_period)
                st.plotly_chart(rsi_chart(rsi_series, period=rsi_period), use_container_width=True)

            if ts_l2:
                st.plotly_chart(spread_chart(ts_l2, bps_series), use_container_width=True)

            if df_fund_view is not None:
                st.plotly_chart(funding_rate_mini(df_fund_view), use_container_width=True)

            if show_atr:
                atr_series = atr(df_view["high"], df_view["low"], df_view["close"], period=atr_period)
                st.plotly_chart(atr_chart(atr_series, period=atr_period), use_container_width=True)

            st.caption(f"{len(df_view):,} bars  |  {df_view.index[0]}  →  {df_view.index[-1]}")

# ═══════════════════════════════════════════════════════════════ Order Book

with tab_ob:
    l2_path = DATA_DIR / "l2" / exchange_folder / symbol

    if not l2_path.exists():
        st.info(f"No L2 data found at `{l2_path}`.")
    else:
        if st.button("Load order book snapshots", key="load_l2"):
            with st.spinner("Parsing L2 Parquet files…"):
                try:
                    from core.parser import l2_to_orderbook
                    snapshots = l2_to_orderbook(l2_path)
                    st.session_state["l2_snapshots"] = snapshots
                    st.success(f"Loaded {len(snapshots):,} snapshots")
                except Exception as e:
                    st.error(f"Failed to load L2 data: {e}")

    snapshots = st.session_state.get("l2_snapshots")
    if snapshots:
        idx = st.slider("Snapshot", min_value=0, max_value=len(snapshots) - 1,
                        value=len(snapshots) - 1, key="ob_slider")
        snap = snapshots[idx]
        col1, col2, col3 = st.columns(3)
        col1.metric("Mid price", f"${snap.mid:,.4f}")
        col2.metric("Spread", f"{snap.spread_bps:.2f} bps")
        col3.metric("Timestamp", str(snap.timestamp)[:19])
        st.plotly_chart(depth_chart(snap), use_container_width=True)

        ts = [s.timestamp for s in snapshots]
        spreads = [s.spread_bps for s in snapshots]
        st.plotly_chart(
            spread_chart(ts, spreads, title="Spread over time (bps)", height=200),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════ Funding

with tab_funding:
    funding_path = DATA_DIR / "funding" / exchange_folder / symbol

    if not funding_path.exists():
        st.info(f"No funding data at `{funding_path}`.")
    else:
        if st.button("Load funding data", key="load_funding"):
            with st.spinner("Reading Parquet files…"):
                try:
                    parts = list(funding_path.glob("*.parquet"))
                    if not parts:
                        st.warning("No Parquet files found.")
                    else:
                        from core.parser import funding_to_snapshots
                        fund_snaps = funding_to_snapshots(funding_path)
                        st.session_state["funding_snapshots"] = fund_snaps
                        if fund_snaps:
                            df_f = pd.DataFrame([
                                {"funding_rate": s.rate, "funding_rate_ann_bps": s.rate_annualized,
                                 "mark_price": s.mark_price, "oracle_price": s.oracle_price}
                                for s in fund_snaps
                            ], index=pd.DatetimeIndex([s.timestamp for s in fund_snaps]))
                            st.session_state["funding_df"] = df_f
                            st.success(f"Loaded {len(fund_snaps):,} snapshots")
                except Exception as e:
                    st.error(f"Failed: {e}")

    df_f = st.session_state.get("funding_df")
    if df_f is not None and not df_f.empty:
        st.plotly_chart(funding_chart(df_f), use_container_width=True)
        st.dataframe(df_f.tail(50), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════ Sentiment

with tab_sentiment:
    sentiment_dir = DATA_DIR / "sentiment"
    if not sentiment_dir.exists():
        st.info(f"No sentiment data at `{sentiment_dir}`.")
    else:
        csv_files = list(sentiment_dir.glob("*.csv"))
        if not csv_files:
            st.info("No CSV files found in sentiment directory.")
        else:
            selected_files = st.multiselect(
                "Files to load",
                [f.name for f in csv_files],
                default=[f.name for f in csv_files],
                key="sentiment_files",
            )
            if selected_files and st.button("Load sentiment data", key="load_sentiment"):
                with st.spinner("Loading…"):
                    try:
                        frames = []
                        for fname in selected_files:
                            df_s = pd.read_csv(sentiment_dir / fname)
                            if "source" not in df_s.columns:
                                df_s["source"] = Path(fname).stem
                            frames.append(df_s)
                        st.session_state["sentiment_df"] = pd.concat(frames, ignore_index=True)
                        st.success(f"Loaded {len(st.session_state['sentiment_df']):,} posts")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    df_s = st.session_state.get("sentiment_df")
    if df_s is not None and not df_s.empty:
        sources = df_s["source"].unique().tolist() if "source" in df_s.columns else []
        if sources:
            selected_src = st.multiselect("Filter by source", sources, default=sources, key="src_filter")
            df_s = df_s[df_s["source"].isin(selected_src)]

        st.plotly_chart(sentiment_scatter(df_s), use_container_width=True)

        score_col = next((c for c in ["score", "sentiment_score", "compound"] if c in df_s.columns), None)
        if score_col:
            import plotly.express as px
            fig_hist = px.histogram(df_s, x=score_col, color="source" if "source" in df_s.columns else None,
                                    nbins=40, template="plotly_dark", height=250, title="Score distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

        display_cols = [c for c in ["source", "created_at", "post", score_col] if c and c in df_s.columns]
        st.dataframe(df_s[display_cols].head(200) if display_cols else df_s.head(200),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════════════════ Macro

with tab_macro:
    macro_dir = DATA_DIR / "macro_snapshots"
    if not macro_dir.exists():
        st.info(f"No macro data at `{macro_dir}`.")
    else:
        macro_files = list(macro_dir.glob("*.csv"))
        if not macro_files:
            st.info("No CSV files found in macro_snapshots directory.")
        else:
            if st.button("Load macro data", key="load_macro"):
                with st.spinner("Loading…"):
                    try:
                        macro_frames: dict[str, pd.DataFrame] = {}
                        for f in macro_files:
                            df_m = pd.read_csv(f)
                            macro_frames[f.stem] = df_m
                        st.session_state["macro_data"] = macro_frames
                        st.success(f"Loaded {len(macro_frames)} macro files")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    macro_data: dict = st.session_state.get("macro_data", {})
    for name, df_m in macro_data.items():
        numeric_cols = df_m.select_dtypes("number").columns.tolist()
        if numeric_cols:
            col_pick = st.selectbox(f"{name} — column", numeric_cols, key=f"macro_col_{name}")
            st.plotly_chart(macro_chart(df_m, col_pick, f"{name} / {col_pick}"),
                            use_container_width=True)
