"""
Strategy Explorer & Backtester — unified EDA, options and backtest dashboard.

Launch from project root:
    streamlit run app/Explorer.py
"""

import inspect
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

from components.lse_data import TIMEFRAMES, BACKTEST_TIMEFRAMES, build_universe, get_api_key, load_bars_cached, fetch_catalog
from components import options_tab, ui
from components.charts import (
    atr_chart, bollinger_traces, bootstrap_ci_chart, candlestick_chart,
    equity_chart, macd_chart, mc_distribution_chart, mc_fan_chart,
    permutation_null_chart, regime_bar_chart, returns_hist_chart, rsi_chart,
    signal_log_chart, sweep_bar_chart, sweep_heatmap, trade_markers,
    vol_regime_chart, volume_bars,
)
from components.forms import backtest_config_form, signal_form, sizer_form, stop_form
from components.style import inject

st.set_page_config(
    page_title="Strategy Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()
st.title("Strategy Explorer")

# ── Mandatory, preloaded LSE credentials ──────────────────────────────────────
# The key is read from the environment only — never shown or editable in the UI.

api_key = get_api_key()
if not api_key:
    st.error(
        "LSE API key is required. Add `LSE_DATA=your_key` to the project `.env` file, "
        "then reload this page."
    )
    st.stop()


def _to_naive(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return dti.tz_convert("UTC").tz_localize(None) if dti.tz is not None else dti


# The Market tab's asset picker owns this key. The sidebar renders before the tab
# body, so it reads the selection from session state — Streamlit reruns the whole
# script on a change, and by then the new value is already stored.
_VIEW_ASSET_KEY = "market_asset"


def _parse_value_spec(raw: str, cast, max_values: int = 500) -> list:
    """
    Parse a parameter-sweep value spec into a sorted, de-duplicated list.

    Accepts comma-separated literals and `start:stop:step` ranges (inclusive of `stop`
    when it lands on the grid); `start:stop` implies step 1. The two mix freely::

        "5, 10, 15"          → [5, 10, 15]
        "20:100:20"          → [20, 40, 60, 80, 100]
        "5, 20:100:20, 250"  → [5, 20, 40, 60, 80, 100, 250]

    `cast` is int or float, taken from the strategy parameter's default. Raises
    ValueError with a user-facing message on malformed input or over-large ranges.
    """
    out: list = []
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        if ":" not in tok:
            try:
                out.append(cast(tok))
            except ValueError:
                raise ValueError(f"{tok!r} is not a valid {cast.__name__} value.") from None
            continue

        parts = [p.strip() for p in tok.split(":")]
        if len(parts) not in (2, 3) or not all(parts):
            raise ValueError(f"Bad range {tok!r} — use start:stop or start:stop:step.")
        try:
            start, stop = float(parts[0]), float(parts[1])
            step = float(parts[2]) if len(parts) == 3 else 1.0
        except ValueError:
            raise ValueError(f"Bad range {tok!r} — start, stop and step must be numbers.") from None
        if step <= 0:
            raise ValueError(f"Step must be positive in {tok!r}.")
        if stop < start:
            raise ValueError(f"Range {tok!r} ends before it starts.")
        # Count-based generation (not accumulation) so float steps don't drift.
        n = int((stop - start) / step + 1e-9) + 1
        if n > max_values:
            raise ValueError(f"Range {tok!r} expands to {n} values — limit is {max_values}.")
        out.extend(cast(round(start + i * step, 10)) for i in range(n))

    uniq = sorted(set(out))
    if len(uniq) > max_values:
        raise ValueError(f"{len(uniq)} values requested — limit is {max_values}.")
    return uniq


# ── Indicator defaults (used even before data loads) ──────────────────────────

overlay_ema = False
overlay_sma = False
overlay_bb = False
show_rsi = True
show_atr = False
show_macd = False
show_vol_regime = False
ema_fast, ema_slow = 12, 26
sma_period = 50
bb_window, bb_std = 20, 2.0
rsi_period = 14
atr_period = 14
macd_fast, macd_slow, macd_sig = 12, 26, 9
vol_window = 20

# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sb(text: str) -> None:
    """Render a lightweight sidebar section label."""
    st.markdown(f'<p class="sb-label">{text}</p>', unsafe_allow_html=True)


with st.sidebar:
    # ── Config ────────────────────────────────────────────────────────────
    # Capital and risk sit at the top: they frame every number the rest of the
    # app reports, so they are read before anything is selected or loaded.
    _sb("Config")
    config = backtest_config_form(st.sidebar, key_prefix="cfg")

    # ── Symbol ────────────────────────────────────────────────────────────
    st.divider()
    _sb("Symbol")

    _CAT_KEY = "_lse_catalog"
    if _CAT_KEY not in st.session_state:
        with st.spinner("Loading symbol catalog…"):
            _rows = fetch_catalog(api_key)
            st.session_state[_CAT_KEY] = _rows if _rows else []

    _catalog: list[dict] = st.session_state.get(_CAT_KEY) or []

    _NO_CANDLE_TERMS = ("option", "economic", "bond", "yield", "derivative", "credit")

    def _has_candle_data(r: dict) -> bool:
        cat     = r.get("category", "").lower()
        dataset = r.get("dataset",  "").lower()
        has_history = bool(r.get("first"))
        excluded = any(t in cat or t in dataset for t in _NO_CANDLE_TERMS)
        return has_history and not excluded

    if _catalog:
        _candle_catalog = [r for r in _catalog if _has_candle_data(r)]

        _cat_col, _ref_col = st.columns([4, 1])
        _raw_cats = sorted({r.get("category", "") for r in _candle_catalog if r.get("category")})
        _sel_cat = _cat_col.selectbox(
            "Asset type",
            ["(All)"] + _raw_cats,
            key="cat_filter",
            format_func=lambda x: x.title() if x != "(All)" else x,
            label_visibility="collapsed",
        )
        if _ref_col.button("↻", key="refresh_catalog", help="Reload symbol list"):
            st.session_state.pop(_CAT_KEY, None)
            st.rerun()

        _filtered = _candle_catalog if _sel_cat == "(All)" else [
            r for r in _candle_catalog if r.get("category") == _sel_cat
        ]
        _seen: set[str] = set()
        _deduped: list[dict] = []
        for _r in sorted(_filtered, key=lambda r: r.get("symbol", "")):
            if _r.get("symbol") and _r["symbol"] not in _seen:
                _seen.add(_r["symbol"])
                _deduped.append(_r)

        def _sym_label(r: dict) -> str:
            base = r["symbol"]
            if r.get("name"):
                base += f" — {r['name']}"
            if r.get("country"):
                base += f" ({r['country']})"
            return base

        _labels = [_sym_label(r) for r in _deduped]
        _default_idx = next(
            (i for i, r in enumerate(_deduped) if r["symbol"] == "AAPL"), 0
        )
        _choice_idx = st.selectbox(
            "Symbol", range(len(_labels)), index=_default_idx,
            format_func=lambda i: _labels[i],
            key="sym_cat",
            label_visibility="collapsed",
        )
        _chosen = _deduped[_choice_idx]
        symbol = _chosen["symbol"]

        def _date_only(val) -> str:
            if not val:
                return "—"
            try:
                return pd.to_datetime(val).strftime("%Y-%m-%d")
            except Exception:
                return str(val)[:10]

        st.caption(f"{_date_only(_chosen.get('first'))} → {_date_only(_chosen.get('last'))}")
    else:
        symbol = st.text_input("Symbol", value="AAPL", key="sym").upper()
        st.caption("Symbol catalog unavailable — check connectivity, then reload.")

    # ── Period ────────────────────────────────────────────────────────────
    _sb("Period")
    timeframe = st.selectbox(
        "Timeframe", TIMEFRAMES, index=6, key="tf", label_visibility="collapsed",
    )
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("From", value=date.today() - timedelta(days=730),
                                  min_value=date(1990, 1, 1), key="start")
    end_date   = col_e.date_input("To",   value=date.today(),
                                  min_value=date(1990, 1, 1), key="end")

    # ── Universe ──────────────────────────────────────────────────────────
    st.divider()
    _pending_syms: list[str] = list(st.session_state.get("_uni_pending", []))

    _uni_lbl_col, _uni_clr_col = st.columns([3, 2])
    _uni_lbl_col.markdown('<p class="sb-label">Universe</p>', unsafe_allow_html=True)
    if _pending_syms and _uni_clr_col.button(
        "Clear all", key="clear_uni", help="Remove all queued assets"
    ):
        st.session_state["_uni_pending"] = []
        st.rerun()

    if st.button(f"＋ Add  {symbol}", use_container_width=True, key="add_to_uni"):
        if symbol not in _pending_syms:
            st.session_state["_uni_pending"] = _pending_syms + [symbol]
        st.rerun()

    if _pending_syms:
        for _psym in list(_pending_syms):
            _pc1, _pc2 = st.columns([5, 1])
            _pc1.markdown(f'<div class="uni-chip">{_psym}</div>', unsafe_allow_html=True)
            if _pc2.button("×", key=f"rm_uni_{_psym}", help=f"Remove {_psym}"):
                st.session_state["_uni_pending"] = [s for s in _pending_syms if s != _psym]
                st.rerun()
    else:
        st.caption("Add symbols then click Load to backtest the universe.")

    _n_pending = len(_pending_syms)
    load_btn = st.button(
        f"Load Universe  ({_n_pending})" if _n_pending else "Load Data",
        type="primary", use_container_width=True, key="load",
    )

    _loaded_syms = st.session_state.get("universe_symbols", [])
    df_loaded: pd.DataFrame | None = st.session_state.get("main_ohlcv")
    if _loaded_syms:
        st.caption(f"Loaded: {' · '.join(_loaded_syms)}")

    # ── Chain ─────────────────────────────────────────────────────────────
    # Sits under Universe because its underlying comes from the loaded assets —
    # specifically whichever one the Market tab has selected. The state it returns
    # drives the option-chain section of that tab.
    st.divider()
    _sb("Chain")
    _chain_state = options_tab.sidebar_controls(
        st.session_state.get(_VIEW_ASSET_KEY) or (_loaded_syms[0] if _loaded_syms else None),
        api_key,
    )

    # ── Indicators (only when data is loaded) ─────────────────────────────
    if df_loaded is not None:
        st.divider()
        _sb("Indicators")

        overlay_ema = st.checkbox("EMA", value=False, key="ema")
        if overlay_ema:
            _ei1, _ei2 = st.columns(2)
            ema_fast = int(_ei1.number_input("Fast", value=12, step=1, min_value=2, key="ef"))
            ema_slow = int(_ei2.number_input("Slow", value=26, step=1, min_value=2, key="es"))

        overlay_sma = st.checkbox("SMA", value=False, key="sma")
        if overlay_sma:
            sma_period = int(st.number_input("SMA period", value=50, step=1, min_value=2, key="sp"))

        overlay_bb = st.checkbox("Bollinger Bands", value=False, key="bb")
        if overlay_bb:
            _bi1, _bi2 = st.columns(2)
            bb_window = int(_bi1.number_input("Window", value=20,  step=1,   min_value=5,   key="bbw"))
            bb_std    =     _bi2.number_input("Std",    value=2.0, step=0.5, min_value=0.5, key="bbs")

        show_rsi = st.checkbox("RSI", value=False, key="rsi")
        if show_rsi:
            rsi_period = int(st.number_input("RSI period", value=14, step=1, min_value=2, key="rp"))

        show_atr = st.checkbox("ATR", value=False, key="atr")
        if show_atr:
            atr_period = int(st.number_input("ATR period", value=14, step=1, min_value=2, key="ap"))

        show_macd = st.checkbox("MACD", value=False, key="macd")
        if show_macd:
            _mi1, _mi2, _mi3 = st.columns(3)
            macd_fast = int(_mi1.number_input("Fast",   value=12, step=1, min_value=2, key="mf"))
            macd_slow = int(_mi2.number_input("Slow",   value=26, step=1, min_value=2, key="ms"))
            macd_sig  = int(_mi3.number_input("Signal", value=9,  step=1, min_value=2, key="mg"))

        show_vol_regime = st.checkbox("Volatility Regime", value=False, key="vreg")
        if show_vol_regime:
            vol_window = int(st.number_input("Vol window", value=20, step=1, min_value=5, key="vw"))

    # ── Strategy ──────────────────────────────────────────────────────────
    st.divider()
    _sb("Signal")
    signal_cls, sig_params = signal_form(st.sidebar, key_prefix="sig")

    st.divider()
    _sb("Sizer")
    sizer = sizer_form(st.sidebar, key_prefix="sizer")

    st.divider()
    _sb("Stop Loss")
    stop = stop_form(st.sidebar, key_prefix="stop")

    st.divider()
    can_backtest = timeframe in BACKTEST_TIMEFRAMES
    run_bt = st.button(
        "Run Backtest", type="primary", use_container_width=True,
        disabled=not can_backtest,
        help=None if can_backtest else f"'{timeframe}' is not supported for backtesting.",
    )

# ── Data loading ──────────────────────────────────────────────────────────────

if load_btn:
    _pending = list(st.session_state.get("_uni_pending", []))
    _all_symbols = _pending if _pending else [symbol]

    _universe_dfs: dict[str, pd.DataFrame] = {}
    with st.spinner(f"Fetching {', '.join(_all_symbols)} {timeframe} bars from LSE…"):
        for _sym in _all_symbols:
            _df_sym = load_bars_cached(
                _sym, timeframe,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                api_key,
                cache_key_prefix="main",
            )
            if _df_sym is not None:
                _universe_dfs[_sym] = _df_sym

    if _universe_dfs:
        _primary = (
            _all_symbols[0] if _all_symbols[0] in _universe_dfs
            else next(iter(_universe_dfs))
        )
        df = _universe_dfs[_primary]
        st.session_state["main_ohlcv"]       = df
        st.session_state["main_symbol"]      = _primary
        st.session_state["main_timeframe"]   = timeframe
        st.session_state["universe_ohlcv"]   = _universe_dfs
        st.session_state["universe_symbols"] = list(_universe_dfs.keys())
        for k in ["main_bt_result", "main_ht_results",
                  "main_sweep_result", "main_regime_result", "main_mc_result"]:
            st.session_state.pop(k, None)
        _failed = [_s for _s in _all_symbols if _s not in _universe_dfs]
        if _failed:
            st.warning(f"Could not load: {', '.join(_failed)}")
        if isinstance(df.index, pd.DatetimeIndex) and len(df):
            actual_start = df.index[0].date()
            if actual_start > start_date:
                st.info(
                    f"Earliest available bar for {_primary} is **{actual_start}** "
                    f"(requested from {start_date}). Loaded data from that point."
                )
        st.rerun()

# ── Run backtest ──────────────────────────────────────────────────────────────

if run_bt:
    stored_ohlcv = st.session_state.get("main_ohlcv")
    if signal_cls is None:
        st.error("No signal selected.")
    elif stored_ohlcv is None:
        st.error("Load data first.")
    else:
        bt_sym = st.session_state.get("main_symbol", symbol)
        bt_tf  = st.session_state.get("main_timeframe", timeframe)
        _bt_uni_ohlcv = st.session_state.get("universe_ohlcv") or {bt_sym: stored_ohlcv}
        _bt_uni_syms  = st.session_state.get("universe_symbols") or [bt_sym]
        with st.spinner("Running backtest…"):
            try:
                from strategy.built_in import SingleAssetStrategy
                from testing.backtester.engine import Backtester
                from testing.backtester.costs import CompositeCostModel, aggressive_cost_stack

                strategy = signal_cls(symbol=bt_sym, **sig_params) \
                    if issubclass(signal_cls, SingleAssetStrategy) \
                    else signal_cls(**sig_params)

                uni = build_universe(_bt_uni_syms, _bt_uni_ohlcv) \
                    if len(_bt_uni_syms) > 1 \
                    else build_universe(bt_sym, stored_ohlcv)
                cost = CompositeCostModel(models=aggressive_cost_stack())
                bt   = Backtester(strategy=strategy, config=config, cost_model=cost,
                                  sizer=sizer, stop_loss=stop)
                result_new = bt.run(universe=uni, timeframe=bt_tf)

                st.session_state["main_bt_result"] = result_new
                for k in ["main_ht_results", "main_sweep_result",
                          "main_regime_result", "main_mc_result"]:
                    st.session_state.pop(k, None)
                _meta = getattr(result_new, "meta", None) or {}
                st.success(
                    "Backtest complete — vectorised fast path."
                    if _meta.get("vectorized") else "Backtest complete."
                )
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.exception(e)

df: pd.DataFrame | None = st.session_state.get("main_ohlcv")
main_symbol:    str = st.session_state.get("main_symbol", symbol)
main_timeframe: str = st.session_state.get("main_timeframe", timeframe)
result = st.session_state.get("main_bt_result")

# ── Tabs ──────────────────────────────────────────────────────────────────────

(tab_market, tab_result, tab_sweep,
 tab_regime, tab_mc, tab_hypothesis) = st.tabs([
    "Market", "Result", "Sweep", "Regime", "Simulation", "Hypothesis Tests",
])

# ── Guard: no bars loaded yet ─────────────────────────────────────────────────
# Fill every tab with the same empty state, then stop — nothing below this point
# works without price history (the option chain now takes its underlying from the
# loaded universe too).

if df is None:
    for _tab in (tab_market, tab_result, tab_sweep, tab_regime, tab_mc, tab_hypothesis):
        with _tab:
            ui.empty_state(
                "Enter a symbol and date range in the sidebar, then click **Load Data**."
            )
    st.stop()

# ════════════════════════════════════════════════════════════════════ Market tab

with tab_market:
    _exp_uni_syms  = st.session_state.get("universe_symbols", [main_symbol])
    _exp_uni_ohlcv = st.session_state.get("universe_ohlcv", {main_symbol: df})

    ui.intro(
        "Price, indicators and the live option surface for one asset — the spot view "
        "and the vol view describe the same instrument."
    )

    with ui.panel():
        # Always rendered, even for a single-asset universe: this widget is the one
        # asset selector for the tab, and the sidebar chain controls read its value
        # from session state to decide which underlying to fetch.
        vc0, vc1, vc2 = st.columns([2, 2, 2])
        _view_sym = vc0.selectbox("Asset", _exp_uni_syms, index=0, key=_VIEW_ASSET_KEY)
        _view_df = _exp_uni_ohlcv.get(_view_sym, df)

        idx_naive = _to_naive(_view_df.index)
        min_date = idx_naive.min().date()
        max_date = idx_naive.max().date()
        view_start = vc1.date_input("View from", value=min_date, min_value=min_date,
                                    max_value=max_date, key="vs")
        view_end   = vc2.date_input("View to", value=max_date, min_value=min_date,
                                    max_value=max_date, key="ve")

    mask = (
        (idx_naive >= pd.Timestamp(view_start)) &
        (idx_naive < pd.Timestamp(view_end) + pd.Timedelta(days=1))
    )
    df_view = _view_df[mask]

    if df_view.empty:
        st.warning("No data in the selected view range.")
        st.stop()

    returns = df_view["close"].pct_change().dropna()
    _tot_ret = (df_view["close"].iloc[-1] / df_view["close"].iloc[0] - 1) * 100

    st.divider()

    ui.metric_row([
        {"label": "Last close", "value": f"${df_view['close'].iloc[-1]:,.4f}"},
        {"label": "Bars", "value": f"{len(df_view):,}"},
        {"label": "Ann. vol",
         "value": f"{returns.std() * (252**0.5) * 100:.1f}%" if len(returns) > 1 else "—",
         "help": "Standard deviation of bar returns, scaled to a calendar year."},
        {"label": "Buy & hold", "value": f"{_tot_ret:,.2f}%",
         "help": "What the asset itself did over this window — the bar any strategy "
                 "has to clear."},
    ])

    from strategy.indicators import ema, sma, bollinger, rsi as _rsi, atr as _atr

    overlays: dict = {}
    if overlay_ema:
        overlays[f"EMA {ema_fast}"] = ema(df_view["close"], ema_fast)
        overlays[f"EMA {ema_slow}"] = ema(df_view["close"], ema_slow)
    if overlay_sma:
        overlays[f"SMA {sma_period}"] = sma(df_view["close"], sma_period)

    fig = candlestick_chart(df_view, overlays=overlays, title=f"{_view_sym} — {main_timeframe}")

    if overlay_bb:
        mid, upper, lower = bollinger(df_view["close"], window=bb_window, num_std=bb_std)
        for trace in bollinger_traces(mid, upper, lower):
            fig.add_trace(trace)

    # No trade markers here, deliberately: Market stays a view of the instrument
    # itself, unchanged by whether a backtest has been run. Trades on price live on
    # the Result tab.
    ui.section("Price")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(volume_bars(df_view), use_container_width=True)

    if show_rsi or show_atr or show_macd or show_vol_regime:
        ui.section("Indicators")
    if show_rsi:
        st.plotly_chart(
            rsi_chart(_rsi(df_view["close"], period=rsi_period), period=rsi_period),
            use_container_width=True,
        )
    if show_atr:
        st.plotly_chart(
            atr_chart(_atr(df_view["high"], df_view["low"], df_view["close"], period=atr_period),
                      period=atr_period),
            use_container_width=True,
        )
    if show_macd:
        st.plotly_chart(
            macd_chart(df_view["close"], fast=macd_fast, slow=macd_slow, signal=macd_sig),
            use_container_width=True,
        )
    if show_vol_regime:
        rv   = df_view["close"].pct_change().rolling(vol_window).std() * (252**0.5) * 100
        q_lo = rv.expanding(min_periods=vol_window).quantile(0.33)
        q_hi = rv.expanding(min_periods=vol_window).quantile(0.66)
        st.plotly_chart(vol_regime_chart(rv, q_lo, q_hi, vol_window),
                        use_container_width=True)

    # ── Option chain for the same asset ───────────────────────────────────
    st.divider()
    _chain_built = options_tab.render_section(_chain_state, api_key)

    # ── Tables and downloads, last ────────────────────────────────────────
    ui.section("Data")
    with st.expander("Returns analysis"):
        from scipy import stats as sp_stats

        col_dist, col_stats_ex = st.columns([3, 2])
        with col_dist:
            st.plotly_chart(returns_hist_chart(returns), use_container_width=True)

        with col_stats_ex:
            r = returns.dropna()
            ann = 252**0.5
            sharpe = (r.mean() / r.std() * ann) if r.std() > 0 else 0
            skew = float(sp_stats.skew(r))
            kurt = float(sp_stats.kurtosis(r))
            _, pval_norm = sp_stats.normaltest(r)
            st.table(pd.DataFrame({
                "Metric": [
                    "Mean bar return %", "Std bar return %", "Ann. vol %",
                    "Ann. return %", "Sharpe (ann.)",
                    "Skewness", "Excess kurtosis", "Normality p-val",
                ],
                "Value": [
                    f"{r.mean()*100:.4f}", f"{r.std()*100:.4f}", f"{r.std()*ann*100:.2f}",
                    f"{((1+r.mean())**252 - 1)*100:.2f}", f"{sharpe:.3f}",
                    f"{skew:.3f}", f"{kurt:.3f}", f"{pval_norm:.4f}",
                ],
            }))

    with st.expander(f"Raw OHLCV — {len(df_view):,} bars"):
        st.dataframe(
            df_view.style.format({c: "{:,.4f}" for c in ["open", "high", "low", "close"]}),
            use_container_width=True,
        )
        csv = df_view.reset_index().to_csv(index=False).encode()
        st.download_button(
            "Download CSV", csv,
            file_name=f"{_view_sym}_{main_timeframe}.csv", mime="text/csv",
        )

    if _chain_built:
        options_tab.chain_table_expander(_chain_built)

# ════════════════════════════════════════════════════════════════════ Result tab

with tab_result:
    if result is None:
        ui.empty_state("Configure a signal in the sidebar and click **Run Backtest**.")
    else:
        summary = result.summary()

        # ── Horizon mode ──────────────────────────────────────────────────
        is_annual  = summary.get("annualized", True)
        pfx        = "ann" if is_annual else "period"
        scale_lbl  = "Ann." if is_annual else "Period"

        _eq_tmp = result.equity_curve
        if isinstance(_eq_tmp.index, pd.DatetimeIndex) and len(_eq_tmp) > 1:
            _cal_years = (_eq_tmp.index[-1] - _eq_tmp.index[0]).total_seconds() / (365.25 * 24 * 3600)
            _cal_label = f"{_cal_years:.2f} yr"
        else:
            _cal_label = "—"

        _bt_meta = getattr(result, "meta", None) or {}
        ui.intro(
            f"{_cal_label} of {main_timeframe} bars · "
            f"{'annualised' if is_annual else 'period-only'} metrics · "
            f"{'vectorised' if _bt_meta.get('vectorized') else 'event-loop'} engine"
        )
        if not is_annual:
            ui.verdict(
                "warn",
                f"**Sub-year backtest (~{_cal_label})** — Sharpe, Sortino, return and vol "
                "describe the observed period only and are not extrapolated to a year. "
                "CAGR equals total return.",
            )

        # ── Headline: did it make money, and what did it risk ─────────────
        _ret = summary.get("total_return_pct", 0)
        _dd  = summary.get("max_drawdown_pct", 0)
        _sharpe = summary.get("sharpe_ratio", 0)

        ui.section("Performance")
        ui.metric_row([
            {"label": "Total return", "value": f"{_ret:,.2f}%"},
            {"label": "CAGR (ann.)", "value": f"{summary.get('cagr_pct', 0):,.2f}%"} if is_annual else None,
            {"label": f"Sharpe ({scale_lbl})", "value": f"{_sharpe:,.3f}",
             "help": "Return per unit of volatility. Below ~1 is usually noise."},
            {"label": "Max drawdown", "value": f"{_dd:,.2f}%",
             "help": "Deepest peak-to-trough fall in equity."},
            {"label": "Trades", "value": f"{summary.get('num_trades', 0):,}"},
        ])

        _calmar_val = summary.get("calmar_ratio")
        _calmar_str = f"{_calmar_val:,.3f}" if isinstance(_calmar_val, float) else "∞"
        _calmar_lbl = "Calmar (CAGR/MaxDD)" if is_annual else "Recovery (Return/MaxDD)"
        _pf_val = summary.get("profit_factor")
        _pf_str = f"{_pf_val:,.3f}" if isinstance(_pf_val, float) else "∞"

        ui.section("Risk & quality")
        ui.metric_row([
            {"label": _calmar_lbl, "value": _calmar_str,
             "help": "Return earned per unit of worst-case drawdown."},
            {"label": f"Sortino ({scale_lbl})", "value": f"{summary.get('sortino_ratio', 0):,.3f}",
             "help": "Like Sharpe, but only downside volatility is penalised."},
            {"label": f"Vol ({scale_lbl})", "value": f"{summary.get(f'{pfx}_volatility_pct') or 0.0:,.2f}%"},
            {"label": "Profit factor", "value": _pf_str,
             "help": "Gross wins ÷ gross losses. Below 1.0 loses money."},
            {"label": "Win rate", "value": f"{summary.get('win_rate_pct', 0):,.1f}%",
             "help": "Share of trades that closed positive — on its own it says "
                     "nothing about profitability."},
        ])

        ui.section("Costs & exposure")
        ui.metric_row([
            {"label": "Total fees", "value": f"${summary.get('total_fees', 0):,.2f}",
             "help": "What the strategy paid to trade — compare against total return."},
            {"label": "Avg win", "value": f"{summary.get('avg_win_pct', 0):,.2f}%"},
            {"label": "Avg loss", "value": f"{summary.get('avg_loss_pct', 0):,.2f}%"},
            {"label": "% in market", "value": f"{summary.get('pct_in_market', 0):,.1f}%",
             "help": "Share of bars holding a position. Low values mean the Sharpe "
                     "rests on few observations."},
        ])

        # Fees eating the edge is the single most common silent failure.
        _fees = summary.get("total_fees", 0) or 0
        _gross_gain = result.config.initial_capital * (_ret / 100)
        if _fees > 0 and _gross_gain > 0 and _fees > _gross_gain:
            ui.verdict("warn", f"Fees (${_fees:,.2f}) exceed net profit "
                               f"(${_gross_gain:,.2f}) — the edge is being paid to the broker.")
        elif _ret > 0 and _sharpe < 0.5:
            ui.verdict("warn", f"Profitable but low Sharpe ({_sharpe:.2f}) — the return "
                               "is not well separated from noise. Check Hypothesis Tests.")
        elif _ret <= 0:
            ui.verdict("bad", f"The strategy lost {abs(_ret):,.2f}% over this period.")

        st.divider()

        ui.section("Equity curve")
        eq = result.equity_curve
        dd = (eq - eq.cummax()) / eq.cummax()
        st.plotly_chart(equity_chart(eq, dd), use_container_width=True)

        trades_df = result.trades_df()

        _rt_uni_syms  = st.session_state.get("universe_symbols", [main_symbol])
        _rt_uni_ohlcv = st.session_state.get("universe_ohlcv", {main_symbol: df})

        if len(_rt_uni_syms) > 1:
            ui.section("Trades by asset")
            for _rt_sym in _rt_uni_syms:
                _rt_ohlcv = _rt_uni_ohlcv.get(_rt_sym)
                if _rt_ohlcv is None:
                    continue
                with st.expander(f"{_rt_sym}", expanded=True):
                    _rt_fig = candlestick_chart(_rt_ohlcv, title=f"{_rt_sym} — {main_timeframe}")
                    if not trades_df.empty:
                        _rt_sym_trades = (
                            trades_df[trades_df["meta_symbol"] == _rt_sym]
                            if "meta_symbol" in trades_df.columns
                            else trades_df
                        )
                        if not _rt_sym_trades.empty:
                            _rt_fig = trade_markers(_rt_fig, _rt_sym_trades)
                    st.plotly_chart(_rt_fig, use_container_width=True)
        else:
            ui.section("Trades on price")
            price_fig = candlestick_chart(df, title=f"{main_symbol} — {main_timeframe}")
            if not trades_df.empty:
                price_fig = trade_markers(price_fig, trades_df)
            st.plotly_chart(price_fig, use_container_width=True)

        sig_log = getattr(result, "signal_log", None)
        if sig_log is not None and not sig_log.empty:
            ui.section("Signal log")
            st.plotly_chart(signal_log_chart(sig_log, height=320), use_container_width=True)
            with st.expander("Signal log table"):
                display_log = sig_log.copy()
                if "timestamp" in display_log.columns:
                    ts_conv = pd.to_datetime(display_log["timestamp"], unit="ms", errors="coerce")
                    if not ts_conv.isna().all():
                        display_log["timestamp"] = ts_conv

                def _color_side(val):
                    return {"LONG": "color: #26a69a", "SHORT": "color: #ef5350"}.get(val, "color: #9ba3b8")

                st.dataframe(
                    display_log.style.map(
                        _color_side,
                        subset=["side"] if "side" in display_log.columns else [],
                    ),
                    use_container_width=True,
                )

        if trades_df.empty:
            ui.empty_state("No completed trades.")
        else:
            with st.expander(f"Trade log ({len(trades_df):,} trades)"):
                def _color_pnl(val):
                    if isinstance(val, (int, float)):
                        return f"color: {'#26a69a' if val > 0 else '#ef5350' if val < 0 else 'inherit'}"
                    return ""

                display_cols = [c for c in [
                    "timestamp", "meta_symbol", "side", "size", "entry_price", "exit_price",
                    "pnl", "pnl_pct", "fees", "reason_entry", "reason_exit",
                ] if c in trades_df.columns]
                st.dataframe(
                    trades_df[display_cols].style.map(
                        _color_pnl, subset=["pnl"] if "pnl" in display_cols else [],
                    ),
                    use_container_width=True,
                )
                csv = trades_df[display_cols].to_csv(index=False).encode()
                st.download_button(
                    "Download trades CSV", csv,
                    file_name=f"{main_symbol}_trades.csv", mime="text/csv",
                )

# ══════════════════════════════════════════════════════════ Hypothesis Tests tab

# Friendly names for the battery — the raw test ids read as internals.
_TEST_LABELS = {
    "sharpe_significance": "Sharpe ≠ 0",
    "mean_return": "Mean return > 0",
    "normality": "Returns normal",
    "stationarity": "Returns stationary",
}

with tab_hypothesis:
    if result is None:
        ui.empty_state("Run a backtest first (Result tab).")
    else:
        ui.intro(
            "Ask whether this result is distinguishable from luck. Every test reports a "
            "p-value — the chance of seeing a result this good if the strategy had no edge."
        )
        with ui.panel():
            ht_col1, ht_col2 = st.columns(2)
            n_permutations = ht_col1.number_input(
                "Permutation samples", value=2_000, step=500, min_value=200, key="ht_perms",
                help="Reshuffles of the trade order used to build the null distribution.",
            )
            n_bootstrap = ht_col2.number_input(
                "Bootstrap samples", value=2_000, step=500, min_value=200, key="ht_boot",
                help="Resamples of the trade set used to build the confidence intervals.",
            )
            ci_level = st.slider("Confidence level", min_value=0.80, max_value=0.99,
                                 value=0.95, step=0.01, key="ht_ci")
            run_ht_clicked = ui.run_button("Run Hypothesis Tests", "run_ht")

        if run_ht_clicked:
            try:
                from testing.hypothesis import HypothesisTests, PermutationTest, BootstrapCI
                with st.spinner("Running hypothesis tests…"):
                    tests = HypothesisTests.run_all(result)
                    pt    = PermutationTest(metric="sharpe_ratio",
                                            n_permutations=int(n_permutations)).run(result)
                    ci    = BootstrapCI(n_bootstrap=int(n_bootstrap), ci=ci_level).run(result)
                st.session_state["main_ht_results"] = (tests, pt, ci)
            except Exception as e:
                st.error(f"Hypothesis tests failed: {e}")
                st.exception(e)

        ht_data = st.session_state.get("main_ht_results")
        if ht_data:
            # Slice, not unpack: tolerates results cached by an earlier session layout.
            tests, pt, ci = ht_data[:3]

            st.divider()

            # ── Permutation test: the headline question ───────────────────
            st.markdown("**Is the trade sequence better than chance?**")
            pv1, pv2 = st.columns([1, 2])
            with pv1:
                _verdict = "Beats chance" if pt.reject_null else "Indistinguishable"
                st.metric("Permutation p-value", f"{pt.p_value:.4f}",
                          delta=_verdict, delta_color="normal" if pt.reject_null else "off")
                st.caption(
                    f"{pt.meta.get('n_permutations', 0):,} reshuffles of "
                    f"{pt.meta.get('n_trades', 0)} trades."
                )
            with pv2:
                st.plotly_chart(permutation_null_chart(pt), use_container_width=True)

            # ── Bootstrap intervals ──────────────────────────────────────
            st.markdown(f"**How precise is each metric?**  ({int(ci_level * 100)}% intervals)")
            st.caption(
                "An interval that spans zero means the sign of that metric is not "
                "established by this many trades."
            )
            st.plotly_chart(bootstrap_ci_chart(ci), use_container_width=True)

            # ── Test battery as verdict cards ────────────────────────────
            st.markdown("**Test battery**")
            # Two per row: a fresh st.columns per pair, so each row's cards share a
            # baseline instead of one long interpretation stretching every card in a
            # single 3-wide row. Four tests lay out as a 2x2 grid.
            for _row_start in range(0, len(tests), 2):
                _cols = st.columns(2)
                for _col, t in zip(_cols, tests[_row_start:_row_start + 2]):
                    with _col, st.container(border=True):
                        _name = _TEST_LABELS.get(t.name, t.name.replace("_", " "))
                        _mark = "✅" if t.reject_null else "•"
                        st.markdown(f"{_mark} **{_name}**")
                        st.metric("p-value", f"{t.p_value:.4f}",
                                  delta="reject H₀" if t.reject_null else "no evidence",
                                  delta_color="normal" if t.reject_null else "off",
                                  label_visibility="collapsed")
                        st.caption(t.interpretation)

            with st.expander("Full report"):
                from testing.hypothesis import report
                st.code(report(tests))

# ══════════════════════════════════════════════════════════════ Param Sweep tab

with tab_sweep:
    if signal_cls is None:
        ui.empty_state("Select a signal in the sidebar first.")
    else:
        ui.intro(
            "Re-run the strategy across a grid of parameter values. Every combination "
            "tried is another chance to fit noise — keep the grid small and honest."
        )
        sweep_params = [
            k for k, v in inspect.signature(signal_cls.__init__).parameters.items()
            if k != "self"
            and v.default is not inspect.Parameter.empty
            and isinstance(v.default, (int, float))
        ]

        if not sweep_params:
            ui.empty_state("This strategy has no numeric parameters to sweep.")
        else:
            _VAL_HELP = (
                "Comma-separated values, `start:stop:step` ranges (stop inclusive), or a "
                "mix of both — e.g. `5,10,15`, `20:100:20`, or `5, 20:100:20, 250`. "
                "`start:stop` steps by 1."
            )

            with ui.panel():
                sc1, sc2 = st.columns(2)
                p1      = sc1.selectbox("Parameter 1", sweep_params, key="sweep_p1")
                p1_vals = sc1.text_input("Values or range", placeholder="e.g. 5,10,15 or 5:50:5",
                                         help=_VAL_HELP, key="sweep_p1v")

                p2_opts = ["(none)"] + [p for p in sweep_params if p != p1]
                p2      = sc2.selectbox("Parameter 2 (optional)", p2_opts, key="sweep_p2")
                p2_vals = sc2.text_input("Values or range", placeholder="e.g. 20,30,40 or 20:200:20",
                                         help=_VAL_HELP, key="sweep_p2v") \
                    if p2 != "(none)" else ""

                metric_pick = st.selectbox("Optimise metric", [
                    "sharpe_ratio", "total_return_pct", "max_drawdown_pct",
                    "calmar_ratio", "win_rate_pct", "profit_factor",
                ], key="sweep_metric")

                # Live preview so the grid size is known before committing to the run.
                try:
                    _sig_defaults = inspect.signature(signal_cls.__init__).parameters
                    _pv = _parse_value_spec(
                        p1_vals, int if isinstance(_sig_defaults[p1].default, int) else float)
                    _pv2 = _parse_value_spec(
                        p2_vals, int if isinstance(_sig_defaults[p2].default, int) else float) \
                        if p2_vals and p2 != "(none)" else []
                    if _pv:
                        _preview = f"{p1}: {_pv}"
                        if _pv2:
                            _preview += f" · {p2}: {_pv2}"
                        st.caption(f"{len(_pv) * max(len(_pv2), 1)} combo(s) — {_preview}")
                except ValueError as _e:
                    st.caption(f"⚠️ {_e}")

                run_sweep_clicked = ui.run_button("Run Sweep", "run_sweep")

            if run_sweep_clicked:
                _sw_sig = inspect.signature(signal_cls.__init__).parameters

                def _cast_sweep(name: str, raw: str) -> list:
                    default = _sw_sig[name].default if name in _sw_sig else 0.0
                    fn = int if isinstance(default, int) else float
                    return _parse_value_spec(raw, fn)

                # Parse outside the run's try/except so a bad spec reads as a warning
                # rather than a backtest traceback.
                vals1: list = []
                vals2: list = []
                try:
                    vals1 = _cast_sweep(p1, p1_vals)
                    vals2 = _cast_sweep(p2, p2_vals) if p2_vals and p2 != "(none)" else []
                    if not vals1:
                        st.warning("Enter at least one value or range for Parameter 1.")
                except ValueError as e:
                    st.warning(str(e))

                try:
                    if vals1:
                        from testing.backtester.costs import CompositeCostModel, aggressive_cost_stack
                        from testing.backtester.stress import ParamSweep

                        param_grid = {p1: vals1}
                        if vals2:
                            param_grid[p2] = vals2

                        # Pass current sidebar signal params as fixed_params for non-swept keys,
                        # so sweep is consistent with the rest of the session configuration.
                        swept_keys = set(param_grid.keys())
                        fixed = {k: v for k, v in sig_params.items() if k not in swept_keys}

                        n_combos = len(vals1) * max(len(vals2), 1)
                        _sw_uni_ohlcv = st.session_state.get("universe_ohlcv") or {main_symbol: df}
                        _sw_uni_syms  = st.session_state.get("universe_symbols") or [main_symbol]
                        uni = build_universe(_sw_uni_syms, _sw_uni_ohlcv) \
                            if len(_sw_uni_syms) > 1 \
                            else build_universe(main_symbol, df)
                        cost = CompositeCostModel(models=aggressive_cost_stack())

                        with st.spinner(f"Running {n_combos} backtest(s)…"):
                            sweep_result = ParamSweep(
                                strategy_cls=signal_cls,
                                param_grid=param_grid,
                                config=config,
                                cost_model=cost,
                                sizer=sizer,
                                stop_loss=stop,
                                fixed_params=fixed,
                            ).run(universe=uni, timeframe=main_timeframe)

                        st.session_state["main_sweep_result"] = (
                            sweep_result, p1, p2, metric_pick, vals1, vals2,
                        )
                        st.success(f"Sweep complete — {n_combos} trial(s).")
                except Exception as e:
                    st.error(f"Sweep failed: {e}")
                    st.exception(e)

    sweep_data = st.session_state.get("main_sweep_result")
    if sweep_data:
        _sw, _p1, _p2, _metric, _v1, _v2 = sweep_data
        _sdf: pd.DataFrame = _sw.summary

        _param_cols = [c for c in [_p1, _p2] if c and c != "(none)" and c in _sdf.columns]

        st.divider()

        # ── Best / worst ──────────────────────────────────────────────────
        _valid_col = _sdf[_metric].dropna() if _metric in _sdf.columns else pd.Series(dtype=float)
        if not _valid_col.empty:
            _best_row  = _sw.best(_metric)
            _worst_row = _sw.worst(_metric)

            def _py(v):
                """Unwrap numpy scalars so they render as 40, not np.int64(40)."""
                return v.item() if hasattr(v, "item") else v

            def _fmt_metric(v) -> str:
                v = _py(v)
                return f"{v:,.4f}" if isinstance(v, (int, float)) and pd.notna(v) else str(v)

            def _fmt_params(row) -> str:
                """`fast=40, slow=200` — flat and readable, no dict/numpy repr."""
                parts = []
                for k in _param_cols:
                    if k not in row.index:
                        continue
                    v = _py(row[k])
                    if isinstance(v, float) and pd.notna(v):
                        v = f"{v:g}"
                    parts.append(f"{k}={v}")
                return ", ".join(parts) or "—"

            ui.metric_row([
                {"label": f"Best {_metric}", "value": _fmt_metric(_best_row[_metric]),
                 "delta": _fmt_params(_best_row)},
                {"label": f"Worst {_metric}", "value": _fmt_metric(_worst_row[_metric]),
                 "delta": _fmt_params(_worst_row)},
                {"label": "Combos tried", "value": f"{len(_valid_col):,}",
                 "help": "Every combination is another chance to fit noise — the more "
                         "you try, the better the best one looks by luck alone."},
                {"label": "Spread", "value": _fmt_metric(_valid_col.max() - _valid_col.min()),
                 "help": "Best minus worst. A wide spread over a small grid means the "
                         "metric is highly parameter-sensitive."},
            ])

            # A peak that only exists at one grid point is a spike, not a plateau.
            _q75 = float(_valid_col.quantile(0.75))
            _near_best = int((_valid_col >= _q75).sum())
            if len(_valid_col) >= 4 and _near_best <= 1:
                ui.verdict("warn", "The best result is an isolated spike in the grid — "
                                   "neighbouring parameters do much worse, which usually "
                                   "means it is fitted to noise rather than a real plateau.")

        # ── Surface ───────────────────────────────────────────────────────
        ui.section("Parameter surface")
        if len(_param_cols) == 2 and _metric in _sdf.columns:
            _plot_df = _sdf[_param_cols + [_metric]].dropna(subset=[_metric])
            if not _plot_df.empty:
                _pivot = _plot_df.pivot(index=_p2, columns=_p1, values=_metric)
                st.plotly_chart(sweep_heatmap(_pivot, _metric, _p1, _p2),
                                use_container_width=True)
        elif _metric in _sdf.columns and _p1 in _sdf.columns:
            _plot_df = _sdf[[_p1, _metric]].dropna(subset=[_metric]).sort_values(_p1)
            if not _plot_df.empty:
                st.plotly_chart(
                    sweep_bar_chart(_plot_df[_p1], _plot_df[_metric], _metric, _p1),
                    use_container_width=True,
                )

        # ── Results table: param cols + key metrics only ──────────────────
        _KEY_SWEEP_METRICS = [
            "sharpe_ratio", "total_return_pct", "cagr_pct",
            "max_drawdown_pct", "calmar_ratio", "sortino_ratio",
            "win_rate_pct", "profit_factor", "num_trades", "total_fees",
        ]
        _show_cols = _param_cols + [c for c in _KEY_SWEEP_METRICS if c in _sdf.columns]
        if "error" in _sdf.columns:
            _show_cols.append("error")
        with st.expander(f"All {len(_sdf):,} results"):
            st.dataframe(_sdf[_show_cols].sort_values(_p1, ignore_index=True),
                         use_container_width=True)

# ═══════════════════════════════════════════════════════════════════ Regime tab

# Trend leads and is therefore the default: bull/bear is the split most strategies
# are actually exposed to, and it is the one a directional edge most often fails.
_REGIME_MODES = {
    "Trend (SMA)": "Splits bars by price above or below its moving average.",
    "Volatility": "Splits bars into low / medium / high realised-volatility regimes.",
    "Volume": "Splits bars into low / medium / high traded-volume regimes.",
}

with tab_regime:
    if result is None:
        ui.empty_state("Run a backtest first (Result tab).")
    elif signal_cls is None:
        ui.empty_state("Select a signal in the sidebar first.")
    else:
        ui.intro(
            "Re-run the strategy inside each market regime separately. An edge that only "
            "survives one regime is a bet on that regime persisting."
        )
        with ui.panel():
            rgc1, rgc2 = st.columns([2, 1])
            regime_choice = rgc1.selectbox("Classifier", list(_REGIME_MODES), key="regime_choice")
            sma_win = (
                rgc2.number_input("SMA window", value=50, step=5, min_value=10, key="regime_sma")
                if regime_choice == "Trend (SMA)" else 50
            )
            st.caption(_REGIME_MODES[regime_choice])
            run_regime_clicked = ui.run_button("Run Regime Test", "run_regime")

        if run_regime_clicked:
            try:
                from strategy.built_in import SingleAssetStrategy
                from testing.backtester.stress import RegimeStressTest
                from testing.backtester.costs import CompositeCostModel, aggressive_cost_stack

                strategy_reg = signal_cls(symbol=main_symbol, **sig_params) \
                    if issubclass(signal_cls, SingleAssetStrategy) \
                    else signal_cls(**sig_params)

                if regime_choice == "Volatility":
                    regime_fn = None
                elif regime_choice == "Trend (SMA)":
                    _w = sma_win
                    regime_fn = lambda df_: RegimeStressTest.trend_regime(df_, sma_window=_w)
                else:
                    regime_fn = RegimeStressTest.volume_regime

                cost = CompositeCostModel(models=aggressive_cost_stack())
                _rg_uni_ohlcv = st.session_state.get("universe_ohlcv") or {main_symbol: df}
                _rg_uni_syms  = st.session_state.get("universe_symbols") or [main_symbol]
                uni = build_universe(_rg_uni_syms, _rg_uni_ohlcv) \
                    if len(_rg_uni_syms) > 1 \
                    else build_universe(main_symbol, df)

                with st.spinner("Running regime stress test…"):
                    regime_new = RegimeStressTest(
                        regime_fn=regime_fn, config=config, cost_model=cost,
                    ).run(strategy=strategy_reg, universe=uni)

                st.session_state["main_regime_result"] = regime_new
            except Exception as e:
                st.error(f"Regime test failed: {e}")
                st.exception(e)

    regime_result = st.session_state.get("main_regime_result")
    if regime_result is not None and not regime_result.summary.empty:
        _rdf = regime_result.summary
        _failed = _rdf[_rdf["error"].notna()] if "error" in _rdf.columns else _rdf.iloc[:0]
        _ok = _rdf.drop(index=_failed.index) if not _failed.empty else _rdf
        # A regime can also come back without metrics (too few bars to score).
        if "sharpe_ratio" in _ok.columns:
            _ok = _ok.dropna(subset=["sharpe_ratio"])

        st.divider()

        if not _ok.empty and "sharpe_ratio" in _ok.columns:
            # Consistency across regimes is the headline: one great regime and two
            # bad ones is a worse strategy than three mediocre ones.
            _n_profit = int((_ok["total_return_pct"] > 0).sum()) \
                if "total_return_pct" in _ok.columns else 0
            _n_tot = len(_ok)
            _best = _ok.loc[_ok["sharpe_ratio"].idxmax()]
            _worst = _ok.loc[_ok["sharpe_ratio"].idxmin()]
            _spread = float(_best["sharpe_ratio"] - _worst["sharpe_ratio"])

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Profitable regimes", f"{_n_profit}/{_n_tot}")
            g2.metric("Best regime", str(_best["regime"]),
                      delta=f"Sharpe {_best['sharpe_ratio']:.2f}", delta_color="off")
            g3.metric("Worst regime", str(_worst["regime"]),
                      delta=f"Sharpe {_worst['sharpe_ratio']:.2f}", delta_color="off")
            g4.metric("Sharpe spread", f"{_spread:,.2f}",
                      help="Best minus worst regime Sharpe. A wide spread means the "
                           "edge is regime-dependent.")

            if _n_profit == _n_tot:
                st.success(f"Profitable in all {_n_tot} regimes — the edge is regime-agnostic.")
            elif _n_profit == 0:
                st.error("Unprofitable in every regime.")
            else:
                st.warning(
                    f"Profitable in {_n_profit} of {_n_tot} regimes — the edge depends on "
                    f"the market being in a **{_best['regime']}** state."
                )

            b1, b2 = st.columns(2)
            with b1:
                st.plotly_chart(regime_bar_chart(_ok, "sharpe_ratio", "Sharpe ratio"),
                                use_container_width=True)
            with b2:
                if "total_return_pct" in _ok.columns:
                    st.plotly_chart(regime_bar_chart(_ok, "total_return_pct", "Total return (%)"),
                                    use_container_width=True)

        if not _failed.empty:
            for _, row in _failed.iterrows():
                st.error(f"Regime **{row['regime']}** failed: {row['error']}")

        # ── Per-regime detail ─────────────────────────────────────────────
        for regime_name, res in regime_result.results.items():
            rs = res.summary()
            _ret = rs.get("total_return_pct", 0)
            with st.expander(
                f"{regime_name}  ·  {_ret:+.2f}%  ·  Sharpe {rs.get('sharpe_ratio', 0):.2f}  "
                f"·  {rs.get('num_trades', 0)} trades"
            ):
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Return", f"{_ret:,.2f}%")
                rc2.metric("Sharpe", f"{rs.get('sharpe_ratio', 0):,.3f}")
                rc3.metric("Max DD", f"{rs.get('max_drawdown_pct', 0):,.2f}%")
                rc4.metric("Trades", f"{rs.get('num_trades', 0)}")
                eq_r = res.equity_curve
                dd_r = (eq_r - eq_r.cummax()) / eq_r.cummax()
                st.plotly_chart(equity_chart(eq_r, dd_r), use_container_width=True)

        with st.expander("Regime summary table"):
            st.dataframe(_rdf, use_container_width=True)

# ═══════════════════════════════════════════════════════════════ Simulation tab

_MC_METHODS = {
    "bootstrap": ("Bootstrap", "Resample trades with replacement — tests luck of the draw."),
    "shuffle": ("Shuffle", "Reorder the same trades — tests luck of the sequence."),
    "block_bootstrap": ("Block bootstrap", "Resample runs of trades — keeps streaks intact."),
}

with tab_mc:
    if result is None:
        ui.empty_state("Run a backtest first (Result tab).")
    else:
        trades_df_mc = result.trades_df()
        if trades_df_mc.empty or "pnl" not in trades_df_mc.columns:
            st.warning("No trades with PnL data available to simulate.")
        else:
            ui.intro(
                f"Resample the {len(trades_df_mc)} realised trades to ask how much of this "
                "equity curve is edge and how much is draw order."
            )
            with ui.panel():
                mc1, mc2, mc3 = st.columns([2, 1, 1])
                mc_method = mc1.selectbox(
                    "Method", list(_MC_METHODS), key="mc_method",
                    format_func=lambda k: _MC_METHODS[k][0],
                )
                n_sims = mc2.number_input("Simulations", value=1000, step=100,
                                          min_value=100, max_value=10000, key="mc_n")
                mc_seed = mc3.number_input("Seed", value=42, step=1, key="mc_seed")
                st.caption(_MC_METHODS[mc_method][1])
                run_mc_clicked = ui.run_button("Run Simulation", "run_mc")

            if run_mc_clicked:
                try:
                    from testing.backtester.stress import MonteCarloStress
                    with st.spinner(f"Running {n_sims} simulations…"):
                        mc_new = MonteCarloStress(
                            n_simulations=int(n_sims), seed=int(mc_seed), method=mc_method,
                        ).run(result)
                    st.session_state["main_mc_result"] = mc_new
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    st.exception(e)

    mc_result = st.session_state.get("main_mc_result")
    if mc_result is not None and not mc_result.summary.empty:
        meta = mc_result.meta
        mc_df = mc_result.summary
        _obs_ret = meta.get("observed_return_pct")
        _obs_dd = meta.get("observed_max_dd_pct")
        _p_profit = meta.get("prob_profit")
        _med_ret = meta.get("median_return", 0.0)

        st.divider()

        # ── Headline: where the observed run sits in its own outcome cloud ──
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Median return", f"{_med_ret:,.2f}%",
                  delta=f"{_obs_ret - _med_ret:+.2f}% observed"
                  if _obs_ret is not None else None, delta_color="off")
        h2.metric("5th pctl return", f"{meta.get('5th_pctl_return', 0):,.2f}%",
                  help="A bad-luck draw: 1 run in 20 does worse than this.")
        h3.metric("95th pctl return", f"{meta.get('95th_pctl_return', 0):,.2f}%",
                  help="A good-luck draw: 1 run in 20 does better than this.")
        h4.metric("Median max DD", f"{meta.get('median_max_dd', 0):,.2f}%")
        h5.metric("P(profit)", f"{_p_profit * 100:,.1f}%" if _p_profit is not None else "—",
                  help="Share of simulated runs that finish above the starting capital.")

        # A run in the top decile of its own resamples is a warning, not a win.
        if _obs_ret is not None:
            _pctile = float((mc_df["total_return_pct"] < _obs_ret).mean() * 100)
            if _pctile >= 90:
                st.warning(
                    f"The observed run lands at the **{_pctile:.0f}th percentile** of its own "
                    "resamples — the realised curve is a lucky ordering more than a typical one."
                )
            elif _pctile <= 10:
                st.info(
                    f"The observed run lands at the **{_pctile:.0f}th percentile** of its own "
                    "resamples — the realised ordering was unusually unkind."
                )

        if meta.get("equity_bands"):
            st.plotly_chart(
                mc_fan_chart(meta["equity_bands"], observed=meta.get("observed_equity"),
                             initial=meta.get("initial_capital")),
                use_container_width=True,
            )

        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(
                mc_distribution_chart(mc_df["total_return_pct"], observed=_obs_ret,
                                      title="Total return"),
                use_container_width=True,
            )
        with d2:
            st.plotly_chart(
                mc_distribution_chart(mc_df["max_drawdown_pct"], observed=_obs_dd,
                                      title="Max drawdown"),
                use_container_width=True,
            )

        with st.expander("Simulation statistics"):
            st.dataframe(mc_df.describe(), use_container_width=True)
            st.download_button(
                "Download MC results CSV",
                mc_df.to_csv(index=False).encode(),
                file_name="monte_carlo_results.csv",
                mime="text/csv",
            )
