"""
Trading Dashboard — Streamlit entry point.

Launch from project root:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# ── Path setup (must happen before any internal imports) ──────────────────────
_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
_SRC = _ROOT / "src"
for _p in [str(_SRC), str(_ROOT), str(_APP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from components.style import inject

# ── Page config (must be first Streamlit command) ─────────────────────────────

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()

# ── Signal discovery ──────────────────────────────────────────────────────────

@st.cache_resource
def _discover_signals():
    """Import known signal modules once so @register_signal decorators fire."""
    for mod in [
        "strategy.built_in",
        "trading.strategy_live_demo",
        "trading.strategy_backtest_demo",
    ]:
        try:
            __import__(mod)
        except Exception:
            pass

_discover_signals()

# ── Global session state ──────────────────────────────────────────────────────

if "runner" not in st.session_state:
    from components.engine_runner import EngineRunner
    st.session_state["runner"] = EngineRunner()

DATA_DIR = _ROOT / "data"
st.session_state["data_dir"] = DATA_DIR

# ── Home page ─────────────────────────────────────────────────────────────────

st.title("Trading Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Data Explorer**\n\nBrowse OHLCV, order book snapshots, funding rates, sentiment scores, and macro data.")
with col2:
    st.info("**Backtester**\n\nConfigure signals, sizers, and stops, run vectorised backtests, inspect equity curves and trade logs, sweep parameters.")
with col3:
    runner = st.session_state["runner"]
    status = runner.status
    if status == "running":
        st.success("**Live Engine**\n\nEngine is **running**. Go to the Live page to monitor it.")
    elif status == "error":
        st.error("**Live Engine**\n\nEngine stopped with an error. Check the Live page.")
    else:
        st.warning("**Live Engine**\n\nEngine is stopped. Go to the Live page to configure and launch it.")

st.divider()
st.caption(f"Data root: `{DATA_DIR}`")
