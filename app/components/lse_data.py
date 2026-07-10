"""LSE (London Strategic Edge) historical bar fetching for the UI."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "src"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Timeframes offered in the UI — all supported by the LSE API
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

# Subset safe to pass to core's timeframe_to_seconds / backtester
BACKTEST_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def get_api_key(override: str = "") -> str:
    """Return the LSE API key: UI field if filled, else .env LSE_DATA value."""
    if override.strip():
        return override.strip()
    try:
        from dotenv import dotenv_values
        env = dotenv_values(_ROOT / ".env")
        return env.get("LSE_DATA", "")
    except Exception:
        return ""


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    api_key: str = "",
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from the LSE API.

    Returns a DataFrame with a UTC-aware DatetimeIndex and columns
    [open, high, low, close, volume]. Raises on missing credentials or empty data.
    """
    try:
        from lse import LSE
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: lse-data. Install with: pip install 'lse-data[frames]'"
        ) from exc

    key = get_api_key(api_key)
    if not key:
        raise ValueError(
            "LSE API key not set. Add LSE_DATA=your_key to .env or enter it in the sidebar."
        )

    client = LSE(api_key=key)
    rows = client.candles(symbol, timeframe, start=start, end=end)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} {timeframe} {start} → {end}.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    if "volume" not in df.columns:
        df["volume"] = 0

    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_bars_cached(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    api_key: str = "",
    cache_key_prefix: str = "lse",
) -> pd.DataFrame | None:
    """Session-state cached wrapper around fetch_ohlcv. Returns None on error."""
    cache_key = f"_bars_{cache_key_prefix}_{symbol}_{timeframe}_{start}_{end}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        df = fetch_ohlcv(symbol, timeframe, start, end, api_key)
        st.session_state[cache_key] = df
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None


def build_universe(symbol: str, df: pd.DataFrame):
    """Wrap an OHLCV DataFrame in a Universe for backtester consumption."""
    from core.universe import Universe
    uni = Universe(symbols=[symbol])
    uni.add_asset(symbol, df)
    return uni
