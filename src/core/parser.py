"""
core/parser.py — Parse multi-level L2, funding rate, and OHLCV data.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from core.models import FundingSnapshot, OrderBookSnapshot, OrderBookLevel


# ── Timeframe registry ────────────────────────────────────────────────────────

# (pandas_resample_freq, seconds_per_bar, approx_bars_per_year)
# bars_per_year = 365 * 24 * 3600 / seconds_per_bar
_TIMEFRAME_MAP: dict[str, tuple[str, int, int]] = {
    "1m":  ("1min",   60,      525_600),
    "2m":  ("2min",   120,     262_800),
    "3m":  ("3min",   180,     175_200),
    "5m":  ("5min",   300,     105_120),
    "10m": ("10min",  600,      52_560),
    "15m": ("15min",  900,      35_040),
    "30m": ("30min",  1_800,    17_520),
    "1h":  ("1h",     3_600,     8_760),
    "2h":  ("2h",     7_200,     4_380),
    "4h":  ("4h",    14_400,     2_190),
    "6h":  ("6h",    21_600,     1_460),
    "8h":  ("8h",    28_800,     1_095),
    "12h": ("12h",   43_200,       730),
    "1d":  ("1D",    86_400,       365),
}

TIMEFRAMES = list(_TIMEFRAME_MAP.keys())


def timeframe_to_pandas(tf: str) -> str:
    """Return the pandas resample frequency string for a timeframe label."""
    entry = _TIMEFRAME_MAP.get(tf)
    if entry is None:
        raise ValueError(f"Unknown timeframe '{tf}'. Valid: {TIMEFRAMES}")
    return entry[0]


def timeframe_to_seconds(tf: str) -> int:
    """Return seconds per bar for a timeframe label."""
    entry = _TIMEFRAME_MAP.get(tf)
    if entry is None:
        raise ValueError(f"Unknown timeframe '{tf}'. Valid: {TIMEFRAMES}")
    return entry[1]


def timeframe_to_bars_per_year(tf: str) -> int:
    """Return approximate number of bars per calendar year for a timeframe label."""
    entry = _TIMEFRAME_MAP.get(tf)
    if entry is None:
        raise ValueError(f"Unknown timeframe '{tf}'. Valid: {TIMEFRAMES}")
    return entry[2]


# ── OHLCV from raw trades ─────────────────────────────────────────────────────


def trades_to_ohlcv(
    input_folder: str | Path,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """
    Read raw trade ticks from parquet and resample into OHLCV bars.

    Parameters
    ----------
    input_folder : directory of parquet files (one or more files)
    timeframe    : bar size — one of TIMEFRAMES (e.g. "1m", "5m", "1h", "1d")

    Returns
    -------
    DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
    """
    pandas_freq = timeframe_to_pandas(timeframe)

    data = pd.read_parquet(input_folder, engine="pyarrow")
    data["datetime"] = pd.to_datetime(data["timestamp"], unit="ms")
    data.set_index("datetime", inplace=True)

    ohlcv = data["price"].resample(pandas_freq).ohlc().dropna()
    ohlcv["volume"] = data["size"].resample(pandas_freq).sum()
    return ohlcv


def trades_to_ohlc(input_folder: str | Path) -> pd.DataFrame:
    """Backward-compatible alias for trades_to_ohlcv with 1-minute bars."""
    return trades_to_ohlcv(input_folder, timeframe="1m")


# ── Funding rate parsing ──────────────────────────────────────────────────────


def funding_to_snapshots(
    input_folder: str | Path,
) -> list[FundingSnapshot]:
    """
    Parse funding rate parquet files into a list of FundingSnapshots.

    The parquet schema expected is:
        timestamp    : int64  (milliseconds since epoch)
        funding_rate : float64
        mark_price   : float64  (may be 0 / null when unavailable)
        oracle_price : float64  (may be 0 / null when unavailable)

    Returns snapshots sorted chronologically.
    """
    data = pd.read_parquet(input_folder, engine="pyarrow")
    data = _ensure_funding_datetime_index(data)
    data = data.sort_index()

    snapshots: list[FundingSnapshot] = []
    for ts, row in data.iterrows():
        rate = float(row.get("funding_rate", 0.0) or 0.0)
        mark = float(row.get("mark_price", 0.0) or 0.0)
        oracle = float(row.get("oracle_price", 0.0) or 0.0)
        # Annualise: Hyperliquid posts 8h funding → 3 × 365 = 1095 periods/year
        # Rate might already be per-period or annualised — we store both.
        # Convention: if |rate| < 0.1 treat it as per-period (e.g. 0.0001)
        # and annualise assuming 3 payments per day (8h standard).
        if abs(rate) < 0.1:
            rate_ann_bps = rate * 3 * 365 * 1e4
        else:
            rate_ann_bps = rate
            rate = rate / (3 * 365 * 1e4)
        snapshots.append(
            FundingSnapshot(
                timestamp=ts,
                rate=rate,
                rate_annualized=rate_ann_bps,
                oracle_price=oracle,
                mark_price=mark,
            )
        )
    return snapshots


def _ensure_funding_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    for col in ("timestamp", "ts", "time", "epoch"):
        if col in df.columns:
            vals = df[col]
            if pd.api.types.is_numeric_dtype(vals):
                sample = vals.dropna().iloc[0] if not vals.empty else 0
                unit = "ns" if sample > 1e15 else "ms" if sample > 1e12 else "s"
                df = df.copy()
                df.index = pd.to_datetime(vals, unit=unit, utc=True)
            else:
                df = df.copy()
                df.index = pd.to_datetime(vals, utc=True)
            df.index.name = "datetime"
            return df

    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def align_funding_to_ohlcv(
    snapshots: list[FundingSnapshot],
    ohlcv: pd.DataFrame,
    method: str = "last",
) -> list[FundingSnapshot]:
    """
    Pick one FundingSnapshot per OHLCV bar — mirrors align_l2_to_ohlcv.

    Methods:
      "last"    — most recent snapshot at or before the bar close (default)
      "nearest" — closest snapshot by absolute time
      "forward" — next snapshot at or after the bar open

    Returns exactly len(ohlcv) snapshots, one per bar.
    """
    if not snapshots:
        raise ValueError("No funding snapshots to align")

    snap_times = pd.DatetimeIndex(
        [s.timestamp for s in snapshots]
    ).tz_localize("UTC") if snapshots[0].timestamp.tzinfo is None else pd.DatetimeIndex(
        [s.timestamp for s in snapshots]
    )

    bar_times = ohlcv.index
    if bar_times.tz is None:
        bar_times = bar_times.tz_localize("UTC")

    aligned: list[FundingSnapshot] = []

    if method == "last":
        snap_idx = pd.Series(range(len(snapshots)), index=snap_times).sort_index()
        for bt in bar_times:
            mask = snap_idx.index <= bt
            if mask.any():
                aligned.append(snapshots[snap_idx[mask].iloc[-1]])
            else:
                aligned.append(snapshots[0])

    elif method == "nearest":
        snap_ns = snap_times.astype("int64").values
        for bt in bar_times:
            bt_ns = bt.value
            idx = int(np.argmin(np.abs(snap_ns - bt_ns)))
            aligned.append(snapshots[idx])

    elif method == "forward":
        snap_idx = pd.Series(range(len(snapshots)), index=snap_times).sort_index()
        for bt in bar_times:
            mask = snap_idx.index >= bt
            if mask.any():
                aligned.append(snapshots[snap_idx[mask].iloc[0]])
            else:
                aligned.append(snapshots[-1])

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'last', 'nearest', or 'forward'.")

    return aligned


# ── L2 order book parsing ─────────────────────────────────────────────────────


def l2_to_orderbook(
    input_folder: str | Path,
    ohlcv_data: pd.DataFrame | None = None,
    aligned: bool = True,
) -> list[OrderBookSnapshot]:
    data = pd.read_parquet(input_folder, engine="pyarrow")

    data["datetime"] = pd.to_datetime(data["timestamp"], unit="ms")
    data.set_index("datetime", inplace=True)
    data.drop("timestamp", axis=1, inplace=True)

    orderbook = parse_l2(data)
    if aligned and ohlcv_data is not None:
        orderbook = align_l2_to_ohlcv(orderbook, ohlcv_data)

    return orderbook


# ── Column detection ──────────────────────────────────────────────────────────


def _detect_levels(df: pd.DataFrame) -> int:
    """Auto-detect number of L2 levels from column names."""
    last_row_bid_px = len(df["bids_px"].iloc[-1])
    last_row_bid_sz = len(df["bids_sz"].iloc[-1])
    last_row_ask_px = len(df["asks_px"].iloc[-1])
    last_row_ask_sz = len(df["asks_sz"].iloc[-1])
    level = min(last_row_bid_px, last_row_bid_sz, last_row_ask_px, last_row_ask_sz)
    if not level or level < 1:
        raise KeyError(f"Cannot detect L2 levels. Got: {list(df.columns)}")
    return level


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a DatetimeIndex."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    ts_aliases = ["timestamp", "ts", "epoch", "time", "exchange_ts"]
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for alias in ts_aliases:
        if alias in cols_lower:
            col = cols_lower[alias]
            vals = df[col]
            if vals.dtype in ("int64", "float64"):
                sample = vals.iloc[0]
                if sample > 1e15:
                    unit = "ns"
                elif sample > 1e12:
                    unit = "ms"
                else:
                    unit = "s"
                df = df.copy()
                df.index = pd.to_datetime(vals, unit=unit)
            else:
                df = df.copy()
                df.index = pd.to_datetime(vals)
            df.index.name = "datetime"
            return df

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


# ── Main L2 parser ────────────────────────────────────────────────────────────


def parse_l2(
    df: pd.DataFrame,
    n_levels: int | None = None,
) -> list[OrderBookSnapshot]:
    """
    Parse a multi-level L2 DataFrame into OrderBookSnapshots.

    Parameters
    ----------
    df       : DataFrame with datetime index and bids_px/bids_sz/asks_px/asks_sz columns.
    n_levels : override auto-detection.

    Returns
    -------
    list[OrderBookSnapshot] — one per row, same length and order as df
    """
    df = _ensure_datetime_index(df)
    levels = n_levels or _detect_levels(df)

    timestamps = df.index
    bids_px_col = df["bids_px"]
    bids_sz_col = df["bids_sz"]
    asks_px_col = df["asks_px"]
    asks_sz_col = df["asks_sz"]

    snapshots: list[OrderBookSnapshot] = []

    for ts, b_pxs, b_szs, a_pxs, a_szs in zip(
        timestamps, bids_px_col, bids_sz_col, asks_px_col, asks_sz_col
    ):
        bids = []
        if isinstance(b_pxs, (list, np.ndarray, tuple)):
            for px, sz in zip(b_pxs[:levels], b_szs[:levels]):
                if (
                    px is not None
                    and sz is not None
                    and not np.isnan(px)
                    and not np.isnan(sz)
                    and sz > 0
                ):
                    bids.append(OrderBookLevel(price=float(px), size=float(sz)))

        asks = []
        if isinstance(a_pxs, (list, np.ndarray, tuple)):
            for px, sz in zip(a_pxs[:levels], a_szs[:levels]):
                if (
                    px is not None
                    and sz is not None
                    and not np.isnan(px)
                    and not np.isnan(sz)
                    and sz > 0
                ):
                    asks.append(OrderBookLevel(price=float(px), size=float(sz)))

        snapshots.append(
            OrderBookSnapshot(
                timestamp=ts,
                bids=bids,
                asks=asks,
            )
        )

    return snapshots


# ── Align L2 snapshots to OHLCV bars ─────────────────────────────────────────


def align_l2_to_ohlcv(
    snapshots: list[OrderBookSnapshot],
    ohlcv: pd.DataFrame,
    method: str = "last",
) -> list[OrderBookSnapshot]:
    """
    Pick one L2 snapshot per OHLCV bar.

    Methods:
      "last"    — last snapshot at or before the bar close (default, most realistic)
      "nearest" — closest snapshot by absolute time
      "vwap"    — merge all snapshots within the bar into combined depth

    Returns exactly len(ohlcv) snapshots aligned 1:1 with the bars.
    """
    if not snapshots:
        raise ValueError("No snapshots to align")

    snap_times = pd.DatetimeIndex([s.timestamp for s in snapshots])
    bar_times = ohlcv.index
    aligned: list[OrderBookSnapshot] = []

    if method == "last":
        snap_idx = pd.Series(range(len(snapshots)), index=snap_times).sort_index()
        for bt in bar_times:
            mask = snap_idx.index <= bt
            if mask.any():
                aligned.append(snapshots[snap_idx[mask].iloc[-1]])
            else:
                aligned.append(snapshots[0])

    elif method == "nearest":
        snap_ts_ns = snap_times.astype("int64").values
        for bt in bar_times:
            bt_ns = bt.value
            idx = int(np.argmin(np.abs(snap_ts_ns - bt_ns)))
            aligned.append(snapshots[idx])

    elif method == "vwap":
        freq = pd.infer_freq(bar_times) or "1h"
        snap_idx = pd.Series(range(len(snapshots)), index=snap_times).sort_index()

        for i in range(len(bar_times)):
            start = bar_times[i - 1] if i > 0 else bar_times[i] - pd.Timedelta(freq)
            end = bar_times[i]
            indices = snap_idx[
                (snap_idx.index > start) & (snap_idx.index <= end)
            ].values

            if len(indices) == 0:
                before = snap_idx.index <= end
                aligned.append(
                    snapshots[snap_idx[before].iloc[-1]]
                    if before.any()
                    else snapshots[0]
                )
                continue

            bid_agg: dict[float, float] = defaultdict(float)
            ask_agg: dict[float, float] = defaultdict(float)
            for si in indices:
                for lvl in snapshots[si].bids:
                    bid_agg[lvl.price] += lvl.size
                for lvl in snapshots[si].asks:
                    ask_agg[lvl.price] += lvl.size

            aligned.append(
                OrderBookSnapshot(
                    timestamp=end,
                    bids=[
                        OrderBookLevel(p, s)
                        for p, s in sorted(bid_agg.items(), key=lambda x: -x[0])
                    ],
                    asks=[
                        OrderBookLevel(p, s)
                        for p, s in sorted(ask_agg.items(), key=lambda x: x[0])
                    ],
                )
            )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'last', 'nearest', or 'vwap'."
        )

    return aligned
