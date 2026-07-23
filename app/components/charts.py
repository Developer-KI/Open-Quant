"""Reusable Plotly chart builders for the trading dashboard."""
from __future__ import annotations

import math
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

_DARK = "plotly_dark"
_GREEN = "#26a69a"
_RED = "#ef5350"
_BLUE = "#2196F3"
_ORANGE = "#FF9800"
_PURPLE = "#9C27B0"
_CYAN = "#00BCD4"

_OVERLAY_COLORS = [_BLUE, _ORANGE, _PURPLE, _CYAN, "#4CAF50", "#F44336"]

# Shared style constants matching the dashboard terminal aesthetic
_GRID = dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
_LEGEND_H = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
_MARGIN_MAIN = dict(l=50, r=20, t=50, b=20)
_MARGIN_MINI = dict(l=50, r=20, t=10, b=20)


def candlestick_chart(
    df: pd.DataFrame,
    overlays: dict[str, pd.Series] | None = None,
    title: str = "",
    height: int = 500,
) -> go.Figure:
    """
    OHLCV candlestick with optional indicator overlays.
    overlays: {label: pd.Series} added as line traces on the same y-axis.
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color=_GREEN,
        decreasing_line_color=_RED,
    ))
    for i, (name, series) in enumerate((overlays or {}).items()):
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=name,
            line=dict(color=_OVERLAY_COLORS[i % len(_OVERLAY_COLORS)], width=1.5),
        ))
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template=_DARK,
        height=height,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def bollinger_traces(mid: pd.Series, upper: pd.Series, lower: pd.Series) -> list[go.Scatter]:
    """Return three Scatter traces for Bollinger Bands (add to an existing figure)."""
    return [
        go.Scatter(x=upper.index, y=upper.values, name="BB Upper",
                   line=dict(color=_ORANGE, width=1, dash="dot"), showlegend=True),
        go.Scatter(x=lower.index, y=lower.values, name="BB Lower",
                   line=dict(color=_ORANGE, width=1, dash="dot"),
                   fill="tonexty", fillcolor="rgba(255,152,0,0.05)", showlegend=True),
        go.Scatter(x=mid.index, y=mid.values, name="BB Mid",
                   line=dict(color=_ORANGE, width=1), showlegend=True),
    ]


def volume_bars(df: pd.DataFrame) -> go.Figure:
    """Color-coded volume bar chart."""
    colors = [_GREEN if c >= o else _RED for c, o in zip(df["close"], df["open"])]
    fig = go.Figure(go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volume"))
    fig.update_layout(
        template=_DARK, height=150,
        hovermode="x unified",
        margin=_MARGIN_MINI,
        showlegend=False,
        yaxis_title="Volume",
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def rsi_chart(series: pd.Series, period: int = 14) -> go.Figure:
    """RSI line chart with overbought/oversold reference lines."""
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.08)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color=_RED, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=_GREEN, opacity=0.5)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        name=f"RSI({period})", line=dict(color=_ORANGE, width=1.5),
    ))
    fig.update_layout(
        template=_DARK, height=150,
        hovermode="x unified",
        yaxis=dict(range=[0, 100]),
        margin=_MARGIN_MINI,
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def atr_chart(series: pd.Series, period: int = 14) -> go.Figure:
    """ATR area chart."""
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values,
        name=f"ATR({period})", line=dict(color=_PURPLE, width=1.5),
        fill="tozeroy", fillcolor="rgba(156,39,176,0.12)",
    ))
    fig.update_layout(
        template=_DARK, height=150,
        hovermode="x unified",
        margin=_MARGIN_MINI,
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def equity_chart(equity_curve: pd.Series, drawdown: pd.Series) -> go.Figure:
    """Equity curve (top) + drawdown % (bottom), shared x-axis."""
    fig = sp.make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.02,
    )
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        name="Equity", line=dict(color=_BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=(drawdown.values * 100),
        name="Drawdown %", line=dict(color=_RED, width=1),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.25)",
    ), row=2, col=1)
    fig.update_layout(
        template=_DARK, height=460,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        legend=_LEGEND_H,
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def trade_markers(fig: go.Figure, trades_df: pd.DataFrame) -> go.Figure:
    """Overlay entry (triangle) and exit (x) markers on an existing price figure."""
    if trades_df is None or trades_df.empty:
        return fig

    def _to_str(v):
        s = str(v)
        return s.split(".")[-1] if "." in s else s

    side_col = trades_df["side"].map(_to_str) if "side" in trades_df.columns else pd.Series(["LONG"] * len(trades_df), index=trades_df.index)
    ts_col = "timestamp"
    ep_col = "entry_price"
    xp_col = "exit_price"

    for side, color, sym in [("LONG", _GREEN, "triangle-up"), ("SHORT", _RED, "triangle-down")]:
        sub = trades_df[side_col == side]
        if sub.empty:
            continue
        reason_entry = sub["reason_entry"] if "reason_entry" in sub.columns else pd.Series(["—"] * len(sub), index=sub.index)
        custom = pd.DataFrame({
            "entry_price": sub[ep_col],
            "side": side,
            "reason": reason_entry.fillna("—"),
        }).values
        fig.add_trace(go.Scatter(
            x=sub[ts_col], y=sub[ep_col],
            mode="markers", name=f"{side} Entry",
            legendgroup=f"{side} Entry",
            marker=dict(color=color, size=14, symbol=sym, line=dict(color="white", width=1.5)),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[1]} ENTRY</b><br>"
                "Price: %{y:.4f}<br>"
                "Reason: %{customdata[2]}<extra></extra>"
            ),
        ))

    if xp_col not in trades_df.columns:
        return fig
    exits = trades_df.dropna(subset=[xp_col])
    if exits.empty:
        return fig

    pnl_series = exits["pnl"] if "pnl" in exits.columns else pd.Series([0.0] * len(exits), index=exits.index)
    pnl_pct_series = exits["pnl_pct"] if "pnl_pct" in exits.columns else pd.Series([0.0] * len(exits), index=exits.index)
    exit_ts = exits["exit_timestamp"] if "exit_timestamp" in exits.columns else exits[ts_col]

    for label, color, sym, mask in [
        ("Exit WIN",  _GREEN, "star", pnl_series > 0),
        ("Exit LOSS", _RED,   "x",   pnl_series <= 0),
    ]:
        grp = exits[mask]
        if grp.empty:
            continue
        custom = pd.DataFrame({
            "exit_price": grp[xp_col],
            "pnl": pnl_series[mask].fillna(0),
            "pnl_pct": pnl_pct_series[mask].fillna(0),
        }).values
        fig.add_trace(go.Scatter(
            x=exit_ts[mask], y=grp[xp_col],
            mode="markers", name=label,
            legendgroup="Exits",
            marker=dict(color=color, size=11, symbol=sym, line=dict(color="white", width=1.2)),
            customdata=custom,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Price: %{y:.4f}<br>"
                "PnL: %{customdata[1]:.2f} (%{customdata[2]:.2f}%)<extra></extra>"
            ),
        ))
    return fig


def signal_log_chart(signal_log: pd.DataFrame, height: int = 320) -> go.Figure:
    """Two-panel signal log: direction bars (top) + confidence/weight lines (bottom)."""
    ts = pd.to_datetime(signal_log["timestamp"], unit="ms", errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(signal_log["timestamp"], errors="coerce")

    direction = signal_log["side"].map({"LONG": 1, "FLAT": 0, "SHORT": -1}).fillna(0)
    bar_colors = [
        _GREEN if d == 1 else _RED if d == -1 else "rgba(100,100,100,0.35)"
        for d in direction
    ]
    reason = signal_log["reason"] if "reason" in signal_log.columns else pd.Series([""] * len(signal_log))
    side_str = signal_log["side"] if "side" in signal_log.columns else pd.Series([""] * len(signal_log))

    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.60],
        vertical_spacing=0.04,
        subplot_titles=["Direction", "Confidence / Weight"],
    )

    # Row 1 — direction bars
    fig.add_trace(go.Bar(
        x=ts, y=direction,
        marker_color=bar_colors,
        name="Direction",
        showlegend=False,
        customdata=pd.concat([side_str.rename("side"), reason.rename("reason")], axis=1).values,
        hovertemplate="<b>%{customdata[0]}</b><br>Reason: %{customdata[1]}<extra></extra>",
    ), row=1, col=1)

    # Row 2 — confidence + weight
    if "confidence" in signal_log.columns:
        fig.add_trace(go.Scatter(
            x=ts, y=signal_log["confidence"],
            name="Confidence", line=dict(color=_BLUE, width=1.5),
            hovertemplate="Confidence: %{y:.3f}<extra></extra>",
        ), row=2, col=1)

    if "weight" in signal_log.columns:
        fig.add_trace(go.Scatter(
            x=ts, y=signal_log["weight"],
            name="Weight", line=dict(color=_ORANGE, width=1.5),
            hovertemplate="Weight: %{y:.3f}<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        template=_DARK,
        height=height,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        legend=_LEGEND_H,
        bargap=0,
    )
    fig.update_yaxes(
        tickvals=[-1, 0, 1],
        ticktext=["SHORT", "FLAT", "LONG"],
        range=[-1.5, 1.5],
        row=1, col=1,
        **_GRID,
    )
    fig.update_yaxes(range=[0, 1.05], row=2, col=1, **_GRID)
    fig.update_xaxes(**_GRID)
    return fig


def depth_chart(snapshot) -> go.Figure:
    """Mirrored bid/ask depth bars from an OrderBookSnapshot."""
    if snapshot is None:
        return go.Figure()

    bids = snapshot.bids[:20]
    asks = snapshot.asks[:20]

    fig = go.Figure()
    if bids:
        fig.add_trace(go.Bar(
            x=[lvl.price for lvl in bids],
            y=[lvl.size for lvl in bids],
            name="Bids", marker_color=_GREEN,
        ))
    if asks:
        fig.add_trace(go.Bar(
            x=[lvl.price for lvl in asks],
            y=[lvl.size for lvl in asks],
            name="Asks", marker_color=_RED,
        ))
    fig.update_layout(
        template=_DARK, height=350, barmode="overlay",
        hovermode="closest",
        margin=_MARGIN_MAIN,
        xaxis_title="Price", yaxis_title="Size",
        legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def funding_chart(df: pd.DataFrame) -> go.Figure:
    """Funding rate (bps) bar + mark/oracle price lines."""
    fig = sp.make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.6], vertical_spacing=0.02,
    )
    if "funding_rate" in df.columns:
        vals = df["funding_rate"] * 1e4
        colors = [_GREEN if v >= 0 else _RED for v in vals]
        fig.add_trace(go.Bar(
            x=df.index, y=vals, name="Funding (bps)", marker_color=colors,
        ), row=1, col=1)

    for col, color, label in [("mark_price", _BLUE, "Mark"), ("oracle_price", _ORANGE, "Oracle")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=label,
                line=dict(color=color, width=1.5),
            ), row=2, col=1)

    fig.update_layout(
        template=_DARK, height=420,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        legend=_LEGEND_H,
    )
    fig.update_yaxes(title_text="Funding (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def sentiment_scatter(df: pd.DataFrame) -> go.Figure:
    """Sentiment score scatter plot colored by source."""
    if df.empty:
        return go.Figure()

    ts_col = next((c for c in ["created_at", "timestamp", "date"] if c in df.columns), None)
    score_col = next((c for c in ["score", "sentiment_score", "compound"] if c in df.columns), None)
    if not ts_col or not score_col:
        return go.Figure()

    src_colors = {
        "x": "#1DA1F2", "reddit": "#FF4500",
        "telegram": "#0088CC", "chan": "#7A6A4F",
    }
    source_col = "source" if "source" in df.columns else None
    sources = df[source_col].unique() if source_col else ["all"]

    fig = go.Figure()
    for src in sources:
        sub = df[df[source_col] == src] if source_col else df
        fig.add_trace(go.Scatter(
            x=sub[ts_col], y=sub[score_col],
            mode="markers", name=str(src),
            marker=dict(color=src_colors.get(str(src), _PURPLE), size=5, opacity=0.7),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template=_DARK, height=300,
        hovermode="closest",
        margin=_MARGIN_MAIN,
        xaxis_title="Time", yaxis_title="Sentiment Score",
        legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def spread_chart(
    timestamps: list,
    spread_bps: list,
    title: str = "",
    height: int = 150,
) -> go.Figure:
    """Bid-ask spread (bps) time series."""
    fig = go.Figure(go.Scatter(
        x=timestamps, y=spread_bps,
        name="Spread (bps)", line=dict(color=_CYAN, width=1),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.10)",
    ))
    fig.update_layout(
        template=_DARK, height=height,
        hovermode="x unified",
        title=title,
        margin=_MARGIN_MAIN if title else _MARGIN_MINI,
        yaxis_title="Spread (bps)",
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def funding_rate_mini(df: pd.DataFrame) -> go.Figure:
    """Compact funding rate (bps) bar chart — used as a subplot below the price chart."""
    if "funding_rate" not in df.columns:
        return go.Figure()
    vals = df["funding_rate"] * 1e4
    colors = [_GREEN if v >= 0 else _RED for v in vals]
    fig = go.Figure(go.Bar(
        x=df.index, y=vals, name="Funding (bps)", marker_color=colors,
    ))
    fig.add_hline(y=0, line_color="gray", line_width=0.5, opacity=0.4)
    fig.update_layout(
        template=_DARK, height=150,
        hovermode="x unified",
        margin=_MARGIN_MINI,
        yaxis_title="Funding (bps)",
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def macd_chart(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> go.Figure:
    """MACD line + signal line (top) and histogram (bottom), shared x-axis."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    fig = sp.make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.02,
    )
    fig.add_trace(go.Scatter(
        x=macd_line.index, y=macd_line.values,
        name=f"MACD({fast},{slow})", line=dict(color=_BLUE, width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=signal_line.index, y=signal_line.values,
        name=f"Signal({signal})", line=dict(color=_ORANGE, width=1.5),
    ), row=1, col=1)
    bar_colors = [_GREEN if v >= 0 else _RED for v in histogram]
    fig.add_trace(go.Bar(
        x=histogram.index, y=histogram.values,
        name="Histogram", marker_color=bar_colors, showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=0.5, opacity=0.4, row=2, col=1)
    fig.update_layout(
        template=_DARK, height=250,
        hovermode="x unified",
        margin=_MARGIN_MINI,
        legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def macro_chart(df: pd.DataFrame, col: str, label: str, color: str = _BLUE) -> go.Figure:
    """Generic macro time-series area chart."""
    if df.empty or col not in df.columns:
        return go.Figure()
    ts_col = next((c for c in ["timestamp", "date", "datetime"] if c in df.columns), df.index.name or None)
    x = df[ts_col] if ts_col and ts_col in df.columns else df.index
    fig = go.Figure(go.Scatter(
        x=x, y=df[col], name=label,
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.10)",
    ))
    fig.update_layout(
        template=_DARK, height=260, title=label,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ---------------------------------------------------------------------------
# Orderflow & feature charts (research / analysis)
# ---------------------------------------------------------------------------

def plot_orderflow(
    tick_df: pd.DataFrame,
    l2_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    start_time=None,
    end_time=None,
    title: str = "Orderflow & Tick Dynamics",
    height: int = 900,
) -> go.Figure:
    """4-panel orderflow chart: price/BBO/trades, volume, spread %, L2 imbalance."""
    if start_time and end_time:
        tick_sub = tick_df.loc[start_time:end_time]
        l2_sub = l2_df.loc[start_time:end_time]
        ohlcv_sub = ohlcv_df.loc[start_time:end_time]
    else:
        tick_sub = tick_df.copy()
        l2_sub = l2_df.copy()
        ohlcv_sub = ohlcv_df.copy()

    fig = sp.make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2, 0.2],
        subplot_titles=(
            "Price, BBO & Trades",
            "Traded Volume",
            "Bid-Ask Spread",
            "L2 Depth Imbalance (Bid vs Ask Size)",
        ),
    )

    fig.add_trace(go.Candlestick(
        x=ohlcv_sub.index,
        open=ohlcv_sub["open"], high=ohlcv_sub["high"],
        low=ohlcv_sub["low"], close=ohlcv_sub["close"],
        name="1m OHLC",
        increasing_line_color=_GREEN,
        decreasing_line_color=_RED,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=l2_sub["bid_px"],
        mode="lines", line=dict(color="rgba(38,166,154,0.6)", width=1, shape="hv"),
        name="Best Bid",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=l2_sub["ask_px"],
        mode="lines", line=dict(color="rgba(239,83,80,0.6)", width=1, shape="hv"),
        name="Best Ask",
    ), row=1, col=1)

    buys = tick_sub[tick_sub["side"].str.lower() == "buy"]
    sells = tick_sub[tick_sub["side"].str.lower() == "sell"]

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["price"], mode="markers",
        marker=dict(symbol="triangle-up", color="#00ff00", size=6,
                    line=dict(width=0.5, color="black")),
        name="Buy Trade", text=buys["size"],
        hovertemplate="Time: %{x}<br>Price: %{y}<br>Size: %{text}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["price"], mode="markers",
        marker=dict(symbol="triangle-down", color=_RED, size=6,
                    line=dict(width=0.5, color="black")),
        name="Sell Trade", text=sells["size"],
        hovertemplate="Time: %{x}<br>Price: %{y}<br>Size: %{text}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ohlcv_sub.index, y=ohlcv_sub["volume"],
        mode="lines", fill="tozeroy",
        line=dict(color="#82b1ff", width=1, shape="hv"),
        name="1m Volume",
    ), row=2, col=1)

    mid_price = (l2_sub["ask_px"] + l2_sub["bid_px"]) / 2
    pct_spread = ((l2_sub["ask_px"] - l2_sub["bid_px"]) / mid_price) * 100
    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=pct_spread,
        mode="lines", line=dict(color=_ORANGE, width=1.5, shape="hv"),
        fill="tozeroy", name="Spread %",
        hovertemplate="Time: %{x}<br>Spread: %{y:.4f}%<extra></extra>",
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=l2_sub["bid_sz"],
        mode="lines", line=dict(color=_GREEN, width=1, shape="hv"),
        fill="tozeroy", name="Bid Size (Support)",
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=-l2_sub["ask_sz"],
        mode="lines", line=dict(color=_RED, width=1, shape="hv"),
        fill="tozeroy", name="Ask Size (Resistance)",
    ), row=4, col=1)

    fig.update_layout(
        template=_DARK, title_text=title,
        xaxis_rangeslider_visible=False,
        hovermode="x unified", height=height,
        margin=_MARGIN_MAIN, legend=_LEGEND_H,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Spread %", ticksuffix="%", tickformat=".4f", row=3, col=1)
    fig.update_yaxes(title_text="Size (+Bid / -Ask)", row=4, col=1)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


Transformable = (
    str
    | dict[str, pd.Series | pd.DataFrame]
    | Callable[[pd.DataFrame], pd.Series]
)


def plot_features(
    df: pd.DataFrame,
    columns: Transformable | list[Transformable],
    title: str = "Feature Analysis",
    mode: str = "lines",
    separate_subplots: bool = False,
    height: int = 600,
    line_shape: str = "linear",
) -> go.Figure:
    """Generic multi-series chart for feature analysis."""
    if not isinstance(columns, list):
        columns = [columns]

    plot_data = []
    for item in columns:
        if isinstance(item, str):
            plot_data.append((item, df[item]))
        elif isinstance(item, dict):
            for label, series in item.items():
                plot_data.append((label, series))
        elif callable(item):
            series = item(df)
            name = getattr(series, "name", f"Transform_{len(plot_data)}")
            plot_data.append((name, series))

    num_plots = len(plot_data)
    if separate_subplots:
        fig = sp.make_subplots(
            rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        )
        for i, (name, series) in enumerate(plot_data, start=1):
            fig.add_trace(go.Scattergl(
                x=series.index, y=series, mode=mode, name=name, line_shape=line_shape,
            ), row=i, col=1)
            fig.update_yaxes(title_text=name, row=i, col=1)
    else:
        fig = go.Figure()
        for name, series in plot_data:
            fig.add_trace(go.Scattergl(
                x=series.index, y=series, mode=mode, name=name, line_shape=line_shape,
            ))

    fig.update_layout(
        template=_DARK, title_text=title, height=height,
        hovermode="x unified", margin=_MARGIN_MAIN, legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ── Options / IV surface charts ───────────────────────────────────────────────

def iv_smile_chart(surface, expiries=None, height: int = 420) -> go.Figure:
    """Implied-vol smile: market IV points (markers) with the fitted SVI curve (line),
    one colour per expiry, against log-moneyness."""
    expiries = expiries if expiries is not None else surface.expiries
    fig = go.Figure()
    for i, exp in enumerate(expiries):
        x, iv = surface.smile(exp)
        if len(x) == 0:
            continue
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        label = pd.Timestamp(exp).strftime("%Y-%m-%d")
        # Raw market quotes as markers.
        fig.add_trace(go.Scatter(
            x=x, y=iv * 100.0, name=label, mode="markers",
            marker=dict(color=color, size=6),
        ))
        # Smooth SVI fit as the connecting line (omitted when the slice was too thin to fit).
        ck, civ = surface.smile_curve(exp)
        if len(ck):
            fig.add_trace(go.Scatter(
                x=ck, y=civ * 100.0, name=f"{label} (SVI)", mode="lines",
                line=dict(color=color, width=1.5), showlegend=False,
                hoverinfo="skip",
            ))
    fig.update_layout(
        template=_DARK, title="IV Smile", height=height,
        hovermode="x unified", margin=_MARGIN_MAIN, legend=_LEGEND_H,
        xaxis_title=surface.x_label, yaxis_title="Implied Vol (%)",
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def term_structure_chart(surface, height: int = 300) -> go.Figure:
    """ATM implied vol vs time-to-expiry (years)."""
    Ts, ivs = surface.term_structure()
    fig = go.Figure(go.Scatter(
        x=Ts, y=ivs * 100.0, mode="lines+markers", name="ATM IV",
        line=dict(color=_BLUE, width=1.5),
    ))
    fig.update_layout(
        # _MARGIN_MAIN, not _MARGIN_MINI: the mini margin leaves 10px of headroom,
        # which clips this chart's title.
        template=_DARK, title="ATM Term Structure", height=height,
        hovermode="x unified", margin=_MARGIN_MAIN, showlegend=False,
        xaxis_title="Time to Expiry (years)", yaxis_title="Implied Vol (%)",
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def iv_surface_chart(surface, n: int = 40, height: int = 560) -> go.Figure:
    """
    3-D IV surface over the (log-moneyness, time) grid.

    Aspect and camera are set explicitly so the cube fills the frame — the plotly
    default leaves a band of empty space above it and clips the axis labels below.
    """
    X, Y, Z = surface.grid(n=n)
    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z * 100.0, colorscale="Viridis",
        colorbar=dict(title=dict(text="IV %", font=dict(size=11)),
                      thickness=10, len=0.7, x=0.96),
        hovertemplate=(f"{surface.x_label} %{{x:.3f}}<br>T %{{y:.3f}}y"
                       "<br>IV %{z:.1f}%<extra></extra>"),
    ))
    fig.update_layout(
        template=_DARK, height=height, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Implied Volatility Surface", y=0.97),
        scene=dict(
            xaxis_title=surface.x_label,
            yaxis_title="T (years)",
            zaxis_title="IV (%)",
            aspectmode="manual", aspectratio=dict(x=1.3, y=1.3, z=0.85),
            camera=dict(eye=dict(x=1.35, y=1.35, z=1.05),
                        center=dict(x=0, y=0, z=-0.18)),
        ),
    )
    return fig


def greeks_chart(greeks_df: pd.DataFrame, expiry=None, height: int = 420) -> go.Figure:
    """Delta / gamma / vega across strikes for one expiry (stacked subplots)."""
    df = greeks_df.copy()
    if expiry is not None:
        df = df[df["expiry"] == pd.Timestamp(expiry)]
    df = df.sort_values("strike")

    fig = sp.make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Delta", "Gamma", "Vega"),
    )
    for otype, color in (("call", _GREEN), ("put", _RED)):
        sub = df[df["option_type"] == otype]
        if sub.empty:
            continue
        common = dict(x=sub["strike"], mode="lines+markers",
                      line=dict(color=color, width=1.5), legendgroup=otype)
        fig.add_trace(go.Scatter(y=sub["delta"], name=f"{otype} Δ", **common), row=1, col=1)
        fig.add_trace(go.Scatter(y=sub["gamma"], name=f"{otype} Γ", showlegend=False,
                                 **{k: v for k, v in common.items() if k != "name"}), row=2, col=1)
        fig.add_trace(go.Scatter(y=sub["vega"], name=f"{otype} ν", showlegend=False,
                                 **{k: v for k, v in common.items() if k != "name"}), row=3, col=1)
    fig.update_layout(
        template=_DARK, height=height, hovermode="x unified",
        margin=_MARGIN_MAIN, legend=_LEGEND_H,
    )
    fig.update_xaxes(title_text="Strike", row=3, col=1, **_GRID)
    for r in (1, 2, 3):
        fig.update_yaxes(row=r, col=1, **_GRID)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Validation charts — Monte Carlo, regime stress, hypothesis tests
#
# Palette discipline for this block: blue is the single data hue (sequential when
# it encodes uncertainty), green/red are *status* colours reserved for sign and
# verdict and always ship alongside a label, and reference marks (observed value,
# null band) wear neutral ink so they never read as a third series. The trio
# #2196F3 / #26a69a / #ef5350 clears the CVD, normal-vision and contrast checks
# on this dark surface across all pairs.
# ══════════════════════════════════════════════════════════════════════════════

_INK = "#e6e9f0"        # primary reference ink (observed values)
_INK_MUTED = "#8b93a7"  # secondary ink (null bands, percentile rules)

# Metrics whose sign is fixed by construction — status colour would be noise.
_SIGNLESS_METRICS = {"max_drawdown_pct", "win_rate_pct"}


def _signed(value: float) -> str:
    """Status colour for a signed quantity — always paired with a visible label."""
    return _GREEN if value >= 0 else _RED


def mc_fan_chart(bands: dict, observed=None, initial: float | None = None,
                 height: int = 380) -> go.Figure:
    """
    Percentile fan of simulated equity paths, with the observed path on top.

    `bands` maps p5/p25/median/p75/p95 to equity arrays. The nested ribbons are one
    hue at increasing opacity: a sequential encoding of likelihood, not four series.
    """
    fig = go.Figure()
    x = list(range(len(bands["median"])))

    for lo, hi, alpha, label in (("p5", "p95", 0.10, "5-95%"),
                                 ("p25", "p75", 0.22, "25-75%")):
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(bands[hi]) + list(bands[lo])[::-1],
            fill="toself", fillcolor=f"rgba(33,150,243,{alpha})",
            line=dict(width=0), name=label, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=bands["median"], name="Median path",
        line=dict(color=_BLUE, width=2),
        hovertemplate="Trade %{x}<br>Median $%{y:,.0f}<extra></extra>",
    ))
    if observed is not None:
        fig.add_trace(go.Scatter(
            x=list(range(len(observed))), y=list(observed), name="Observed",
            line=dict(color=_INK, width=2, dash="dot"),
            hovertemplate="Trade %{x}<br>Observed $%{y:,.0f}<extra></extra>",
        ))
    if initial is not None:
        fig.add_hline(y=initial, line_dash="dash", line_color=_INK_MUTED, line_width=1,
                      annotation_text="Start", annotation_position="bottom right",
                      annotation_font_color=_INK_MUTED)

    fig.update_layout(
        template=_DARK, height=height, hovermode="x unified",
        margin=_MARGIN_MAIN, legend=_LEGEND_H,
        title="Simulated equity paths",
    )
    fig.update_xaxes(title_text="Trade #", **_GRID)
    fig.update_yaxes(title_text="Equity ($)", **_GRID)
    return fig


def mc_distribution_chart(values, observed: float | None = None, title: str = "",
                          unit: str = "%", height: int = 300) -> go.Figure:
    """
    Outcome histogram with the 5th/median/95th rules and the observed run marked.

    One series, so no legend: the title names it and the reference lines are
    directly labelled.
    """
    vals = pd.Series(values).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=60, marker_color=_BLUE, opacity=0.75,
        marker_line_width=0, name="Simulations",
        hovertemplate="%{x:.2f}" + unit + "<br>%{y} sims<extra></extra>",
    ))

    p5, med, p95 = vals.quantile(0.05), vals.median(), vals.quantile(0.95)
    # Opaque label backgrounds: the observed run often lands right on the median,
    # and overlapping annotations must stay legible when it does.
    _bg = "rgba(17,17,17,0.82)"
    for val, dash, color, label, pos in (
        (p5,  "dot",  _INK_MUTED, f"5th {p5:.1f}{unit}", "top left"),
        (med, "dash", _BLUE,      f"median {med:.1f}{unit}", "top left"),
        (p95, "dot",  _INK_MUTED, f"95th {p95:.1f}{unit}", "top right"),
    ):
        fig.add_vline(x=val, line_dash=dash, line_color=color, line_width=1,
                      annotation_text=label, annotation_position=pos,
                      annotation_font_size=10, annotation_font_color=color,
                      annotation_bgcolor=_bg)
    if observed is not None and pd.notna(observed):
        fig.add_vline(x=observed, line_color=_INK, line_width=2,
                      annotation_text=f"observed {observed:.1f}{unit}",
                      annotation_position="bottom right",
                      annotation_font_size=10, annotation_font_color=_INK,
                      annotation_bgcolor=_bg)

    fig.update_layout(
        template=_DARK, height=height, title=title, showlegend=False,
        margin=dict(l=50, r=20, t=50, b=30), bargap=0.02,
    )
    fig.update_xaxes(title_text=f"{title} ({unit})" if unit else title, **_GRID)
    fig.update_yaxes(title_text="Simulations", **_GRID)
    return fig


def regime_bar_chart(summary: pd.DataFrame, metric: str, label: str,
                     height: int = 300) -> go.Figure:
    """
    One metric across regimes — horizontal bars, coloured by sign and value-labelled.

    Regime identity lives on the axis, so colour is free to encode polarity.
    """
    df = summary.dropna(subset=[metric]).copy()
    df = df.sort_values(metric)
    vals = df[metric].astype(float)

    fig = go.Figure(go.Bar(
        x=vals, y=df["regime"].astype(str), orientation="h",
        marker_color=[_signed(v) for v in vals],
        text=[f"{v:,.2f}" for v in vals], textposition="outside",
        textfont=dict(color=_INK, size=11), cliponaxis=False,
        hovertemplate="%{y}<br>" + label + " %{x:,.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=_INK_MUTED, line_width=1)
    fig.update_layout(
        template=_DARK, height=height, title=label, showlegend=False,
        margin=dict(l=90, r=40, t=50, b=30),
    )
    # Pad the range so outside value labels never run into the regime names.
    lo, hi = float(min(vals.min(), 0)), float(max(vals.max(), 0))
    pad = (hi - lo) * 0.18 or 0.5
    fig.update_xaxes(range=[lo - pad, hi + pad], **_GRID)  # title lives in the chart title
    fig.update_yaxes(**_GRID)
    return fig


def bootstrap_ci_chart(ci: dict, height_per_row: int = 78) -> go.Figure:
    """
    Bootstrap confidence intervals — one small multiple per metric.

    Percentages and ratios never share an axis: each metric gets its own row and
    its own x-scale, with the observed value marked inside its interval.
    """
    metrics = list(ci.keys())
    # The metric name is the row's y-axis title, not a subplot title: a title sits
    # directly on top of the previous row's tick labels and collides with them.
    fig = sp.make_subplots(rows=len(metrics), cols=1, shared_xaxes=False,
                           vertical_spacing=0.16)

    for i, m in enumerate(metrics, start=1):
        v = ci[m]
        lo, hi, obs = float(v["lower"]), float(v["upper"]), float(v["observed"])

        # Pad so the endpoint labels stay inside the plot, and only draw the zero
        # rule when zero is actually near the interval — forcing it into range would
        # squash a win-rate interval of 36–51% into the right-hand tenth of the axis.
        span = (hi - lo) or abs(hi) or 1.0
        xlo, xhi = lo - span * 0.35, hi + span * 0.35
        zero_in_view = xlo <= 0 <= xhi

        # Status colour only where the sign is a real finding. Drawdown is negative
        # by construction, so colouring it red says nothing; likewise a win rate is
        # positive whatever the strategy does.
        if zero_in_view and m not in _SIGNLESS_METRICS:
            color = _GREEN if lo > 0 else (_RED if hi < 0 else _INK_MUTED)
        else:
            color = _INK_MUTED
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[0, 0], mode="lines",
            line=dict(color=color, width=8), showlegend=False,
            hovertemplate=m + "<br>CI %{x:,.3f}<extra></extra>",
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=[obs], y=[0], mode="markers+text",
            marker=dict(color=_INK, size=11, line=dict(color="#111111", width=2)),
            text=[f"{obs:,.2f}"], textposition="top center",
            textfont=dict(color=_INK, size=11), showlegend=False,
            hovertemplate=m + "<br>observed %{x:,.3f}<extra></extra>",
        ), row=i, col=1)
        for edge, anchor in ((lo, "right"), (hi, "left")):
            fig.add_annotation(x=edge, y=0, text=f"{edge:,.2f}", showarrow=False,
                               xanchor=anchor, xshift=-8 if anchor == "right" else 8,
                               font=dict(color=_INK_MUTED, size=10), row=i, col=1)

        if zero_in_view:
            fig.add_vline(x=0, line_color=_INK_MUTED, line_width=1, line_dash="dot",
                          row=i, col=1)
        # Row label as a horizontal annotation: a y-axis title would be rotated,
        # and four rotated titles collide down the left edge.
        fig.add_annotation(text=m.replace("_", " "), showarrow=False,
                           x=xlo, y=1.1, xanchor="left", yanchor="middle",
                           font=dict(size=11, color=_INK_MUTED), row=i, col=1)
        fig.update_yaxes(visible=False, zeroline=False, range=[-1, 1.6], row=i, col=1)
        fig.update_xaxes(range=[xlo, xhi], row=i, col=1, **_GRID)

    fig.update_layout(
        template=_DARK, height=height_per_row * len(metrics) + 40,
        margin=dict(l=30, r=30, t=20, b=20), showlegend=False,
    )
    return fig


def permutation_null_chart(pt, height: int = 190) -> go.Figure:
    """
    Where the observed statistic sits against the shuffled-order null.

    Same metric on both marks, so one axis is correct: a neutral 5-95% null band
    with the observed value marked in reference ink.
    """
    meta = pt.meta or {}
    p5 = float(meta.get("null_p5", 0.0))
    p95 = float(meta.get("null_p95", 0.0))
    null_mean = float(meta.get("null_mean", 0.0))
    obs = float(pt.statistic)

    # Null bands can be very narrow (a shuffled Sharpe often spans <0.01), where a
    # fixed 2-dp label prints the same number at both ends. Scale precision to span.
    span = max(abs(p95 - p5), abs(obs - null_mean), 1e-12)
    dec = min(6, max(2, int(math.ceil(-math.log10(span))) + 2))
    fmt = f"{{:.{dec}f}}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p5, p95], y=[0, 0], mode="lines",
        line=dict(color=_INK_MUTED, width=18), name="Null 5-95%",
        hovertemplate="null %{x}<extra></extra>",
    ))
    # A tick across the band, not a third series — light enough to read against the
    # muted band and to show up in the legend swatch.
    fig.add_trace(go.Scatter(
        x=[null_mean], y=[0], mode="markers",
        marker=dict(color=_INK, size=16, symbol="line-ns-open",
                    line=dict(color=_INK, width=2)),
        name="Null mean", hovertemplate="null mean %{x}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[obs], y=[0], mode="markers+text",
        marker=dict(color=_GREEN if pt.reject_null else _INK, size=14,
                    line=dict(color="#111111", width=2)),
        text=["observed " + fmt.format(obs)], textposition="top center",
        textfont=dict(color=_INK, size=11), name="Observed",
        hovertemplate="observed %{x}<extra></extra>",
    ))
    for edge, anchor in ((p5, "right"), (p95, "left")):
        fig.add_annotation(x=edge, y=0, text=fmt.format(edge), showarrow=False,
                           xanchor=anchor, xshift=-10 if anchor == "right" else 10,
                           font=dict(color=_INK_MUTED, size=10))

    lo, hi = min(p5, obs), max(p95, obs)
    pad = (hi - lo) * 0.35 or abs(hi) * 0.1 or 1.0
    fig.update_layout(
        template=_DARK, height=height, margin=dict(l=30, r=30, t=44, b=26),
        legend=_LEGEND_H, title="Observed vs shuffled-order null",
    )
    fig.update_xaxes(range=[lo - pad, hi + pad], **_GRID)
    fig.update_yaxes(visible=False, range=[-1, 1.6])
    return fig


# ── Sequential / diverging ramps for the sweep surface ───────────────────────
# Sequential: one hue, dark→bright, monotonic in OKLCH lightness (checked).
# Diverging: two poles + a NEUTRAL GRAY midpoint — never a rainbow, and never a
# hue at the middle, which is why plotly's RdYlGn (yellow midpoint) is not used.
_SEQ_BLUE = ["#0d2438", "#144a7c", "#1a6fbb", "#2196F3", "#7cc3f7"]
_DIV_RED_GREEN = [(0.0, "#ef5350"), (0.5, "#5b6273"), (1.0, "#26a69a")]


def returns_hist_chart(returns: pd.Series, height: int = 300) -> go.Figure:
    """Bar-return distribution with a zero rule. One series, so no legend."""
    vals = (returns.dropna() * 100)
    fig = go.Figure(go.Histogram(
        x=vals, nbinsx=80, marker_color=_BLUE, opacity=0.75, marker_line_width=0,
        hovertemplate="%{x:.2f}%<br>%{y} bars<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=_INK_MUTED, line_dash="dash", line_width=1)
    fig.update_layout(
        template=_DARK, height=height, title="Return distribution",
        showlegend=False, margin=dict(l=50, r=20, t=50, b=30), bargap=0.02,
    )
    fig.update_xaxes(title_text="Return (%)", **_GRID)
    fig.update_yaxes(title_text="Bars", **_GRID)
    return fig


def vol_regime_chart(rv: pd.Series, q_lo: pd.Series, q_hi: pd.Series,
                     window: int, height: int = 260) -> go.Figure:
    """
    Rolling annualised vol against its own expanding 33rd/66th percentiles.

    The percentile lines are thresholds, not series — they wear muted ink so the
    one data series keeps the only saturated colour.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=q_hi.index, y=q_hi, name="66th pctl",
        line=dict(color=_INK_MUTED, dash="dot", width=1),
        hovertemplate="66th %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=q_lo.index, y=q_lo, name="33rd pctl",
        line=dict(color=_INK_MUTED, dash="dot", width=1),
        hovertemplate="33rd %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rv.index, y=rv, name=f"Ann. vol ({window})",
        line=dict(color=_BLUE, width=1.8),
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        template=_DARK, height=height, hovermode="x unified",
        title=f"Volatility regime — {window}-bar rolling annualised vol",
        margin=_MARGIN_MAIN, legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(title_text="Ann. vol (%)", **_GRID)
    return fig


def sweep_heatmap(pivot: pd.DataFrame, metric: str, x_name: str, y_name: str,
                  height: int = 420) -> go.Figure:
    """
    Two-parameter sweep surface.

    Magnitude that crosses zero is polarity — a diverging scale anchored so the
    neutral midpoint sits exactly at zero. Otherwise it is plain magnitude and
    gets the single-hue sequential ramp.
    """
    z = [[float(v) if pd.notna(v) else None for v in row] for row in pivot.values]
    flat = [v for row in z for v in row if v is not None]
    lo, hi = (min(flat), max(flat)) if flat else (0.0, 1.0)

    if lo < 0 < hi:
        bound = max(abs(lo), abs(hi))  # symmetric, so the gray midpoint is zero
        scale, zmin, zmax = _DIV_RED_GREEN, -bound, bound
    else:
        scale, zmin, zmax = _SEQ_BLUE, lo, hi

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in pivot.columns], y=[str(r) for r in pivot.index],
        colorscale=scale, zmin=zmin, zmax=zmax,
        xgap=2, ygap=2,  # surface gap between cells
        text=[[f"{v:.3f}" if v is not None else "—" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title=dict(text=metric, font=dict(size=11)), thickness=12),
        hovertemplate=f"{x_name} %{{x}}<br>{y_name} %{{y}}<br>{metric} %{{z:.4f}}<extra></extra>",
    ))
    fig.update_layout(
        template=_DARK, height=height, title=f"{metric} — {x_name} vs {y_name}",
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text=x_name, type="category", **_GRID)
    fig.update_yaxes(title_text=y_name, type="category", **_GRID)
    return fig


def sweep_bar_chart(x_vals, y_vals, metric: str, x_name: str,
                    height: int = 320) -> go.Figure:
    """One-parameter sweep — bars coloured by sign, best value direct-labelled."""
    ys = [float(v) for v in y_vals]
    best = max(range(len(ys)), key=lambda i: ys[i]) if ys else None

    fig = go.Figure(go.Bar(
        x=[str(v) for v in x_vals], y=ys,
        marker_color=[_signed(v) for v in ys],
        text=[f"{v:,.3f}" if i == best else "" for i, v in enumerate(ys)],
        textposition="outside", textfont=dict(color=_INK, size=11), cliponaxis=False,
        hovertemplate=f"{x_name} %{{x}}<br>{metric} %{{y:,.4f}}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=_INK_MUTED, line_width=1)
    fig.update_layout(
        template=_DARK, height=height, title=f"{metric} vs {x_name}",
        showlegend=False, margin=dict(l=50, r=20, t=50, b=40), bargap=0.25,
    )
    fig.update_xaxes(title_text=x_name, type="category", **_GRID)
    fig.update_yaxes(title_text=metric, **_GRID)
    return fig
