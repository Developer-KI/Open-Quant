import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable  # Callable is still needed from typing
import pandas as pd

# TODO experiment with go.<plottype>gl variants vs SVG variants


def plot_orderflow(
    tick_df,
    l2_df,
    ohlcv_df,
    start_time=None,
    end_time=None,
    title="Orderflow & Tick Dynamics",
    height=900,
) -> go.Figure:
    """
    Plots a 3-panel interactive orderflow chart.

    Parameters:
    - tick_df, l2_df, ohlcv_df: Pandas DataFrames with DatetimeIndex
    - start_time, end_time: Strings or datetime objects to slice the data (e.g., '2024-01-01 09:30', '2024-01-01 10:00')
    - title: Title of the plot
    - height: Height of the figure in pixels

    Returns:
    - Plotly Figure object
    """

    # 1. Slice data if timeframes are provided
    if start_time and end_time:
        tick_sub = tick_df.loc[start_time:end_time]
        l2_sub = l2_df.loc[start_time:end_time]
        ohlcv_sub = ohlcv_df.loc[start_time:end_time]
    else:
        tick_sub = tick_df.copy()
        l2_sub = l2_df.copy()
        ohlcv_sub = ohlcv_df.copy()

    # 2. Initialize Subplots
    fig = make_subplots(
        rows=4,
        cols=1,
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

    # --- ROW 1: PRICE, L2 BBO & TRADES ---
    fig.add_trace(
        go.Candlestick(
            x=ohlcv_sub.index,
            open=ohlcv_sub["open"],
            high=ohlcv_sub["high"],
            low=ohlcv_sub["low"],
            close=ohlcv_sub["close"],
            name="1m OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=l2_sub.index,
            y=l2_sub["bid_px"],
            mode="lines",
            line=dict(color="rgba(38, 166, 154, 0.6)", width=1, shape="hv"),
            name="Best Bid",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=l2_sub.index,
            y=l2_sub["ask_px"],
            mode="lines",
            line=dict(color="rgba(239, 83, 80, 0.6)", width=1, shape="hv"),
            name="Best Ask",
        ),
        row=1,
        col=1,
    )

    # Trades
    buys = tick_sub[tick_sub["side"].str.lower() == "buy"]
    sells = tick_sub[tick_sub["side"].str.lower() == "sell"]

    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["price"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                color="#00ff00",
                size=6,
                line=dict(width=0.5, color="black"),
            ),
            name="Buy Trade",
            text=buys["size"],
            hovertemplate="Time: %{x}<br>Price: %{y}<br>Size: %{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["price"],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                color="#ff0000",
                size=6,
                line=dict(width=0.5, color="black"),
            ),
            name="Sell Trade",
            text=sells["size"],
            hovertemplate="Time: %{x}<br>Price: %{y}<br>Size: %{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --- ROW 2: VOLUME ---
    fig.add_trace(
        go.Scatter(
            x=ohlcv_sub.index,
            y=ohlcv_sub["volume"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#82b1ff", width=1, shape="hv"),
            name="1m Volume",
        ),
        row=2,
        col=1,
    )
    mid_price = (l2_sub["ask_px"] + l2_sub["bid_px"]) / 2
    pct_spread = ((l2_sub["ask_px"] - l2_sub["bid_px"]) / mid_price) * 100

    fig.add_trace(
        go.Scatter(
            x=l2_sub.index,
            y=pct_spread,
            mode="lines",
            line=dict(color="#ffa726", width=1.5, shape="hv"),
            fill="tozeroy",
            name="Spread %",
            hovertemplate="Time: %{x}<br>Spread: %{y:.4f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # --- ROW 4: L2 DEPTH IMBALANCE ---
    fig.add_trace(
        go.Scatter(
            x=l2_sub.index,
            y=l2_sub["bid_sz"],
            mode="lines",
            line=dict(color="#26a69a", width=1, shape="hv"),
            fill="tozeroy",
            name="Bid Size (Support)",
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=l2_sub.index,
            y=-l2_sub["ask_sz"],
            mode="lines",
            line=dict(color="#ef5350", width=1, shape="hv"),
            fill="tozeroy",
            name="Ask Size (Resistance)",
        ),
        row=4,
        col=1,
    )

    # --- LAYOUT SETTINGS ---
    fig.update_layout(
        template="plotly_dark",
        title_text=title,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=height,
        margin=dict(l=50, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(
        title_text="Spread %", ticksuffix="%", tickformat=".4f", row=3, col=1
    )
    fig.update_yaxes(title_text="Size (+Bid / -Ask)", row=4, col=1)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")

    return fig


Transformable = (
    str
    | dict[
        str,
        pd.Series | pd.DataFrame,
    ]
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

    if not isinstance(columns, list):
        columns = [columns]

    # Process names and data arrays before plotting
    plot_data = []
    for item in columns:
        if isinstance(item, str):
            plot_data.append((item, df[item]))
        elif isinstance(item, dict):
            # Expecting { "Label": series_or_expression }
            for label, series in item.items():
                plot_data.append((label, series))
        elif callable(item):
            # Pass the DF into the function, it should return a Series
            series = item(df)
            name = getattr(series, "name", f"Transform_{len(plot_data)}")
            plot_data.append((name, series))

    num_plots = len(plot_data)
    if separate_subplots:
        fig = make_subplots(
            rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05
        )
        for i, (name, series) in enumerate(plot_data, start=1):
            fig.add_trace(
                go.Scattergl(
                    x=series.index,
                    y=series,
                    mode=mode,
                    name=name,
                    line_shape=line_shape,
                ),
                row=i,
                col=1,
            )
            fig.update_yaxes(title_text=name, row=i, col=1)
    else:
        fig = go.Figure()
        for name, series in plot_data:
            fig.add_trace(
                go.Scattergl(
                    x=series.index,
                    y=series,
                    mode=mode,
                    name=name,
                    line_shape=line_shape,
                )
            )

    fig.update_layout(
        template="plotly_dark",
        title_text=title,
        height=height,
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    return fig
