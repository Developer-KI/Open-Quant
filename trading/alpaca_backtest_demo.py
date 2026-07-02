from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.models import BacktestConfig, Side, Allocation
from backtester.engine import Backtester
from strategy.base import SingleAssetStrategy, register_strategy
from strategy.indicators import ema, rsi, atr
from strategy.universe import Universe
from risk.sizing import FixedNotionalSizer
from risk.stops import NopStopLoss
from backtester.costs import NullCostModel


# ═══════════════════════════════════════════════════════════════════════════
#  Data fetching
# ═══════════════════════════════════════════════════════════════════════════
def load_credentials() -> dict:
    load_dotenv()
    return {
        "key": os.getenv("ALP_PAPER_KEY", ""),
        "secret": os.getenv("ALP_PAPER_SECRET", ""),
    }


def fetch_alpaca_bars(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1d",
    api_key: str | None = None,
    api_secret: str | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca for a single symbol.

    Parameters
    ----------
    symbol    : ticker, e.g. "AAPL", "SPY"
    start     : ISO date string, e.g. "2023-01-01"
    end       : ISO date string, e.g. "2024-01-01"
    timeframe : one of "1d", "1h", "30m", "15m", "5m", "1m"
    api_key   : Alpaca key; falls back to ALPACA_KEY env var
    api_secret: Alpaca secret; falls back to ALPACA_SECRET env var

    Returns
    -------
    DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: alpaca-py. Install with: pip install alpaca-py"
        ) from exc

    key = api_key or os.environ.get("ALPACA_KEY", "")
    secret = api_secret or os.environ.get("ALPACA_SECRET", "")
    if not key or not secret:
        raise ValueError(
            "Alpaca credentials required. Set ALPACA_KEY and ALPACA_SECRET "
            "environment variables, or pass api_key/api_secret directly."
        )

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    tf_map = {
        "1d":  TimeFrame.Day,
        "1h":  TimeFrame.Hour,
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "5m":  TimeFrame(5, TimeFrameUnit.Minute),
        "1m":  TimeFrame.Minute,
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {list(tf_map)}")

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe],
        start=pd.Timestamp(start, tz="US/Eastern"),
        end=pd.Timestamp(end, tz="US/Eastern"),
        adjustment="all",  # split + dividend adjusted
    )
    bars = client.get_stock_bars(req)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy: EMA crossover with RSI filter
# ═══════════════════════════════════════════════════════════════════════════


@register_strategy("alpaca_ema_rsi")
class AlpacaEMARSIStrategy(SingleAssetStrategy):
    """
    Long-only EMA crossover filtered by RSI.

    Entry:  fast EMA crosses above slow EMA AND RSI is not overbought
    Exit:   fast EMA crosses below slow EMA OR RSI becomes overbought
    Weight: proportional to ATR-normalised trend strength, capped at 1.0

    Designed for daily equity bars on Alpaca paper trading.
    """

    def __init__(
        self,
        symbol: str,
        fast: int = 10,
        slow: int = 30,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        atr_period: int = 14,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period

    @property
    def params(self) -> dict:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "rsi_period": self.rsi_period,
            "rsi_overbought": self.rsi_overbought,
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        data["ema_fast"] = ema(data["close"], self.fast)
        data["ema_slow"] = ema(data["close"], self.slow)
        data["rsi"] = rsi(data["close"], self.rsi_period)
        data["atr"] = atr(data["high"], data["low"], data["close"], self.atr_period)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.slow + self.atr_period:
            return Allocation()

        ef = data["ema_fast"].iat[idx]
        es = data["ema_slow"].iat[idx]
        rsi_val = data["rsi"].iat[idx]
        atr_val = data["atr"].iat[idx]
        price = data["close"].iat[idx]

        if any(v != v for v in (ef, es, rsi_val, atr_val)):  # NaN check
            return Allocation()

        bull_cross = ef > es
        rsi_ok = rsi_val < self.rsi_overbought

        if rsi_ok:
            if bull_cross:
                trend_strength = (ef - es) / (atr_val if atr_val > 0 else price * 0.01)
                weight = min(trend_strength * 0.5, 1.0)
                return Allocation(
                    side=Side.LONG,
                    weight=weight,
                    confidence=min(trend_strength / 3, 1.0),
                    reason=f"EMA cross up | RSI={rsi_val:.0f} | strength={trend_strength:.2f}",
                )
            elif not bull_cross:
                trend_strength = (es - ef) / (atr_val if atr_val > 0 else price * 0.01)
                weight = min(trend_strength * 0.5, 1.0)
                return Allocation(
                    side=Side.SHORT,
                    weight=weight,
                    confidence=min(trend_strength / 3, 1.0),
                    reason=f"EMA cross up | RSI={rsi_val:.0f} | strength={trend_strength:.2f}",
                )

        return Allocation(reason=f"no signal | EMA bull={bull_cross} | RSI={rsi_val:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Demo runner
# ═══════════════════════════════════════════════════════════════════════════


def demo(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-01-01",
    timeframe: str = "1h",
):
    creds = load_credentials()
    print(f"\nFetching {symbol} {timeframe} bars from Alpaca ({start} → {end})...")
    data = fetch_alpaca_bars(symbol, start=start, end=end, timeframe=timeframe, api_key=creds["key"], api_secret=creds["secret"])
    print(f"  {len(data)} bars loaded  |  {data.index[0].date()} → {data.index[-1].date()}")

    universe = Universe(symbols=[symbol])
    universe.add_asset(symbol, data)

    config = BacktestConfig(
        initial_capital=1_000_000.0,
        max_position_pct=1.0,
        leverage=1.0,
    )

    strategy = AlpacaEMARSIStrategy(symbol=symbol, fast=50, slow=200)
    sizer = FixedNotionalSizer()
    stoploss = NopStopLoss()
    no_cost = NullCostModel()

    bt = Backtester(
        strategy=strategy,
        config=config,
        sizer=sizer,
        stop_loss=stoploss,
        cost_model=no_cost,
    )

    print("\nRunning backtest...")
    result = bt.run(universe=universe, timeframe=timeframe)

    run_dir = result.save(f"alpaca_demo")
    print(f"Backtest saved to: {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alpaca backtest demo")
    parser.add_argument("--symbol",    default="SPY",        help="Ticker symbol")
    parser.add_argument("--start",     default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",       default="2024-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--timeframe", default="1d",
                        choices=["1d", "1h", "30m", "15m", "5m", "1m"],
                        help="Bar size")
    args = parser.parse_args()

    demo(symbol=args.symbol, start=args.start, end=args.end, timeframe=args.timeframe)
