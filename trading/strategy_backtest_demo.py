"""
examples/multi_asset_demo.py — Demonstrates the multi-asset strategy framework.

Shows four patterns:
  1. Wrapping an existing Signal (backward compat)
  2. Running different signals per asset
  3. Built-in pairs spread strategy
  4. Custom strategy with auxiliary data (funding rates)
"""

from pathlib import Path

from core.models import BacktestConfig, Side
from core.parser import trades_to_ohlcv, l2_to_orderbook, funding_to_snapshots, align_funding_to_ohlcv

from backtester.engine import Backtester
from backtester.costs import CompositeCostModel, aggressive_cost_stack
from backtester.stress import SignalStressTest

from strategy.base import Signal, SignalResult, register_signal
from strategy.indicators import ema
from risk.sizing import VolatilityTargetSizer, CompositeSizer, KellySizer
from risk.stops import SignalStop

from strategy.universe import Universe
from strategy.built_in import SingleSignalStrategy


# ═══════════════════════════════════════════════════════════════════════════
#  Pattern 1: Wrap an existing single-asset Signal
# ═══════════════════════════════════════════════════════════════════════════


@register_signal("demo_ema_cross")
class DemoEMACross(Signal):
    """Simple EMA crossover for demo."""

    def __init__(self, fast: int = 12, slow: int = 26, **kw):
        super().__init__(**kw)
        self.fast = fast
        self.slow = slow

    @property
    def params(self):
        return {"fast": self.fast, "slow": self.slow}

    def setup(self, data, l2=None):
        data["ema_fast"] = ema(data["close"], self.fast)
        data["ema_slow"] = ema(data["close"], self.slow)

    def generate(self, data, idx):
        if idx < self.slow:
            return SignalResult()
        fast_val = data["ema_fast"].iat[idx]
        slow_val = data["ema_slow"].iat[idx]
        if fast_val > slow_val:
            return SignalResult(
                target_side=Side.LONG,
                target_weight=0.8,
                confidence=0.6,
                reason=f"EMA cross up",
            )
        elif fast_val < slow_val:
            return SignalResult(
                target_side=Side.SHORT,
                target_weight=0.8,
                confidence=0.6,
                reason=f"EMA cross down",
            )
        return SignalResult()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def demo():
    timeframe = "1m"
    eth_data = trades_to_ohlcv(
        DATA_DIR / "trades" / "HYPERLIQUID_PERPETUALS" / "ETH",
        timeframe=timeframe,
    )

    config = BacktestConfig(
        initial_capital=1000,
        risk_per_trade=0.02,
        max_position_pct=0.25,
        leverage=1.0,
        taker_fee_bps=5.0,
        slippage_bps=1.0,
    )

    sizer = CompositeSizer(
        sizers=[VolatilityTargetSizer(target_vol=0.15)]
    )

    stop = SignalStop()

    cost = CompositeCostModel(models=aggressive_cost_stack())

    signal = DemoEMACross(fast=2, slow=5)

    # A universe can hold as many strategies as it wants and can include axiliary data sources
    universe = Universe(symbols=["ETH"])  
    universe.add_asset("ETH", eth_data)

    bt = Backtester(
        signal=signal, config=config, cost_model=cost, sizer=sizer, stop_loss=stop
    )
    result = bt.run(data=eth_data, timeframe=timeframe)

    print(result.summary())

    run_dir = result.save("demo_ema_cross")
    print(f"Backtest saved to: {run_dir}")

    stress = SignalStressTest(
        signal_cls=DemoEMACross,
        param_grid={"fast": [2, 3], "slow": [5, 7]},
        cost_model=cost,
    )

    sweep = stress.run(data=eth_data, timeframe=timeframe)
    sweep.plot_heatmap(x="fast", y="slow", save_path=f"{run_dir}/heatmap_signal_sweep.png")


if __name__ == "__main__":
    demo()