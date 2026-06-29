"""
run_live_ema.py — Deploy the EMA crossover signal to Hyperliquid.

This mirrors ema_crossover_strategy.py but runs LIVE instead of backtesting.
The same Signal, Sizer, and StopLoss objects are reused verbatim.

Usage:
    # 1. Set your credentials in config.json or environment variables:
    #    export HL_ACCOUNT_ADDRESS="0x..."
    #    export HL_SECRET_KEY="0x..."
    #
    # 2. Run:
    #    python run_live_ema.py

    # The engine will:
    #    - Connect to Hyperliquid testnet WebSocket
    #    - Fetch 200 bars of history for indicator warm-up
    #    - Start processing live 1m bars
    #    - Execute trades via the Hyperliquid API
    #    - Log all activity to logs/live_engine.log
    #    - Write closed trades to logs/live/live_trades.csv
"""

import os
from dotenv import load_dotenv

import pandas as pd

from core.models import LiveConfig, Side
from execution.live_engine import LiveEngine

from strategy.base import Signal, SignalResult, register_signal
from strategy.indicators import compute_atr_column, ema
from risk.sizing import VolatilityTargetSizer, CompositeSizer, KellySizer


@register_signal("ema_crossover")
class EMACrossover(Signal):
    """Go long when fast EMA > slow EMA, short when fast < slow."""

    def __init__(self, fast: int = 12, slow: int = 26, **kw):
        super().__init__(**kw)
        self.fast = fast
        self.slow = slow

    @property
    def params(self):
        return dict(fast=self.fast, slow=self.slow)

    def setup(self, data: pd.DataFrame, l2=None):
        data["ema_fast"] = ema(data["close"], self.fast)
        data["ema_slow"] = ema(data["close"], self.slow)
        compute_atr_column(data)

    def generate(self, data: pd.DataFrame, idx: int) -> SignalResult:
        if idx < self.slow:
            return SignalResult()

        fast_val = data["ema_fast"].iat[idx]
        slow_val = data["ema_slow"].iat[idx]

        if fast_val > slow_val:
            return SignalResult(
                target_side=Side.LONG,
                target_weight=1.0,
                confidence=0.7,
                reason=f"EMA {self.fast} > EMA {self.slow}",
            )
        elif fast_val < slow_val:
            return SignalResult(
                target_side=Side.SHORT,
                target_weight=1.0,
                confidence=0.7,
                reason=f"EMA {self.fast} < EMA {self.slow}",
            )
        return SignalResult()


# ── Load credentials ─────────────────────────────────────────────────

def load_credentials() -> dict:
    load_dotenv()

    return {
        "account_address": os.getenv("HL_ACCOUNT_ADDRESS", ""),
        "secret_key": os.getenv("HL_SECRET_KEY", ""),
    }

# ── Configure (same params as your backtest) ─────────────────────────

def main():
    creds = load_credentials()

    if not creds["account_address"] or not creds["secret_key"]:
        print("ERROR: Set HL_ACCOUNT_ADDRESS and HL_SECRET_KEY env vars")
        return

    config = LiveConfig(
        exchange="hyperliquid",
        account_address=creds["account_address"],
        secret_key=creds["secret_key"],
        use_testnet=True,       # ← switch to False for mainnet
        symbol="ETH",           # ← your Hyperliquid coin
        bar_interval_s=60,      # 1-minute bars (same as backtest)
        warmup_bars=200,
        max_daily_trades=50,
        max_daily_loss_pct=5.0,
        # Sizing / risk — same values as your backtest
        risk_per_trade=0.02,
        max_position_pct=0.25,
        leverage=1.0,
    )

    # Same signal, sizer, and stop as the backtest
    signal = EMACrossover(fast=16, slow=22)

    sizer = CompositeSizer(
        sizers=[VolatilityTargetSizer(target_vol=0.15),
                KellySizer(kelly_frac=0.5)],
        mode="avg",
    )

    # ── Launch ───────────────────────────────────────────────────────
    engine = LiveEngine(
        signal=signal,
        config=config,
        sizer=sizer,
    )

    print(f"Starting live engine on {'testnet' if config.use_testnet else 'MAINNET'}...")
    print(f"Symbol: {config.symbol} | Signal: {signal.__class__.__name__}")
    print(f"Press Ctrl+C to stop.\n")

    engine.start()  # Blocks until Ctrl+C or kill switch


if __name__ == "__main__":
    main()