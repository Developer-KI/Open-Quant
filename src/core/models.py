"""
core/models.py — Canonical data models for the entire trading system.

All other modules import their data types from here.
Design rule: no imports from other src/ modules in this file.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class Side(enum.Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


# ── Order Book ───────────────────────────────────────────────────────────────


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Single L2 snapshot — lists of (price, size) per side, best → worst."""

    timestamp: pd.Timestamp
    bids: list[OrderBookLevel]  # best bid first
    asks: list[OrderBookLevel]  # best ask first

    @property
    def book_depth(self) -> int:
        return min(len(self.bids), len(self.asks))

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2.0
        return np.nan

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return np.nan

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        if mid and mid > 0:
            return (self.spread / mid) * 1e4
        return np.nan

    def depth_at(self, pct_from_mid: float = 0.01) -> dict[str, float]:
        """Cumulative size within `pct_from_mid` of the midprice."""
        mid = self.mid
        bid_depth = sum(
            lvl.size for lvl in self.bids if lvl.price >= mid * (1 - pct_from_mid)
        )
        ask_depth = sum(
            lvl.size for lvl in self.asks if lvl.price <= mid * (1 + pct_from_mid)
        )
        return {"bid_depth": bid_depth, "ask_depth": ask_depth}

    def vwap_fill_price(self, size: float, side: Side) -> float:
        """Walk the book and return the VWAP fill price for a given size."""
        levels = self.asks if side == Side.LONG else self.bids
        remaining = abs(size)
        cost = 0.0
        for lvl in levels:
            take = min(remaining, lvl.size)
            cost += take * lvl.price
            remaining -= take
            if remaining <= 0:
                break
        filled = abs(size) - remaining
        if filled == 0:
            return np.nan
        return cost / filled


# ── Funding Rate Snapshot ─────────────────────────────────────────────────────


@dataclass
class FundingSnapshot:
    """
    Single funding rate observation for a perpetual swap.

    Fields:
        timestamp:        when this rate was observed / published
        rate:             per-period funding rate (e.g. 0.0001 = 1 bps)
        rate_annualized:  convenience pre-computed annual rate in bps
        oracle_price:     spot / index price the exchange uses as reference
        mark_price:       fair price used for funding & liquidation calcs
    """
    timestamp: pd.Timestamp
    rate: float
    rate_annualized: float
    oracle_price: float = 0.0
    mark_price: float = 0.0


# ── Trade / Position ─────────────────────────────────────────────────────────


@dataclass
class Trade:
    timestamp: pd.Timestamp
    side: Side
    size: float
    entry_price: float
    exit_price: float | None = None
    exit_timestamp: pd.Timestamp | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    reason_entry: str = ""
    reason_exit: str = ""
    bar_values: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "exit_timestamp": self.exit_timestamp,
            "side": self.side.name,
            "size": self.size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "fees": self.fees,
            "slippage": self.slippage,
            "reason_entry": self.reason_entry,
            "reason_exit": self.reason_exit,
            **{f"bar_{k}": v for k, v in self.bar_values.items()},
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }


@dataclass
class Position:
    side: Side = Side.FLAT
    size: float = 0.0
    entry_price: float = 0.0
    entry_timestamp: pd.Timestamp | None = None
    unrealized_pnl: float = 0.0


# ── Allocation (lives here so risk/ can import it without touching strategy/) ──


@dataclass
class Allocation:
    """Desired state for one asset — output of SingleAssetStrategy.bar()."""

    side: Side = Side.FLAT
    weight: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    order_type: str = "market"
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ── Fill result (shared across all exchanges) ─────────────────────────────────


@dataclass
class FillResult:
    """Result of an order submission — exchange-agnostic."""
    success: bool
    fill_price: float = 0.0
    filled_size: float = 0.0
    order_id: str = ""
    status: str = ""
    exchange: str = ""
    raw: dict = None

    def __post_init__(self):
        if self.raw is None:
            self.raw = {}


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    position_sizing: str = "fixed_fractional"
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.25
    use_l2_fills: bool = False
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    funding_rate_annual_bps: float = 0.0
    slippage_model: str = "fixed"
    slippage_bps: float = 1.0
    margin_type: str = "cross"
    leverage: float = 1.0
    export_path: str = "backtest_output"


@dataclass
class ExchangeCredentials:
    """
    Credentials for one exchange account.

    Each exchange has different auth requirements:
      • Hyperliquid: account_address + secret_key
      • Binance:     api_key + api_secret
    """
    exchange: str = ""
    api_key: str = ""
    api_secret: str = ""
    account_address: str = ""
    secret_key: str = ""
    testnet: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    symbol_map: dict[str, str] = field(default_factory=dict)


@dataclass
class LiveConfig:
    """All settings needed to run a strategy live on one or more exchanges."""

    # ── Exchange credentials ─────────────────────────────────────────────
    exchange: str = "hyperliquid"
    account_address: str = ""
    secret_key: str = ""
    api_key: str = ""
    api_secret: str = ""
    use_testnet: bool = True
    exchanges: list[ExchangeCredentials] | None = None

    # ── Market ───────────────────────────────────────────────────────────
    symbol: str = "ETH"
    symbols: list[str] | None = None
    bar_interval_s: int = 60

    # ── Warm-up ──────────────────────────────────────────────────────────
    warmup_bars: int = 200
    max_bars_in_memory: int = 2000

    # ── Position sizing / risk ───────────────────────────────────────────
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.25
    leverage: float = 1.0
    margin_type: str = "cross"

    # ── Risk hard limits ─────────────────────────────────────────────────
    max_open_orders: int = 3
    max_daily_trades: int = 50
    max_daily_loss_pct: float = 5.0
    cooldown_after_loss_s: int = 0

    # ── Order execution ──────────────────────────────────────────────────
    order_type: str = "market"
    limit_chase_bps: float = 2.0
    cancel_stale_after_s: float = 10.0
    reduce_only_exits: bool = True

    # ── Logging ──────────────────────────────────────────────────────────
    log_dir: str = "logs/live"
    log_level: str = "INFO"
    trade_log_csv: str = "live_trades.csv"

    @property
    def active_symbols(self) -> list[str]:
        if self.symbols:
            return list(self.symbols)
        return [self.symbol]

    @property
    def is_multi_asset(self) -> bool:
        return len(self.active_symbols) > 1

    @property
    def is_multi_exchange(self) -> bool:
        return self.exchanges is not None and len(self.exchanges) > 1

    @property
    def api_url(self) -> str:
        if self.exchange == "hyperliquid":
            if self.use_testnet:
                return "https://api.hyperliquid-testnet.xyz"
            return "https://api.hyperliquid.xyz"
        if self.exchange == "binance":
            if self.use_testnet:
                return "https://testnet.binancefuture.com"
            return "https://fapi.binance.com"
        return ""

    @property
    def ws_url(self) -> str:
        if self.exchange == "hyperliquid":
            if self.use_testnet:
                return "wss://api.hyperliquid-testnet.xyz/ws"
            return "wss://api.hyperliquid.xyz/ws"
        if self.exchange == "binance":
            if self.use_testnet:
                return "wss://fstream.binancefuture.com/ws"
            return "wss://fstream.binance.com/ws"
        return ""

    def get_credentials(self) -> list[ExchangeCredentials]:
        if self.exchanges:
            return list(self.exchanges)
        return [ExchangeCredentials(
            exchange=self.exchange,
            api_key=self.api_key,
            api_secret=self.api_secret,
            account_address=self.account_address,
            secret_key=self.secret_key,
            testnet=self.use_testnet,
        )]


# ── Portfolio aggregation ─────────────────────────────────────────────────────


@dataclass
class ExchangePosition:
    """Position on a specific exchange."""
    exchange: str
    symbol: str
    side: Side = Side.FLAT
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class AggregatedPosition:
    """
    Net position across all exchanges for one symbol.

    If you're long 10 ETH on Hyperliquid and short 6 ETH on Binance,
    the net is long 4 ETH.
    """
    symbol: str
    net_size: float = 0.0
    net_side: Side = Side.FLAT
    gross_long: float = 0.0
    gross_short: float = 0.0
    per_exchange: list[ExchangePosition] = field(default_factory=list)

    @property
    def is_hedged(self) -> bool:
        return self.gross_long > 0 and self.gross_short > 0

    @property
    def gross_exposure(self) -> float:
        return self.gross_long + self.gross_short
