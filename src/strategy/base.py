"""
strategy/base.py — Abstract Strategy base, PortfolioTarget, and SingleAssetStrategy.

The engine calls:
  1. strategy.setup(universe)        — once, to pre-compute indicators
  2. strategy.generate(ctx) per bar  — returns PortfolioTarget
  3. engine rebalances positions to match per-asset sizing, stops, costs

SingleAssetStrategy is a convenience base for strategies that trade one symbol.
Subclass it, implement `bar(data, idx) -> Allocation`, and optionally override
`setup_data(data, l2)`.  Everything else is auto-wired.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.models import Side, OrderBookSnapshot, Position, FundingSnapshot, Allocation
from .universe import Universe


# ── Portfolio target ─────────────────────────────────────────────────────────


@dataclass
class PortfolioTarget:
    """
    Output of Strategy.generate() — the desired portfolio state.

    Maps symbol → Allocation.  Assets not in the dict are assumed FLAT
    (i.e. close any existing position).
    """
    allocations: dict[str, Allocation] = field(default_factory=dict)
    timestamp: pd.Timestamp | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, symbol: str) -> Allocation:
        return self.allocations.get(symbol, Allocation())

    def __setitem__(self, symbol: str, alloc: Allocation):
        self.allocations[symbol] = alloc

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.allocations

    def active_symbols(self) -> list[str]:
        """Symbols with non-FLAT desired position."""
        return [s for s, a in self.allocations.items() if a.side != Side.FLAT]

    @property
    def total_weight(self) -> float:
        return sum(a.weight for a in self.allocations.values() if a.side != Side.FLAT)

    def normalize(self, max_total: float = 1.0):
        """Scale weights down proportionally if total exceeds max_total."""
        total = self.total_weight
        if total > max_total and total > 0:
            scale = max_total / total
            for alloc in self.allocations.values():
                alloc.weight *= scale


# ── Strategy context (passed to generate each bar) ──────────────────────────


@dataclass
class StrategyContext:
    """
    Everything a Strategy sees when generating targets for one bar.

    Contains the full universe (all assets + aux data), the current bar index,
    portfolio-level state, and per-asset position info.
    """
    universe: Universe
    bar_idx: int
    timestamp: pd.Timestamp
    equity: float
    positions: dict[str, Position]       # symbol → current position
    trade_history: list = field(default_factory=list)

    def price(self, symbol: str) -> float:
        """Current close price of an asset."""
        ohlcv = self.universe.ohlcv(symbol)
        if self.bar_idx < len(ohlcv):
            return ohlcv["close"].iat[self.bar_idx]
        return float("nan")

    def prices(self) -> dict[str, float]:
        """Current close prices for all assets."""
        return {s: self.price(s) for s in self.universe.symbols}

    def ohlcv(self, symbol: str) -> pd.DataFrame:
        """Full OHLCV up to current bar (inclusive)."""
        return self.universe.ohlcv(symbol).iloc[: self.bar_idx + 1]

    def aux(self, source_name: str) -> pd.DataFrame:
        """Auxiliary data up to current bar."""
        df = self.universe.aux(source_name)
        return df.iloc[: self.bar_idx + 1]

    def l2(self, symbol: str) -> OrderBookSnapshot | None:
        """Current L2 snapshot for an asset."""
        l2_list = self.universe.l2(symbol)
        if l2_list and self.bar_idx < len(l2_list):
            return l2_list[self.bar_idx]
        return None

    def funding(self, symbol: str) -> FundingSnapshot | None:
        """Current funding rate snapshot for an asset."""
        return self.universe.funding_at(symbol, self.bar_idx)

    def is_positioned(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.side != Side.FLAT

    def net_exposure(self) -> float:
        """Net dollar exposure as fraction of equity."""
        total = 0.0
        for sym, pos in self.positions.items():
            if pos.side != Side.FLAT:
                px = self.price(sym)
                direction = 1 if pos.side == Side.LONG else -1
                total += direction * pos.size * px
        return total / self.equity if self.equity > 0 else 0.0


# ── Abstract Strategy base ──────────────────────────────────────────────────


class Strategy(abc.ABC):
    """
    Multi-asset trading strategy.

    Subclass and implement:
      • setup(universe)      — pre-compute indicators across all assets
      • generate(ctx)        — return PortfolioTarget for current bar
      • params               — tunable parameters for stress testing
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def setup(self, universe: Universe):
        """
        Pre-compute indicators on all assets in the universe.
        Called once before the backtest loop (or on live engine start).
        """
        ...

    @abc.abstractmethod
    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        """
        Return the desired portfolio allocation for the current bar.

        The engine will diff this against current positions and execute
        the necessary trades (entries, exits, rebalances).
        """
        ...

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, Any]:
        """Dict of tunable parameters (for optimization / stress tests)."""
        ...

    def set_params(self, new: dict[str, Any]):
        for k, v in new.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def generate_all(
        self, universe: Universe
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
        """
        Optional batch generation for all bars at once.

        Returns ``(sides, weights)`` where:
          - ``sides[sym]``   — ``int8`` array of Side values per bar
          - ``weights[sym]`` — ``float64`` array of allocation weights per bar

        The engine activates the vectorised fast path when this returns non-``None``
        **and** all stops are :class:`~risk.stops.NopStopLoss` **and** all sizers
        are :attr:`~risk.sizing.Sizer.vectorizable`.

        Return ``None`` (default) to fall back to the per-bar :meth:`generate` path.
        """
        return None

    def on_fill(self, symbol: str, side: Side, size: float, price: float):
        """Optional callback when a fill occurs (for bookkeeping)."""
        pass


# ── SingleAssetStrategy — convenience base replacing Signal ─────────────────


class SingleAssetStrategy(Strategy):
    """
    Convenience base for single-asset strategies.

    Implement ``bar(data, idx) -> Allocation`` and optionally override
    ``setup_data(data, l2)`` for indicator pre-computation.
    ``setup``, ``generate``, and ``generate_all`` are auto-wired.

    Migration from the old Signal API:
      Signal.setup(data, l2)         →  SingleAssetStrategy.setup_data(data, l2)
      Signal.generate(data, idx)     →  SingleAssetStrategy.bar(data, idx)
      @register_signal("name")       →  @register_strategy("name")
      SignalResult(target_side=...,  →  Allocation(side=..., weight=...)
                   target_weight=...)
    """

    def __init__(self, symbol: str, **kw):
        super().__init__(**kw)
        self.symbol = symbol

    def setup_data(
        self, data: pd.DataFrame, l2: list[OrderBookSnapshot] | None = None
    ):
        """Optional: pre-compute indicator columns on data in-place."""
        pass

    @abc.abstractmethod
    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        """Return the desired allocation for bar ``idx``."""
        ...

    # ── Auto-wired — generally do not override ──────────────────────────

    def setup(self, universe: Universe):
        data = universe.ohlcv(self.symbol)
        l2 = universe.l2(self.symbol)
        self.setup_data(data, l2)

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        data = ctx.universe.ohlcv(self.symbol)
        alloc = self.bar(data, ctx.bar_idx)
        target = PortfolioTarget(timestamp=ctx.timestamp)
        target[self.symbol] = alloc
        return target

    def generate_all(
        self, universe: Universe
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
        data = universe.ohlcv(self.symbol)
        n = len(data)
        sides = np.zeros(n, dtype=np.int8)
        weights = np.zeros(n, dtype=np.float64)
        for i in range(n):
            a = self.bar(data, i)
            sides[i] = np.int8(a.side.value)
            weights[i] = a.weight
        return {self.symbol: sides}, {self.symbol: weights}

    @property
    def params(self) -> dict[str, Any]:
        return {}


# ── Strategy registry ────────────────────────────────────────────────────────


_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(name: str):
    """Class decorator: @register_strategy("pairs_eth_btc")."""
    def _wrap(cls):
        _STRATEGY_REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return _wrap


def get_strategy(name: str) -> type[Strategy]:
    if name not in _STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy '{name}' not registered. Available: {list(_STRATEGY_REGISTRY)}"
        )
    return _STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    return list(_STRATEGY_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════
#  MultiExchangeTarget — output of a CrossExchangeStrategy
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MultiExchangeTarget:
    """
    Desired portfolio state across all exchanges.

    Maps (exchange, symbol) → Allocation.
    Missing keys are assumed FLAT (close any existing position).
    """
    allocations: dict[tuple[str, str], Allocation] = field(default_factory=dict)
    timestamp: pd.Timestamp | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: tuple[str, str]) -> Allocation:
        return self.allocations.get(key, Allocation())

    def __setitem__(self, key: tuple[str, str], alloc: Allocation):
        self.allocations[key] = alloc

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self.allocations

    def for_exchange(self, exchange: str) -> PortfolioTarget:
        """Extract a single-exchange PortfolioTarget (for sizer/stop compat)."""
        allocs = {}
        for (ex, sym), alloc in self.allocations.items():
            if ex == exchange:
                allocs[sym] = alloc
        return PortfolioTarget(
            allocations=allocs,
            timestamp=self.timestamp,
        )

    def active_legs(self) -> list[tuple[str, str, Allocation]]:
        """All non-FLAT allocations as (exchange, symbol, alloc)."""
        return [
            (ex, sym, a)
            for (ex, sym), a in self.allocations.items()
            if a.side != Side.FLAT
        ]

    def symbols_on(self, exchange: str) -> list[str]:
        """Symbols with non-FLAT allocations on a specific exchange."""
        return [
            sym for (ex, sym), a in self.allocations.items()
            if ex == exchange and a.side != Side.FLAT
        ]

    @property
    def total_weight(self) -> float:
        return sum(
            a.weight for a in self.allocations.values()
            if a.side != Side.FLAT
        )

    @property
    def exchanges(self) -> list[str]:
        """All exchanges referenced in this target."""
        return list({ex for ex, _ in self.allocations.keys()})

    def normalize(self, max_total: float = 1.0):
        """Scale all weights down proportionally if total exceeds max."""
        total = self.total_weight
        if total > max_total and total > 0:
            scale = max_total / total
            for alloc in self.allocations.values():
                alloc.weight *= scale

    @staticmethod
    def from_per_exchange(
        targets: dict[str, PortfolioTarget],
    ) -> MultiExchangeTarget:
        """
        Merge per-exchange PortfolioTargets into one MultiExchangeTarget.
        Used by the engine when running independent per-exchange strategies.
        """
        merged = MultiExchangeTarget()
        for exchange, pt in targets.items():
            merged.timestamp = merged.timestamp or pt.timestamp
            for sym, alloc in pt.allocations.items():
                merged[(exchange, sym)] = alloc
        return merged


# ═══════════════════════════════════════════════════════════════════════════
#  CrossExchangeContext — everything the strategy sees per bar
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CrossExchangeContext:
    """
    Full state visible to a CrossExchangeStrategy on each bar.

    Provides per-exchange universes, positions, equity, and the shared
    portfolio aggregator for net-position queries.
    """
    universes: dict[str, Universe]
    bar_idx: int
    timestamp: pd.Timestamp
    total_equity: float
    equity_by_exchange: dict[str, float]
    # exchange → {symbol → Position}
    positions: dict[str, dict[str, Position]]
    portfolio: Any  # MultiExchangePortfolio (avoid circular import)
    trade_history: list = field(default_factory=list)

    # ── Convenience methods ──────────────────────────────────────────

    def price(self, symbol: str, exchange: str | None = None) -> float:
        """
        Current close price of a symbol.
        If exchange given, read from that exchange's universe.
        Otherwise, return first available.
        """
        if exchange and exchange in self.universes:
            u = self.universes[exchange]
            try:
                ohlcv = u.ohlcv(symbol)
                if self.bar_idx < len(ohlcv):
                    return ohlcv["close"].iat[self.bar_idx]
            except KeyError:
                pass
        for u in self.universes.values():
            try:
                ohlcv = u.ohlcv(symbol)
                if self.bar_idx < len(ohlcv):
                    return ohlcv["close"].iat[self.bar_idx]
            except KeyError:
                continue
        return float("nan")

    def prices(self, exchange: str | None = None) -> dict[str, float]:
        """Current close prices for all symbols."""
        if exchange and exchange in self.universes:
            syms = self.universes[exchange].symbols
            return {s: self.price(s, exchange) for s in syms}
        all_syms = set()
        for u in self.universes.values():
            all_syms.update(u.symbols)
        return {s: self.price(s) for s in all_syms}

    def ohlcv(self, symbol: str, exchange: str | None = None) -> pd.DataFrame:
        """Full OHLCV up to current bar."""
        for ex_name in ([exchange] if exchange else self.universes.keys()):
            if ex_name in self.universes:
                try:
                    return self.universes[ex_name].ohlcv(symbol).iloc[:self.bar_idx + 1]
                except KeyError:
                    continue
        return pd.DataFrame()

    def funding(self, symbol: str, exchange: str | None = None) -> FundingSnapshot | None:
        """Current funding rate snapshot for a symbol on a given (or any) exchange."""
        for ex_name in ([exchange] if exchange else self.universes.keys()):
            if ex_name in self.universes:
                snap = self.universes[ex_name].funding_at(symbol, self.bar_idx)
                if snap is not None:
                    return snap
        return None

    def position_on(self, exchange: str, symbol: str) -> Position:
        """Get position on a specific exchange."""
        return self.positions.get(exchange, {}).get(symbol, Position())

    def net_position(self, symbol: str):
        """Aggregated position across all exchanges."""
        return self.portfolio.net_position(symbol)

    def is_positioned(self, symbol: str, exchange: str | None = None) -> bool:
        """Check if positioned on a specific or any exchange."""
        if exchange:
            pos = self.position_on(exchange, symbol)
            return pos.side != Side.FLAT
        for ex_positions in self.positions.values():
            if ex_positions.get(symbol, Position()).side != Side.FLAT:
                return True
        return False

    def net_exposure_pct(self, symbols: list[str] | None = None) -> float:
        """Net dollar exposure as fraction of total equity."""
        if self.total_equity <= 0:
            return 0.0
        syms = symbols or list({
            s for u in self.universes.values() for s in u.symbols
        })
        px = self.prices()
        return self.portfolio.net_exposure(syms, px) / self.total_equity


# ═══════════════════════════════════════════════════════════════════════════
#  CrossExchangeStrategy — ABC
# ═══════════════════════════════════════════════════════════════════════════


class CrossExchangeStrategy(abc.ABC):
    """
    Strategy that operates across multiple exchanges simultaneously.

    Unlike Strategy (which sees one Universe and returns PortfolioTarget),
    this sees ALL exchanges and returns MultiExchangeTarget with explicit
    routing per (exchange, symbol).

    Subclass and implement:
      • setup(universes)       — pre-compute indicators per exchange
      • generate(ctx)          — return MultiExchangeTarget
      • params                 — tunable parameters
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def setup(self, universes: dict[str, Universe]):
        """
        Pre-compute indicators across all exchanges.
        Called once before each bar processing cycle.
        universes is keyed by exchange name.
        """
        ...

    @abc.abstractmethod
    def generate(self, ctx: CrossExchangeContext) -> MultiExchangeTarget:
        """
        Generate desired allocations across all exchanges.

        The engine diffs this against current positions per exchange
        and executes the necessary trades on each.
        """
        ...

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, Any]:
        ...

    def set_params(self, new: dict[str, Any]):
        for k, v in new.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def on_fill(
        self,
        exchange: str,
        symbol: str,
        side: Side,
        size: float,
        price: float,
    ):
        """Optional callback when a fill occurs on any exchange."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CrossExchangeStrategy registry
# ═══════════════════════════════════════════════════════════════════════════


_CROSS_STRATEGY_REGISTRY: dict[str, type[CrossExchangeStrategy]] = {}


def register_cross_strategy(name: str):
    """Class decorator: @register_cross_strategy("funding_arb")."""
    def _wrap(cls):
        _CROSS_STRATEGY_REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return _wrap


def get_cross_strategy(name: str) -> type[CrossExchangeStrategy]:
    if name not in _CROSS_STRATEGY_REGISTRY:
        raise KeyError(
            f"CrossExchangeStrategy '{name}' not registered. "
            f"Available: {list(_CROSS_STRATEGY_REGISTRY)}"
        )
    return _CROSS_STRATEGY_REGISTRY[name]


def list_cross_strategies() -> list[str]:
    return list(_CROSS_STRATEGY_REGISTRY.keys())
