"""
risk/stops.py — Modular stop-loss and take-profit framework.

Moved here from strategy/stoploss.py so the risk layer is independent of the
strategy layer. strategy/stoploss.py now re-exports from here for backward
compatibility.

Dependency: core/ only — no imports from strategy/ or execution/.
"""

from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.models import Position, Side, OrderBookSnapshot


# ── Stop result ───────────────────────────────────────────────────────────────


@dataclass
class StopResult:
    triggered: bool = False
    exit_price: float = 0.0
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# ── Stop context ──────────────────────────────────────────────────────────────


@dataclass
class StopContext:
    position: Position
    bar_idx: int
    open: float
    high: float
    low: float
    close: float
    data: pd.DataFrame | None = None
    l2: OrderBookSnapshot | None = None
    bar_data: dict[str, float] = field(default_factory=dict)


# ── Abstract base ─────────────────────────────────────────────────────────────


class StopLoss(abc.ABC):
    """
    Base class for stop-loss / take-profit strategies.

    Lifecycle:
      1. on_entry(position, ctx) — initialize state when position opens
      2. update(ctx)             — update trailing levels on each bar
      3. check(ctx) → StopResult — check if any exit triggered
      4. reset()                 — called when position closes
    """

    @abc.abstractmethod
    def on_entry(self, position: Position, ctx: StopContext): ...

    @abc.abstractmethod
    def update(self, ctx: StopContext): ...

    @abc.abstractmethod
    def check(self, ctx: StopContext) -> StopResult: ...

    def reset(self):
        pass

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, Any]: ...

    def set_params(self, new: dict[str, Any]):
        for k, v in new.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"


# ── Concrete implementations ──────────────────────────────────────────────────


class FixedPercentStop(StopLoss):
    """Fixed percentage stop-loss and optional take-profit from entry price."""

    def __init__(self, sl_pct: float = 2.0, tp_pct: float | None = None):
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self._sl_price: float = 0
        self._tp_price: float | None = None
        self._side: Side = Side.FLAT

    @property
    def params(self):
        return dict(sl_pct=self.sl_pct, tp_pct=self.tp_pct)

    def on_entry(self, position: Position, ctx: StopContext):
        self._side = position.side
        entry = position.entry_price
        if self._side == Side.LONG:
            self._sl_price = entry * (1 - self.sl_pct / 100)
            self._tp_price = entry * (1 + self.tp_pct / 100) if self.tp_pct else None
        else:
            self._sl_price = entry * (1 + self.sl_pct / 100)
            self._tp_price = entry * (1 - self.tp_pct / 100) if self.tp_pct else None

    def update(self, ctx: StopContext):
        pass

    def check(self, ctx: StopContext) -> StopResult:
        if self._side == Side.LONG and ctx.low <= self._sl_price:
            return StopResult(True, self._sl_price, f"Fixed SL hit @ {self._sl_price:.4f} (-{self.sl_pct}%)")
        if self._side == Side.SHORT and ctx.high >= self._sl_price:
            return StopResult(True, self._sl_price, f"Fixed SL hit @ {self._sl_price:.4f} (+{self.sl_pct}%)")
        if self._tp_price is not None:
            if self._side == Side.LONG and ctx.high >= self._tp_price:
                return StopResult(True, self._tp_price, f"Fixed TP hit @ {self._tp_price:.4f} (+{self.tp_pct}%)")
            if self._side == Side.SHORT and ctx.low <= self._tp_price:
                return StopResult(True, self._tp_price, f"Fixed TP hit @ {self._tp_price:.4f} (-{self.tp_pct}%)")
        return StopResult()

    def reset(self):
        self._sl_price = 0
        self._tp_price = None
        self._side = Side.FLAT


class ATRStop(StopLoss):
    """Stop-loss at N × ATR from entry. Optional TP at M × ATR."""

    def __init__(self, atr_mult_sl: float = 2.0, atr_mult_tp: float | None = 3.0, atr_col: str = "atr"):
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_tp = atr_mult_tp
        self.atr_col = atr_col
        self._sl_price: float = 0
        self._tp_price: float | None = None
        self._side: Side = Side.FLAT

    @property
    def params(self):
        return dict(atr_mult_sl=self.atr_mult_sl, atr_mult_tp=self.atr_mult_tp)

    def on_entry(self, position: Position, ctx: StopContext):
        self._side = position.side
        entry = position.entry_price
        atr = ctx.bar_data.get(self.atr_col, entry * 0.02)
        if self._side == Side.LONG:
            self._sl_price = entry - self.atr_mult_sl * atr
            self._tp_price = entry + self.atr_mult_tp * atr if self.atr_mult_tp else None
        else:
            self._sl_price = entry + self.atr_mult_sl * atr
            self._tp_price = entry - self.atr_mult_tp * atr if self.atr_mult_tp else None

    def update(self, ctx: StopContext):
        pass

    def check(self, ctx: StopContext) -> StopResult:
        if self._side == Side.LONG and ctx.low <= self._sl_price:
            return StopResult(True, self._sl_price, f"ATR SL hit @ {self._sl_price:.4f} ({self.atr_mult_sl}×ATR)")
        if self._side == Side.SHORT and ctx.high >= self._sl_price:
            return StopResult(True, self._sl_price, f"ATR SL hit @ {self._sl_price:.4f} ({self.atr_mult_sl}×ATR)")
        if self._tp_price is not None:
            if self._side == Side.LONG and ctx.high >= self._tp_price:
                return StopResult(True, self._tp_price, f"ATR TP hit @ {self._tp_price:.4f} ({self.atr_mult_tp}×ATR)")
            if self._side == Side.SHORT and ctx.low <= self._tp_price:
                return StopResult(True, self._tp_price, f"ATR TP hit @ {self._tp_price:.4f} ({self.atr_mult_tp}×ATR)")
        return StopResult()

    def reset(self):
        self._sl_price = 0
        self._tp_price = None
        self._side = Side.FLAT


class TrailingStop(StopLoss):
    """Trailing stop-loss that follows price by a fixed percentage."""

    def __init__(self, trail_pct: float = 2.0, activation_pct: float = 0.0):
        self.trail_pct = trail_pct
        self.activation_pct = activation_pct
        self._sl_price: float = 0
        self._peak: float = 0
        self._trough: float = float("inf")
        self._entry_price: float = 0
        self._side: Side = Side.FLAT
        self._activated: bool = False

    @property
    def params(self):
        return dict(trail_pct=self.trail_pct, activation_pct=self.activation_pct)

    def on_entry(self, position: Position, ctx: StopContext):
        self._side = position.side
        self._entry_price = position.entry_price
        self._peak = ctx.high
        self._trough = ctx.low
        self._activated = self.activation_pct <= 0
        if self._side == Side.LONG:
            self._sl_price = self._peak * (1 - self.trail_pct / 100)
        else:
            self._sl_price = self._trough * (1 + self.trail_pct / 100)

    def update(self, ctx: StopContext):
        if not self._activated:
            if self._side == Side.LONG:
                gain_pct = (ctx.high - self._entry_price) / self._entry_price * 100
            else:
                gain_pct = (self._entry_price - ctx.low) / self._entry_price * 100
            if gain_pct >= self.activation_pct:
                self._activated = True
        if not self._activated:
            return
        if self._side == Side.LONG:
            if ctx.high > self._peak:
                self._peak = ctx.high
                self._sl_price = self._peak * (1 - self.trail_pct / 100)
        else:
            if ctx.low < self._trough:
                self._trough = ctx.low
                self._sl_price = self._trough * (1 + self.trail_pct / 100)

    def check(self, ctx: StopContext) -> StopResult:
        if not self._activated:
            return StopResult()
        if self._side == Side.LONG and ctx.low <= self._sl_price:
            return StopResult(True, self._sl_price, f"Trailing SL hit @ {self._sl_price:.4f} (peak={self._peak:.4f}, trail={self.trail_pct}%)")
        if self._side == Side.SHORT and ctx.high >= self._sl_price:
            return StopResult(True, self._sl_price, f"Trailing SL hit @ {self._sl_price:.4f} (trough={self._trough:.4f}, trail={self.trail_pct}%)")
        return StopResult()

    def reset(self):
        self._sl_price = 0
        self._peak = 0
        self._trough = float("inf")
        self._entry_price = 0
        self._activated = False
        self._side = Side.FLAT


class TrailingATRStop(StopLoss):
    """Trailing stop using ATR-based distance (Chandelier-style)."""

    def __init__(self, atr_mult: float = 2.5, atr_col: str = "atr"):
        self.atr_mult = atr_mult
        self.atr_col = atr_col
        self._sl_price: float = 0
        self._peak: float = 0
        self._trough: float = float("inf")
        self._side: Side = Side.FLAT

    @property
    def params(self):
        return dict(atr_mult=self.atr_mult)

    def on_entry(self, position: Position, ctx: StopContext):
        self._side = position.side
        atr = ctx.bar_data.get(self.atr_col, ctx.close * 0.02)
        self._peak = ctx.high
        self._trough = ctx.low
        if self._side == Side.LONG:
            self._sl_price = self._peak - self.atr_mult * atr
        else:
            self._sl_price = self._trough + self.atr_mult * atr

    def update(self, ctx: StopContext):
        atr = ctx.bar_data.get(self.atr_col, ctx.close * 0.02)
        if self._side == Side.LONG:
            if ctx.high > self._peak:
                self._peak = ctx.high
            new_sl = self._peak - self.atr_mult * atr
            self._sl_price = max(self._sl_price, new_sl)
        else:
            if ctx.low < self._trough:
                self._trough = ctx.low
            new_sl = self._trough + self.atr_mult * atr
            self._sl_price = min(self._sl_price, new_sl)

    def check(self, ctx: StopContext) -> StopResult:
        if self._side == Side.LONG and ctx.low <= self._sl_price:
            return StopResult(True, self._sl_price, f"Trailing ATR SL @ {self._sl_price:.4f} ({self.atr_mult}×ATR)")
        if self._side == Side.SHORT and ctx.high >= self._sl_price:
            return StopResult(True, self._sl_price, f"Trailing ATR SL @ {self._sl_price:.4f} ({self.atr_mult}×ATR)")
        return StopResult()

    def reset(self):
        self._sl_price = 0
        self._peak = 0
        self._trough = float("inf")
        self._side = Side.FLAT


class BreakevenStop(StopLoss):
    """Move stop to breakeven after price reaches `activation_pct` profit."""

    def __init__(self, activation_pct: float = 1.0, offset_pct: float = 0.1):
        self.activation_pct = activation_pct
        self.offset_pct = offset_pct
        self._sl_price: float = 0
        self._entry_price: float = 0
        self._side: Side = Side.FLAT
        self._activated: bool = False

    @property
    def params(self):
        return dict(activation_pct=self.activation_pct, offset_pct=self.offset_pct)

    def on_entry(self, position: Position, ctx: StopContext):
        self._side = position.side
        self._entry_price = position.entry_price
        self._activated = False
        self._sl_price = 0

    def update(self, ctx: StopContext):
        if self._activated:
            return
        if self._side == Side.LONG:
            gain = (ctx.high - self._entry_price) / self._entry_price * 100
            if gain >= self.activation_pct:
                self._activated = True
                self._sl_price = self._entry_price * (1 + self.offset_pct / 100)
        else:
            gain = (self._entry_price - ctx.low) / self._entry_price * 100
            if gain >= self.activation_pct:
                self._activated = True
                self._sl_price = self._entry_price * (1 - self.offset_pct / 100)

    def check(self, ctx: StopContext) -> StopResult:
        if not self._activated:
            return StopResult()
        if self._side == Side.LONG and ctx.low <= self._sl_price:
            return StopResult(True, self._sl_price, f"Breakeven SL hit @ {self._sl_price:.4f}")
        if self._side == Side.SHORT and ctx.high >= self._sl_price:
            return StopResult(True, self._sl_price, f"Breakeven SL hit @ {self._sl_price:.4f}")
        return StopResult()

    def reset(self):
        self._sl_price = 0
        self._activated = False
        self._side = Side.FLAT


class TimeStop(StopLoss):
    """Exit after a maximum number of bars held."""

    def __init__(self, max_bars: int = 48):
        self.max_bars = max_bars
        self._bars_held: int = 0
        self._side: Side = Side.FLAT

    @property
    def params(self):
        return dict(max_bars=self.max_bars)

    def on_entry(self, position: Position, ctx: StopContext):
        self._side = position.side
        self._bars_held = 0

    def update(self, ctx: StopContext):
        self._bars_held += 1

    def check(self, ctx: StopContext) -> StopResult:
        if self._bars_held >= self.max_bars:
            return StopResult(True, ctx.close, f"Time stop after {self._bars_held} bars")
        return StopResult()

    def reset(self):
        self._bars_held = 0
        self._side = Side.FLAT


class RiskRewardStop(StopLoss):
    """Set SL at a fixed distance, TP at a risk:reward ratio."""

    def __init__(self, sl_pct: float = 1.5, rr_ratio: float = 2.0):
        self.sl_pct = sl_pct
        self.rr_ratio = rr_ratio
        self._sl_price: float = 0
        self._tp_price: float = 0
        self._side: Side = Side.FLAT

    @property
    def params(self):
        return dict(sl_pct=self.sl_pct, rr_ratio=self.rr_ratio)

    def on_entry(self, position: Position, ctx: StopContext):
        self._side = position.side
        entry = position.entry_price
        tp_pct = self.sl_pct * self.rr_ratio
        if self._side == Side.LONG:
            self._sl_price = entry * (1 - self.sl_pct / 100)
            self._tp_price = entry * (1 + tp_pct / 100)
        else:
            self._sl_price = entry * (1 + self.sl_pct / 100)
            self._tp_price = entry * (1 - tp_pct / 100)

    def update(self, ctx: StopContext):
        pass

    def check(self, ctx: StopContext) -> StopResult:
        if self._side == Side.LONG:
            if ctx.low <= self._sl_price:
                return StopResult(True, self._sl_price, f"R:R SL hit @ {self._sl_price:.4f}")
            if ctx.high >= self._tp_price:
                return StopResult(True, self._tp_price, f"R:R TP hit @ {self._tp_price:.4f} ({self.rr_ratio}R)")
        else:
            if ctx.high >= self._sl_price:
                return StopResult(True, self._sl_price, f"R:R SL hit @ {self._sl_price:.4f}")
            if ctx.low <= self._tp_price:
                return StopResult(True, self._tp_price, f"R:R TP hit @ {self._tp_price:.4f} ({self.rr_ratio}R)")
        return StopResult()

    def reset(self):
        self._sl_price = 0
        self._tp_price = 0
        self._side = Side.FLAT


class CompositeStopLoss(StopLoss):
    """Combine multiple stop-loss strategies. First to trigger wins."""

    def __init__(self, stops: list[StopLoss] | None = None):
        self.stops = stops or [FixedPercentStop()]

    @property
    def params(self):
        return {s.__class__.__name__: s.params for s in self.stops}

    def on_entry(self, position: Position, ctx: StopContext):
        for s in self.stops:
            s.on_entry(position, ctx)

    def update(self, ctx: StopContext):
        for s in self.stops:
            s.update(ctx)

    def check(self, ctx: StopContext) -> StopResult:
        for s in self.stops:
            result = s.check(ctx)
            if result.triggered:
                result.meta["triggered_by"] = s.__class__.__name__
                return result
        return StopResult()

    def reset(self):
        for s in self.stops:
            s.reset()

    def deep_copy(self) -> "CompositeStopLoss":
        return copy.deepcopy(self)


class EmbeddedStop(StopLoss):
    """
    Use the stop_loss/take_profit levels embedded in an Allocation.
    Default when no StopLoss is passed to the engine.
    """

    def __init__(self):
        self._sl: float | None = None
        self._tp: float | None = None
        self._side: Side = Side.FLAT

    @property
    def params(self):
        return {}

    def on_entry(self, position: Position, ctx: StopContext):
        self._side = position.side

    def update(self, ctx: StopContext):
        pass

    def check(self, ctx: StopContext) -> StopResult:
        return StopResult()

    def set_levels(self, sl: float | None, tp: float | None):
        self._sl = sl
        self._tp = tp

    def check_with_levels(self, ctx: StopContext) -> StopResult:
        if self._sl is not None:
            if self._side == Side.LONG and ctx.low <= self._sl:
                return StopResult(True, self._sl, f"SL hit @ {self._sl:.4f}")
            if self._side == Side.SHORT and ctx.high >= self._sl:
                return StopResult(True, self._sl, f"SL hit @ {self._sl:.4f}")
        if self._tp is not None:
            if self._side == Side.LONG and ctx.high >= self._tp:
                return StopResult(True, self._tp, f"TP hit @ {self._tp:.4f}")
            if self._side == Side.SHORT and ctx.low <= self._tp:
                return StopResult(True, self._tp, f"TP hit @ {self._tp:.4f}")
        return StopResult()

    def reset(self):
        self._sl = None
        self._tp = None
        self._side = Side.FLAT


class NopStopLoss(StopLoss):
    """
    No-op stop-loss: never triggers.

    Pass ``stop_loss=NopStopLoss()`` to :class:`~backtester.engine.Backtester`
    to satisfy one of the two conditions required for the vectorised fast path.
    Exits happen solely when the signal changes side.
    """

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def on_entry(self, position: Position, ctx: StopContext) -> None:
        pass

    def update(self, ctx: StopContext) -> None:
        pass

    def check(self, ctx: StopContext) -> StopResult:
        return StopResult(triggered=False)

    def reset(self) -> None:
        pass


def default_stop_loss() -> StopLoss:
    return EmbeddedStop()
