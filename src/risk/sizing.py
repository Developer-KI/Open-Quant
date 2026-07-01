"""
risk/sizing.py — Modular position sizing framework.

Moved here from strategy/sizing.py so the risk layer is independent of the
strategy layer. strategy/sizing.py now re-exports from here for backward
compatibility.

Every Sizer takes a SizingContext and returns position size in base-asset units.
Sizers can be composed and stress-tested independently of any strategy.

Dependency: core/ only — no imports from strategy/ or execution/.
"""

from __future__ import annotations
from collections.abc import Callable

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.models import BacktestConfig, Position, Side, OrderBookSnapshot, Allocation


# ── Sizing context ────────────────────────────────────────────────────────────


@dataclass
class SizingContext:
    """Everything a sizer might need to decide position size."""

    equity: float
    price: float
    allocation: Allocation
    config: BacktestConfig
    position: Position
    data: pd.DataFrame | None = None
    bar_idx: int = 0
    trade_history: list | None = None
    l2: OrderBookSnapshot | None = None
    bar_data: dict[str, float] = field(default_factory=dict)


# ── Abstract base ─────────────────────────────────────────────────────────────


class Sizer(abc.ABC):
    """
    Base class for position sizers.

    Return value is position size in base-asset units (e.g. BTC, not USD).
    The engine caps the result at config.max_position_pct after your call.
    """

    @abc.abstractmethod
    def compute(self, ctx: SizingContext) -> float: ...

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, Any]: ...

    def set_params(self, new: dict[str, Any]):
        for k, v in new.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @property
    def vectorizable(self) -> bool:
        """
        True when size can be computed from price alone — no dependency on
        running equity.  Returning True is the second condition required for
        the vectorised backtest fast path.
        """
        return False

    def compute_vectorized(
        self,
        prices: np.ndarray,
        weights: np.ndarray,
        config: BacktestConfig,
    ) -> np.ndarray:
        """
        Compute position sizes for an array of entry bars without equity state.
        Only called when ``vectorizable`` is True.
        """
        del prices, weights, config
        raise NotImplementedError(f"{type(self).__name__} is not vectorizable")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"


# ── Concrete sizers ───────────────────────────────────────────────────────────


class FixedFractionalSizer(Sizer):
    """
    Risk a fixed fraction of equity per trade.

    If a stop-loss is provided on the signal, sizes so that the SL distance
    equals `risk_frac` of equity.  Otherwise falls back to notional sizing.
    """

    def __init__(self, risk_frac: float = 0.02):
        self.risk_frac = risk_frac

    @property
    def params(self):
        return dict(risk_frac=self.risk_frac)

    def compute(self, ctx: SizingContext) -> float:
        dollar_risk = ctx.equity * self.risk_frac * ctx.allocation.weight

        if ctx.allocation.stop_loss is not None and ctx.allocation.stop_loss > 0:
            sl_dist = abs(ctx.price - ctx.allocation.stop_loss)
            if sl_dist > 0:
                return (dollar_risk / sl_dist) * ctx.config.leverage

        notional = dollar_risk * ctx.config.leverage
        return notional / ctx.price if ctx.price > 0 else 0


class FixedNotionalSizer(Sizer):
    """Allocate a fixed dollar notional per trade (or % of equity)."""

    def __init__(self, notional: float | None = None, equity_pct: float = 0.10):
        self.notional = notional
        self.equity_pct = equity_pct

    @property
    def params(self):
        return dict(notional=self.notional, equity_pct=self.equity_pct)

    def compute(self, ctx: SizingContext) -> float:
        if self.notional is not None:
            n = self.notional * ctx.allocation.weight
        else:
            n = ctx.equity * self.equity_pct * ctx.allocation.weight
        n *= ctx.config.leverage
        return n / ctx.price if ctx.price > 0 else 0

    @property
    def vectorizable(self) -> bool:
        return self.notional is not None

    def compute_vectorized(
        self,
        prices: np.ndarray,
        weights: np.ndarray,
        config: BacktestConfig,
    ) -> np.ndarray:
        notional = self.notional * weights * config.leverage
        return np.where(prices > 0, notional / prices, 0.0)


class VolatilityTargetSizer(Sizer):
    """
    Target a specific annualised portfolio volatility.

    size = (target_vol * equity) / (realised_vol * price * √bars_per_year)
    """

    def __init__(
        self, target_vol: float = 0.15, lookback: int = 20, bars_per_year: int = 8760
    ):
        self.target_vol = target_vol
        self.lookback = lookback
        self.bars_per_year = bars_per_year

    @property
    def params(self):
        return dict(
            target_vol=self.target_vol,
            lookback=self.lookback,
            bars_per_year=self.bars_per_year,
        )

    def compute(self, ctx: SizingContext) -> float:
        if ctx.data is None or ctx.bar_idx < self.lookback:
            return ctx.equity * 0.01 * ctx.config.leverage / ctx.price

        closes = ctx.data["close"].iloc[
            max(0, ctx.bar_idx - self.lookback) : ctx.bar_idx + 1
        ]
        rets = closes.pct_change().dropna()
        if len(rets) < 2:
            return ctx.equity * 0.01 * ctx.config.leverage / ctx.price

        bar_vol = rets.std()
        ann_vol = bar_vol * np.sqrt(self.bars_per_year)

        if ann_vol < 1e-12:
            return 0

        dollar_vol_budget = ctx.equity * self.target_vol
        per_unit_vol = ctx.price * ann_vol
        size = (
            (dollar_vol_budget / per_unit_vol)
            * ctx.allocation.weight
            * ctx.config.leverage
        )
        return max(size, 0)


class KellySizer(Sizer):
    """
    Kelly criterion sizing based on historical win rate and payoff ratio.

    Uses a rolling window of recent trades, or falls back to signal confidence.
    `kelly_frac` controls what fraction of full Kelly to use (0.5 = half-Kelly).
    """

    def __init__(
        self,
        kelly_frac: float = 0.5,
        min_trades: int = 20,
        lookback_trades: int = 50,
        floor: float = 0.005,
        cap: float = 0.15,
    ):
        self.kelly_frac = kelly_frac
        self.min_trades = min_trades
        self.lookback_trades = lookback_trades
        self.floor = floor
        self.cap = cap

    @property
    def params(self):
        return dict(
            kelly_frac=self.kelly_frac,
            min_trades=self.min_trades,
            lookback_trades=self.lookback_trades,
            floor=self.floor,
            cap=self.cap,
        )

    def compute(self, ctx: SizingContext) -> float:
        recent = (ctx.trade_history or [])[-self.lookback_trades :]

        if len(recent) >= self.min_trades:
            pnls = [t.pnl for t in recent]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            win_rate = len(wins) / len(pnls)
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 1

            if avg_loss > 0:
                payoff = avg_win / avg_loss
                kelly_pct = win_rate - (1 - win_rate) / payoff
            else:
                kelly_pct = win_rate
        else:
            kelly_pct = ctx.allocation.confidence * 0.5

        kelly_pct = max(kelly_pct, 0) * self.kelly_frac
        alloc_pct = np.clip(kelly_pct, self.floor, self.cap)

        notional = (
            ctx.equity * alloc_pct * ctx.allocation.weight * ctx.config.leverage
        )
        return notional / ctx.price if ctx.price > 0 else 0


class AntiMartingaleSizer(Sizer):
    """
    Increase position size after wins, decrease after losses.
    """

    def __init__(
        self,
        base_risk_frac: float = 0.02,
        streak_multiplier: float = 0.5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 0.25,
    ):
        self.base_risk_frac = base_risk_frac
        self.streak_multiplier = streak_multiplier
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier

    @property
    def params(self):
        return dict(
            base_risk_frac=self.base_risk_frac,
            streak_multiplier=self.streak_multiplier,
            max_multiplier=self.max_multiplier,
            min_multiplier=self.min_multiplier,
        )

    def compute(self, ctx: SizingContext) -> float:
        trades = ctx.trade_history or []

        streak = 0
        is_winning = True
        for t in reversed(trades):
            if is_winning and t.pnl > 0:
                streak += 1
            elif not is_winning and t.pnl <= 0:
                streak -= 1
            else:
                break

        if streak > 0:
            mult = 1.0 + self.streak_multiplier * streak
        elif streak < 0:
            mult = 1.0 / (1.0 + self.streak_multiplier * abs(streak))
        else:
            mult = 1.0

        mult = np.clip(mult, self.min_multiplier, self.max_multiplier)

        notional = (
            ctx.equity
            * self.base_risk_frac
            * mult
            * ctx.allocation.weight
            * ctx.config.leverage
        )
        return notional / ctx.price if ctx.price > 0 else 0


class DrawdownScalingSizer(Sizer):
    """
    Reduce size proportionally during drawdowns.

    At 0% drawdown → full size.
    At `max_dd_threshold` drawdown → `min_scale` of full size.
    """

    def __init__(
        self,
        base_risk_frac: float = 0.02,
        max_dd_threshold: float = 0.15,
        min_scale: float = 0.25,
    ):
        self.base_risk_frac = base_risk_frac
        self.max_dd_threshold = max_dd_threshold
        self.min_scale = min_scale
        self._peak_equity: float = 0.0

    @property
    def params(self):
        return dict(
            base_risk_frac=self.base_risk_frac,
            max_dd_threshold=self.max_dd_threshold,
            min_scale=self.min_scale,
        )

    def compute(self, ctx: SizingContext) -> float:
        self._peak_equity = max(self._peak_equity, ctx.equity)
        current_dd = (
            (self._peak_equity - ctx.equity) / self._peak_equity
            if self._peak_equity > 0
            else 0
        )

        if current_dd >= self.max_dd_threshold:
            scale = self.min_scale
        else:
            scale = 1.0 - (1.0 - self.min_scale) * (current_dd / self.max_dd_threshold)

        notional = (
            ctx.equity
            * self.base_risk_frac
            * scale
            * ctx.allocation.weight
            * ctx.config.leverage
        )
        return notional / ctx.price if ctx.price > 0 else 0


class L2LiquiditySizer(Sizer):
    """
    Size based on available L2 book depth.

    Never take more than `max_participation` of visible liquidity.
    """

    def __init__(
        self,
        base_risk_frac: float = 0.02,
        max_participation: float = 0.10,
        depth_pct: float = 0.005,
    ):
        self.base_risk_frac = base_risk_frac
        self.max_participation = max_participation
        self.depth_pct = depth_pct

    @property
    def params(self):
        return dict(
            base_risk_frac=self.base_risk_frac,
            max_participation=self.max_participation,
            depth_pct=self.depth_pct,
        )

    def compute(self, ctx: SizingContext) -> float:
        base_notional = (
            ctx.equity
            * self.base_risk_frac
            * ctx.allocation.weight
            * ctx.config.leverage
        )
        base_size = base_notional / ctx.price if ctx.price > 0 else 0

        if ctx.l2 is not None:
            depth = ctx.l2.depth_at(self.depth_pct)
            if ctx.allocation.side == Side.LONG:
                available = depth.get("ask_depth", float("inf"))
            else:
                available = depth.get("bid_depth", float("inf"))

            max_size = available * self.max_participation
            base_size = min(base_size, max_size)

        return base_size


class CompositeSizer(Sizer):
    """
    Combine multiple sizers.  Modes: "min", "max", "avg".
    """

    def __init__(
        self,
        sizers: list[Sizer] | None = None,
        weights: list[float] | None = None,
        mode: str = "min",
    ):
        self.sizers = sizers or [FixedFractionalSizer()]
        self.weights = weights or [1.0 / len(self.sizers)] * len(self.sizers)
        self.mode = mode

    @property
    def params(self):
        return {
            "mode": self.mode,
            "weights": self.weights,
            **{
                f"sub_{i}_{s.__class__.__name__}": s.params
                for i, s in enumerate(self.sizers)
            },
        }

    def compute(self, ctx: SizingContext) -> float:
        sizes = [s.compute(ctx) for s in self.sizers]

        if self.mode == "min":
            return min(sizes) if sizes else 0
        elif self.mode == "max":
            return max(sizes) if sizes else 0
        elif self.mode == "avg":
            return sum(s * w for s, w in zip(sizes, self.weights))
        else:
            return min(sizes) if sizes else 0


def default_sizer() -> Sizer:
    return FixedFractionalSizer(risk_frac=0.02)
