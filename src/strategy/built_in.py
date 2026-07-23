"""
strategy/built_in.py — Built-in strategy implementations.

Single-asset:
  SingleAssetStrategy        — convenience base for single-symbol strategies
  CompositeStrategy          — combine multiple SingleAssetStrategy instances with
                               weighted voting (registered as "composite")
  TrendFollowingStrategy     — EMA crossover + ADX filter (registered as "trend_following")

Multi-asset:
  PerAssetStrategy               — run a different SingleAssetStrategy per symbol
  MeanReversionBasketStrategy    — z-score mean reversion with RSI filter
  CrossSectionalMomentumStrategy — rank assets by N-bar return, long top / short bottom
"""

from __future__ import annotations

import abc
import copy

import numpy as np
import pandas as pd

from core.models import Side, Allocation, OrderBookSnapshot
from strategy.indicators import ema, rsi, atr, adx

from .base import (
    Strategy,
    StrategyContext,
    PortfolioTarget,
    register_strategy,
)
from core.universe import Universe


# ── SingleAssetStrategy — convenience base for single-symbol strategies ──────
class SingleAssetStrategy(Strategy):
    """
    Convenience base for single-asset strategies.

    Implement ``bar(data, idx) -> Allocation`` and optionally override
    ``setup_data(data, l2)`` for indicator pre-computation.
    ``setup``, ``generate``, and ``generate_all`` are auto-wired.
    """

    def __init__(self, symbol: str, **kw):
        super().__init__(**kw)
        self.symbol = symbol

    def setup_data(
        self, _data: pd.DataFrame, _l2: list[OrderBookSnapshot] | None = None
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

    def generate_all(self, universe: Universe):
        data = universe.ohlcv(self.symbol)
        n = len(data)
        sides = np.zeros(n, dtype=np.int8)
        weights = np.zeros(n, dtype=np.float64)
        confidences = np.zeros(n, dtype=np.float64)
        reasons: list[str] = []
        metas: list[dict] = []
        for i in range(n):
            a = self.bar(data, i)
            sides[i] = np.int8(a.side.value)
            weights[i] = a.weight
            confidences[i] = a.confidence
            reasons.append(a.reason)
            metas.append(dict(a.meta))
        return (
            {self.symbol: sides},
            {self.symbol: weights},
            {self.symbol: reasons},
            {self.symbol: metas},
            {self.symbol: confidences},
        )

    @property
    def params(self) -> dict:
        return {}


# ── Composite single-asset strategy ─────────────────────────────────────────
class CompositeStrategy(SingleAssetStrategy):
    """
    Combine multiple single-asset strategies with weighted voting.

    Each child strategy's ``bar()`` is called; their ``side * confidence``
    scores are summed with per-strategy weights.  If the aggregate exceeds
    ``threshold`` in either direction, the position is entered.

    Usage:
        comp = CompositeStrategy(
            symbol="BTC",
            strategies=[ema_cross, rsi_strat],
            weights=[0.6, 0.4],
        )
    """

    def __init__(
        self,
        symbol: str,
        strategies: list[SingleAssetStrategy] | None = None,
        weights: list[float] | None = None,
        threshold: float = 0.5,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.strategies = strategies or []
        self.weights = weights or [1.0 / len(self.strategies)] * len(self.strategies)
        self.threshold = threshold

    @property
    def params(self) -> dict:
        return {
            "threshold": self.threshold,
            "weights": self.weights,
            **{f"sub_{i}": s.params for i, s in enumerate(self.strategies)},
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        for s in self.strategies:
            s.setup_data(data, l2)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        score = 0.0
        reasons = []
        for s, w in zip(self.strategies, self.weights):
            a = s.bar(data, idx)
            score += a.side.value * a.confidence * w
            if a.reason:
                reasons.append(f"[{s.__class__.__name__}] {a.reason}")

        if score > self.threshold:
            return Allocation(
                side=Side.LONG,
                weight=min(abs(score), 1.0),
                confidence=abs(score),
                reason=" | ".join(reasons),
            )
        elif score < -self.threshold:
            return Allocation(
                side=Side.SHORT,
                weight=min(abs(score), 1.0),
                confidence=abs(score),
                reason=" | ".join(reasons),
            )
        return Allocation(reason=f"Composite score={score:.3f} below threshold")


# ── Multi-asset per-symbol strategy ─────────────────────────────────────────
class PerAssetStrategy(Strategy):
    """
    Run a different SingleAssetStrategy on each asset independently.

    Usage:
        strategy = PerAssetStrategy(
            strategies={"ETH": eth_strategy, "BTC": btc_strategy},
        )
    """

    def __init__(self, strategies: dict[str, SingleAssetStrategy], **kwargs):
        super().__init__(**kwargs)
        self.strategies = strategies

    @classmethod
    def from_template(
        cls, template: SingleAssetStrategy, symbols: list[str], **kwargs
    ) -> "PerAssetStrategy":
        """Fan a single-asset strategy out across a universe.

        Deep-copies ``template`` once per symbol, rebinding each copy to its own
        symbol, so a strategy written for one asset trades every asset in the
        universe independently. Used by the backtester to auto-wrap a
        SingleAssetStrategy that is run on a multi-asset universe.
        """
        strategies: dict[str, SingleAssetStrategy] = {}
        for sym in symbols:
            s = copy.deepcopy(template)
            s.symbol = sym
            strategies[sym] = s
        return cls(strategies=strategies, **kwargs)

    @property
    def params(self) -> dict:
        return {sym: s.params for sym, s in self.strategies.items()}

    def setup(self, universe: Universe):
        for sym, s in self.strategies.items():
            data = universe.ohlcv(sym)
            l2 = universe.l2(sym)
            s.setup_data(data, l2)

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget(timestamp=ctx.timestamp)
        n_assets = len(self.strategies)
        for sym, s in self.strategies.items():
            data = ctx.universe.ohlcv(sym)
            # ctx.bar_idx is a position on the (possibly shorter) common index; map it
            # back to this symbol's own row by timestamp so assets with ragged histories
            # stay bar-aligned. Falls back to bar_idx when timestamps line up 1:1.
            idx = ctx.bar_idx
            if ctx.timestamp is not None:
                loc = data.index.get_indexer([ctx.timestamp])[0]
                if loc >= 0:
                    idx = loc
            alloc = s.bar(data, idx)
            target[sym] = Allocation(
                side=alloc.side,
                weight=alloc.weight / max(n_assets, 1),
                confidence=alloc.confidence,
                reason=f"[{sym}] {alloc.reason}",
                meta=alloc.meta,
            )
        return target

    def generate_all(self, universe):
        n_assets = len(self.strategies)
        # The engine drives the vectorised path over the universe's common (intersection)
        # index, so every symbol's arrays must share that length. Child single-asset
        # strategies emit arrays aligned to their own (possibly longer) bar count, so
        # realign each onto the common index by timestamp when assets are ragged.
        multi = len(universe.symbols) > 1
        index = universe.common_index() if multi else None
        if index is not None and len(index) == 0:
            index = None

        sides_all = {}
        weights_all = {}
        reasons_all = {}
        metas_all = {}
        confidences_all = {}
        for sym, s in self.strategies.items():
            batch = s.generate_all(universe)
            if batch is None:
                return None
            sides   = batch[0][sym]
            weights = batch[1][sym] / max(n_assets, 1)
            reasons = batch[2][sym]
            metas   = batch[3][sym]
            confs   = batch[4][sym] if len(batch) > 4 else np.zeros(len(sides))

            if index is not None and len(sides) != len(index):
                locs  = universe.ohlcv(sym).index.get_indexer(index)
                valid = locs >= 0
                g     = np.where(valid, locs, 0)
                sides   = np.where(valid, sides[g], 0).astype(np.int8)
                weights = np.where(valid, weights[g], 0.0)
                confs   = np.where(valid, confs[g], 0.0)
                reasons = [reasons[gi] if v else "" for gi, v in zip(g, valid)]
                metas   = [metas[gi] if v else {} for gi, v in zip(g, valid)]

            sides_all[sym]       = sides
            weights_all[sym]     = weights
            reasons_all[sym]     = [f"[{sym}] {r}" for r in reasons]
            metas_all[sym]       = metas
            confidences_all[sym] = confs
        return sides_all, weights_all, reasons_all, metas_all, confidences_all


# ═══════════════════════════════════════════════════════════════════════════
#  Built-in multi-asset strategies
# ═══════════════════════════════════════════════════════════════════════════

@register_strategy("mean_reversion_basket")
class MeanReversionBasketStrategy(Strategy):
    """
    Mean-revert each asset toward its rolling mean, weighted by
    relative deviation.  Uses RSI as a filter.

    When an asset's z-score is extreme AND RSI confirms, take a
    contrarian position.  Portfolio-level exposure is capped.
    """

    def __init__(
        self,
        lookback: int = 40,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        max_total_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.max_total_weight = max_total_weight
        self._indicators: dict[str, dict] = {}

    @property
    def params(self) -> dict:
        return {
            "lookback": self.lookback,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "rsi_period": self.rsi_period,
        }

    def setup(self, universe: Universe):
        for sym in universe.symbols:
            close = universe.close(sym)
            mu = close.rolling(self.lookback).mean()
            sigma = close.rolling(self.lookback).std()
            zscore = (close - mu) / sigma.replace(0, np.nan)
            rsi_vals = rsi(close, self.rsi_period)
            self._indicators[sym] = {"zscore": zscore, "rsi": rsi_vals}

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget(timestamp=ctx.timestamp)
        i = ctx.bar_idx

        for sym in ctx.universe.symbols:
            if sym not in self._indicators:
                continue
            ind = self._indicators[sym]
            if i >= len(ind["zscore"]):
                continue

            z = ind["zscore"].iat[i]
            r = ind["rsi"].iat[i]

            if np.isnan(z) or np.isnan(r):
                continue

            if z < -self.entry_z and r < self.rsi_oversold:
                target[sym] = Allocation(
                    side=Side.LONG,
                    weight=min(abs(z) / 4, 0.3),
                    confidence=min(abs(z) / 3, 1.0),
                    reason=f"MR z={z:.2f} rsi={r:.0f}",
                )
            elif z > self.entry_z and r > self.rsi_overbought:
                target[sym] = Allocation(
                    side=Side.SHORT,
                    weight=min(abs(z) / 4, 0.3),
                    confidence=min(abs(z) / 3, 1.0),
                    reason=f"MR z={z:.2f} rsi={r:.0f}",
                )
            elif abs(z) < self.exit_z:
                target[sym] = Allocation(reason=f"MR exit z={z:.2f}")

        target.normalize(self.max_total_weight)
        return target

    def generate_all(self, universe):
        return self._batch_generate(universe)


# ── Trend Following (single-asset) ──────────────────────────────────────────
@register_strategy("trend_following")
class TrendFollowingStrategy(SingleAssetStrategy):
    """
    EMA crossover with ADX trend-strength filter.

    Entry:
      - LONG  when fast EMA > slow EMA and ADX > adx_threshold
      - SHORT when fast EMA < slow EMA and ADX > adx_threshold
    No position when ADX is below threshold (ranging/choppy market).
    Confidence scales linearly with ADX strength above the threshold.
    """

    def __init__(
        self,
        symbol: str,
        fast: int = 20,
        slow: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.fast = fast
        self.slow = slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period

    @property
    def params(self) -> dict:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "adx_period": self.adx_period,
            "adx_threshold": self.adx_threshold,
            "atr_period": self.atr_period,
        }

    def setup_data(self, data: pd.DataFrame, _l2=None):
        close = data["close"]
        data["_ema_fast"] = ema(close, self.fast)
        data["_ema_slow"] = ema(close, self.slow)
        data["_atr"]      = atr(data["high"], data["low"], close, self.atr_period)
        data["_adx"]      = adx(data["high"], data["low"], close, self.adx_period)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.slow:
            return Allocation(reason="warmup")

        ef      = data["_ema_fast"].iat[idx]
        es      = data["_ema_slow"].iat[idx]
        adx_val = data["_adx"].iat[idx]

        if np.isnan(ef) or np.isnan(es) or np.isnan(adx_val):
            return Allocation(reason="nan")

        if adx_val < self.adx_threshold:
            return Allocation(reason=f"ADX={adx_val:.1f} below threshold (no trend)")

        # confidence grows as ADX moves away from the threshold; cap at 1.0
        confidence = min((adx_val - self.adx_threshold) / 50.0, 1.0)

        if ef > es:
            return Allocation(
                side=Side.LONG,
                weight=confidence,
                confidence=confidence,
                reason=f"Trend long EMA{self.fast}>{self.slow} ADX={adx_val:.1f}",
            )
        return Allocation(
            side=Side.SHORT,
            weight=confidence,
            confidence=confidence,
            reason=f"Trend short EMA{self.fast}<{self.slow} ADX={adx_val:.1f}",
        )


# ── Cross-Sectional Momentum (multi-asset) ───────────────────────────────────
@register_strategy("cross_sectional_momentum")
class CrossSectionalMomentumStrategy(Strategy):
    """
    Rank all assets by N-bar return each bar.
    Go long the top ``long_frac`` and short the bottom ``short_frac``.

    Weight within each leg is equal; total long + short exposure is capped at
    ``max_total_weight`` (split evenly between the two legs).
    """

    def __init__(
        self,
        lookback: int = 60,
        long_frac: float = 0.3,
        short_frac: float = 0.3,
        max_total_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.long_frac = long_frac
        self.short_frac = short_frac
        self.max_total_weight = max_total_weight
        self._momentum: dict[str, pd.Series] = {}

    @property
    def params(self) -> dict:
        return {
            "lookback": self.lookback,
            "long_frac": self.long_frac,
            "short_frac": self.short_frac,
        }

    def setup(self, universe: Universe):
        for sym in universe.symbols:
            self._momentum[sym] = universe.close(sym).pct_change(self.lookback)

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget(timestamp=ctx.timestamp)
        i = ctx.bar_idx

        scores: dict[str, float] = {}
        for sym, mom in self._momentum.items():
            if i < len(mom):
                val = mom.iat[i]
                if not np.isnan(val):
                    scores[sym] = val

        if not scores:
            return target

        ranked = sorted(scores, key=scores.__getitem__)
        n = len(ranked)
        n_long  = max(1, round(n * self.long_frac))
        n_short = max(1, round(n * self.short_frac))
        longs  = ranked[-n_long:]
        shorts = ranked[:n_short]

        w_long  = self.max_total_weight / 2 / n_long
        w_short = self.max_total_weight / 2 / n_short

        for sym in longs:
            target[sym] = Allocation(
                side=Side.LONG,
                weight=w_long,
                confidence=min(abs(scores[sym]), 1.0),
                reason=f"XSMom top mom={scores[sym]:.2%}",
            )
        for sym in shorts:
            target[sym] = Allocation(
                side=Side.SHORT,
                weight=w_short,
                confidence=min(abs(scores[sym]), 1.0),
                reason=f"XSMom bot mom={scores[sym]:.2%}",
            )
        return target

    def generate_all(self, universe):
        return self._batch_generate(universe)

