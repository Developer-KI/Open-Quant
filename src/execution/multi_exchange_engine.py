"""
execution/multi_exchange_engine.py — Multi-exchange live trading engine.

Extracted from live_engine.py. Runs strategies across N exchanges with a
shared MultiExchangePortfolio for cross-exchange hedging and net-exposure
tracking.

Three modes:
  Mode A — cross_strategy: one CrossExchangeStrategy sees all exchanges
  Mode B — per_exchange_strategies: each exchange runs its own Strategy
  Mode C — single strategy replicated (legacy)
"""

from __future__ import annotations

import copy
import csv
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.models import (
    BacktestConfig,
    Position,
    Side,
    Trade,
    LiveConfig,
    ExchangeCredentials,
)
from risk.sizing import Sizer, SizingContext, default_sizer
from risk.stops import StopLoss, StopContext, default_stop_loss
from risk.limits import DailyLimitState, check_daily_loss_limit

from strategy.base import (
    Strategy,
    StrategyContext,
    PortfolioTarget,
    CrossExchangeStrategy,
    CrossExchangeContext,
    MultiExchangeTarget,
)
from strategy.universe import Universe
from strategy.overlay import PortfolioOverlay

from .base_executor_feed import BaseExecutor, BaseFeed, BaseBarBuilder, FillResult, MultiExchangePortfolio
from .factory import create_executor, create_feed, create_bar_builder
from .live_state import _AssetLiveState, LiveState
from .live_engine import _ManualKillSwitch, _sizer_config_shim

logger = logging.getLogger(__name__)


class MultiExchangeEngine:
    """
    Run strategies across multiple exchanges with a shared portfolio.

    Mode A — Cross-exchange strategy (funding arb, stat arb, hedging):
        One CrossExchangeStrategy sees all exchanges and generates a
        MultiExchangeTarget that explicitly routes each leg.

    Mode B — Per-exchange strategies (independent):
        Each exchange runs its own Strategy independently. A shared
        MultiExchangePortfolio provides cross-exchange awareness, and
        an optional PortfolioOverlay can enforce portfolio-level limits.
    """

    def __init__(
        self,
        cross_strategy: CrossExchangeStrategy | None = None,
        per_exchange_strategies: dict[str, Strategy] | None = None,
        overlay: PortfolioOverlay | None = None,
        config: LiveConfig | None = None,
        sizer: Sizer | dict[str, Sizer] | None = None,
        stop_loss: StopLoss | dict[str, StopLoss] | None = None,
    ):
        modes = sum(x is not None for x in [cross_strategy, per_exchange_strategies])
        if modes == 0:
            raise ValueError("Provide one of: cross_strategy= or per_exchange_strategies=")
        if modes > 1:
            raise ValueError("Provide only one of: cross_strategy= or per_exchange_strategies=")

        self.config = config or LiveConfig()
        self._sizer_spec = sizer
        self._stop_loss_spec = stop_loss
        self.overlay = overlay

        self._creds = self.config.get_credentials()
        if len(self._creds) < 2:
            raise ValueError("MultiExchangeEngine requires >= 2 exchanges in config.exchanges")

        self._symbols = self.config.active_symbols
        self._exchange_names = [c.exchange for c in self._creds]

        if cross_strategy is not None:
            self._mode = "cross"
            self.cross_strategy = cross_strategy
            self._per_exchange_strategies: dict[str, Strategy] = {}
        else:
            self._mode = "per_exchange"
            self.cross_strategy = None
            self._per_exchange_strategies = per_exchange_strategies

        self._executors: dict[str, BaseExecutor] = {}
        self._assets: dict[tuple[str, str], _AssetLiveState] = {}
        self._universes: dict[str, Universe] = {}

        self.portfolio = MultiExchangePortfolio()
        self.state = LiveState()

        self._lock = threading.Lock()
        self._running = False
        self._last_processed_bar: dict[str, int] = {}
        self._kill_listener = _ManualKillSwitch(self._manual_kill)

        os.makedirs(self.config.log_dir, exist_ok=True)
        self._trade_log_path = os.path.join(self.config.log_dir, self.config.trade_log_csv)

    @property
    def primary_exchange(self) -> str:
        return self._exchange_names[0]

    # ── Component resolution ──────────────────────────────────────────────

    def _resolve_sizer(self, symbol: str) -> Sizer:
        if isinstance(self._sizer_spec, dict):
            return copy.deepcopy(self._sizer_spec.get(symbol, default_sizer()))
        elif self._sizer_spec is not None:
            return copy.deepcopy(self._sizer_spec)
        return default_sizer()

    def _resolve_stop_loss(self, symbol: str) -> StopLoss:
        if isinstance(self._stop_loss_spec, dict):
            return copy.deepcopy(self._stop_loss_spec.get(symbol, default_stop_loss()))
        elif self._stop_loss_spec is not None:
            return copy.deepcopy(self._stop_loss_spec)
        return default_stop_loss()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        self._setup_logging()
        mode_label = {
            "cross": f"CrossExchange: {self.cross_strategy.__class__.__name__}",
            "per_exchange": "PerExchange: " + ", ".join(
                f"{ex}={s.__class__.__name__}"
                for ex, s in self._per_exchange_strategies.items()
            ),
        }[self._mode]

        logger.info("=" * 60)
        logger.info("MULTI-EXCHANGE ENGINE — %s", mode_label)
        logger.info("  Symbols: %s | Exchanges: %s", ", ".join(self._symbols), ", ".join(self._exchange_names))
        if self.overlay:
            logger.info("  Overlay: %s", self.overlay.__class__.__name__)
        logger.info("=" * 60)

        # 1. Create executors + register with portfolio
        for cred in self._creds:
            ex = create_executor(cred)
            self._executors[cred.exchange] = ex
            self.portfolio.register(ex)
            for sym in self._symbols:
                ex.set_leverage(sym, int(self.config.leverage), cross=self.config.margin_type == "cross")

        # 2. Sync equity
        self.portfolio.refresh_equity()
        self.state.equity = self.portfolio.total_equity()
        self.state.starting_equity = self.state.equity
        self.state.peak_equity = self.state.equity
        logger.info("Total equity: $%.2f", self.state.equity)
        for name, eq in self.portfolio.equity_breakdown().items():
            logger.info("  %s: $%.2f", name, eq)

        # 3. Per-exchange: universe, bar builders
        for cred in self._creds:
            self._universes[cred.exchange] = Universe(symbols=self._symbols)
            for sym in self._symbols:
                key = (cred.exchange, sym)
                ast = _AssetLiveState(
                    symbol=sym,
                    exchange=cred.exchange,
                    position=self._executors[cred.exchange].get_position(sym),
                    stop_loss=self._resolve_stop_loss(sym),
                )
                ast.bar_builder = create_bar_builder(
                    interval_s=self.config.bar_interval_s,
                    max_bars=self.config.max_bars_in_memory,
                    on_bar_close=lambda data, e=cred.exchange, s=sym: self._on_new_bar(e, s, data),
                )
                self._assets[key] = ast

        # 4. Log initial positions
        for cred in self._creds:
            for sym in self._symbols:
                pos = self._executors[cred.exchange].get_position(sym)
                logger.info("  %s/%s: %s %.4f @ %.4f", cred.exchange, sym, pos.side.name, pos.size, pos.entry_price)

        # 5. Warm up
        self._warmup()

        # 6. Start feeds
        self._running = True
        for cred in self._creds:
            for sym in self._symbols:
                key = (cred.exchange, sym)
                ast = self._assets[key]
                ast.feed = create_feed(
                    exchange=cred.exchange, symbol=sym,
                    testnet=cred.testnet, symbol_map=cred.symbol_map,
                )
                ast.feed.start(on_trade=ast.bar_builder.on_trade, on_candle=ast.bar_builder.on_candle)

        logger.info("Multi-exchange engine running | %d exchange(s) × %d symbol(s)", len(self._exchange_names), len(self._symbols))
        self._kill_listener.start()
        self._main_loop()

    def stop(self):
        self._running = False
        self._kill_listener.stop()
        for ast in self._assets.values():
            if ast.feed:
                ast.feed.stop()
        logger.info("Multi-exchange engine stopped | trades=%d | closed=%d", len(self.state.trades), len(self.state.closed_trades))

    def _manual_kill(self):
        logger.critical("MANUAL KILL SWITCH ACTIVATED — flattening all positions")
        self.portfolio.flatten_all(self._symbols)
        self.state.kill_switch = True

    def _main_loop(self):
        heartbeat_interval = 60
        last_heartbeat = time.time()
        try:
            while self._running and not self.state.kill_switch:
                time.sleep(1)
                now = time.time()
                if now - last_heartbeat < heartbeat_interval:
                    continue
                last_heartbeat = now

                self.portfolio.refresh_equity()
                self.state.equity = self.portfolio.total_equity()

                parts = []
                for ex_name in self._exchange_names:
                    for sym in self._symbols:
                        ast = self._assets.get((ex_name, sym))
                        bc = ast.bar_builder.bar_count if ast else 0
                        pos = ast.position if ast else Position()
                        parts.append(f"{ex_name}/{sym}: {bc}bars {pos.side.name} {pos.size:.4f}")

                logger.info("Heartbeat | equity=$%.2f | %s", self.state.equity, " | ".join(parts))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
        finally:
            self.stop()

    # ── Bar processing ────────────────────────────────────────────────────

    def _on_new_bar(self, exchange: str, trigger_symbol: str, data: pd.DataFrame):
        if not self._running or self.state.kill_switch:
            return
        key = (exchange, trigger_symbol)
        ast = self._assets.get(key)
        if ast is None:
            return
        bc = ast.bar_builder.bar_count
        dedup_key = f"{exchange}:{trigger_symbol}"
        if bc == self._last_processed_bar.get(dedup_key, -1):
            return
        self._last_processed_bar[dedup_key] = bc
        threading.Thread(
            target=self._safe_process_bar, args=(exchange, trigger_symbol),
            daemon=True, name=f"mxe-{exchange}-{trigger_symbol}",
        ).start()

    def _safe_process_bar(self, exchange: str, trigger_symbol: str):
        with self._lock:
            try:
                self._process_bar(exchange, trigger_symbol)
            except Exception as e:
                logger.error("Bar processing error [%s/%s]: %s", exchange, trigger_symbol, e, exc_info=True)

    def _process_bar(self, trigger_exchange: str, trigger_symbol: str):
        # 1. Update all universes
        for ex_name in self._exchange_names:
            for sym in self._symbols:
                ast = self._assets.get((ex_name, sym))
                if ast:
                    df = ast.bar_builder.to_dataframe()
                    if len(df) >= 2:
                        self._universes[ex_name].update_asset_bars(sym, df)

        trigger_df = self._assets[(trigger_exchange, trigger_symbol)].bar_builder.to_dataframe()
        if len(trigger_df) < 2:
            return
        ts = trigger_df.index[-1] if isinstance(trigger_df.index, pd.DatetimeIndex) else pd.Timestamp.now()

        # 2. Refresh portfolio equity
        self.portfolio.refresh_equity()
        self.state.equity = self.portfolio.total_equity()
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

        # 3. Kill switch
        if self._check_kill_switch():
            return

        # 4. Sync positions from all exchanges
        all_positions: dict[str, dict[str, Position]] = {}
        for ex_name in self._exchange_names:
            ex = self._executors[ex_name]
            ex_pos = {}
            for sym in self._symbols:
                pos = ex.get_position(sym)
                ex_pos[sym] = pos
                key = (ex_name, sym)
                if key in self._assets:
                    self._assets[key].position = pos
            all_positions[ex_name] = ex_pos

        self.state.positions = all_positions.get(self.primary_exchange, {})

        # 5. Stop-loss checks
        for ex_name in self._exchange_names:
            for sym in self._symbols:
                key = (ex_name, sym)
                ast = self._assets.get(key)
                if ast is None:
                    continue
                pos = ast.position
                if pos.side == Side.FLAT:
                    continue
                universe = self._universes[ex_name]
                try:
                    df = universe.ohlcv(sym)
                except KeyError:
                    continue
                if len(df) < 2:
                    continue
                idx = len(df) - 1
                price = df["close"].iat[idx]
                l2_snap = ast.feed.latest_l2 if ast.feed else None
                bar_dict = {c: df[c].iat[idx] for c in df.columns if np.isscalar(df[c].iat[idx])}

                try:
                    executor = self._executors[ex_name]
                    funding_snap = executor.fetch_funding_rate(sym)
                    if funding_snap is not None:
                        bar_dict["funding_rate"] = funding_snap.rate
                        bar_dict["funding_rate_ann_bps"] = funding_snap.rate_annualized
                        if funding_snap.oracle_price > 0:
                            bar_dict["oracle_price"] = funding_snap.oracle_price
                        if funding_snap.mark_price > 0:
                            bar_dict["mark_price"] = funding_snap.mark_price
                except Exception:
                    pass

                stop_ctx = StopContext(
                    position=pos, bar_idx=idx,
                    open=df["open"].iat[idx], high=df["high"].iat[idx],
                    low=df["low"].iat[idx], close=price,
                    data=df, l2=l2_snap, bar_data=bar_dict,
                )
                ast.stop_loss.update(stop_ctx)
                stop_result = ast.stop_loss.check(stop_ctx)

                if stop_result.triggered:
                    logger.info("STOP on %s/%s: %s", ex_name, sym, stop_result.reason)
                    executor = self._executors[ex_name]
                    close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
                    fill = executor.market_order(sym, close_side, pos.size, reduce_only=True)
                    if fill.success:
                        self._record_trade(ex_name, sym, pos, fill, ts, stop_result.reason)
                        ast.position = Position()
                        ast.open_trade = None
                        ast.stop_loss.reset()

        # 6. Generate targets (mode-dependent)
        merged_target = self._generate_targets(trigger_exchange, trigger_symbol, ts, all_positions)
        if merged_target is None:
            return

        # 7. Apply overlay
        if self.overlay:
            cross_ctx = self._build_cross_ctx(ts, all_positions)
            merged_target = self.overlay.adjust(merged_target, cross_ctx)

        # 8. Execute
        self._execute_target(merged_target, ts, all_positions)

    # ── Target generation ─────────────────────────────────────────────────

    def _generate_targets(self, trigger_exchange, trigger_symbol, ts, all_positions) -> MultiExchangeTarget | None:
        if self._mode == "cross":
            return self._generate_cross(ts, all_positions)
        return self._generate_per_exchange(trigger_exchange, ts, all_positions)

    def _generate_cross(self, ts, all_positions) -> MultiExchangeTarget | None:
        try:
            self.cross_strategy.setup(self._universes)
        except Exception as e:
            logger.error("CrossStrategy setup failed: %s", e)
            return None
        ctx = self._build_cross_ctx(ts, all_positions)
        return self.cross_strategy.generate(ctx)

    def _generate_per_exchange(self, trigger_exchange, ts, all_positions) -> MultiExchangeTarget | None:
        per_exchange_targets: dict[str, PortfolioTarget] = {}
        for ex_name, strat in self._per_exchange_strategies.items():
            universe = self._universes.get(ex_name)
            if universe is None:
                continue
            try:
                strat.setup(universe)
            except Exception as e:
                logger.error("%s strategy setup failed: %s", ex_name, e)
                continue
            bar_idx = 0
            for sym in self._symbols:
                try:
                    bar_idx = max(bar_idx, len(universe.ohlcv(sym)) - 1)
                except KeyError:
                    pass
            ctx = StrategyContext(
                universe=universe, bar_idx=bar_idx, timestamp=ts,
                equity=self.portfolio.total_equity(),
                positions=all_positions.get(ex_name, {}),
                trade_history=self.state.closed_trades,
            )
            per_exchange_targets[ex_name] = strat.generate(ctx)
        return MultiExchangeTarget.from_per_exchange(per_exchange_targets)

    def _build_cross_ctx(self, ts, all_positions) -> CrossExchangeContext:
        bar_idx = 0
        for u in self._universes.values():
            for sym in self._symbols:
                try:
                    bar_idx = max(bar_idx, len(u.ohlcv(sym)) - 1)
                except KeyError:
                    pass
        return CrossExchangeContext(
            universes=self._universes, bar_idx=bar_idx, timestamp=ts,
            total_equity=self.portfolio.total_equity(),
            equity_by_exchange=self.portfolio.equity_breakdown(),
            positions=all_positions, portfolio=self.portfolio,
            trade_history=self.state.closed_trades,
        )

    # ── Execution ─────────────────────────────────────────────────────────

    def _execute_target(self, target: MultiExchangeTarget, ts, all_positions):
        # Phase 1: closes
        for ex_name in self._exchange_names:
            executor = self._executors[ex_name]
            for sym in self._symbols:
                key = (ex_name, sym)
                ast = self._assets.get(key)
                if ast is None:
                    continue
                pos = ast.position
                if pos.side == Side.FLAT:
                    continue
                desired = target[(ex_name, sym)]
                if not (desired.side == Side.FLAT or desired.side != pos.side):
                    continue
                reason = desired.reason or "target_flat"
                close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
                logger.info("CLOSING %s %s %.4f on %s — %s", pos.side.name, sym, pos.size, ex_name, reason)
                fill = executor.market_order(sym, close_side, pos.size, reduce_only=True)
                if fill.success:
                    self._record_trade(ex_name, sym, pos, fill, ts, reason)
                    ast.position = Position()
                    ast.open_trade = None
                    ast.stop_loss.reset()

        # Phase 2: opens
        for ex_name in self._exchange_names:
            executor = self._executors[ex_name]
            for sym in self._symbols:
                key = (ex_name, sym)
                ast = self._assets.get(key)
                if ast is None or ast.position.side != Side.FLAT:
                    continue
                alloc = target[(ex_name, sym)]
                if alloc.side == Side.FLAT or alloc.weight <= 0:
                    continue
                if self.state.daily_trades >= self.config.max_daily_trades:
                    logger.warning("Daily trade limit reached")
                    continue
                price = ast.bar_builder.last_close
                if np.isnan(price) or price <= 0:
                    continue

                max_notional = self.state.equity * alloc.weight * self.config.leverage
                size = max_notional / price if price > 0 else 0

                universe = self._universes.get(ex_name)
                if universe:
                    try:
                        df = universe.ohlcv(sym)
                        idx = len(df) - 1
                        l2_snap = ast.feed.latest_l2 if ast.feed else None
                        bar_dict = {c: df[c].iat[idx] for c in df.columns if np.isscalar(df[c].iat[idx])}
                        try:
                            _fs = self._executors[ex_name].fetch_funding_rate(sym)
                            if _fs is not None:
                                bar_dict["funding_rate"] = _fs.rate
                                bar_dict["funding_rate_ann_bps"] = _fs.rate_annualized
                                if _fs.oracle_price > 0:
                                    bar_dict["oracle_price"] = _fs.oracle_price
                                if _fs.mark_price > 0:
                                    bar_dict["mark_price"] = _fs.mark_price
                        except Exception:
                            pass
                        sizer_cfg = _sizer_config_shim(self.config, self.state.equity)
                        sizing_ctx = SizingContext(
                            equity=self.state.equity, price=price,
                            allocation=alloc, config=sizer_cfg,
                            position=ast.position, data=df, bar_idx=idx,
                            trade_history=self.state.closed_trades,
                            l2=l2_snap, bar_data=bar_dict,
                        )
                        sizer_size = self._resolve_sizer(sym).compute(sizing_ctx)
                        size = min(size, sizer_size)
                    except Exception:
                        pass

                if size <= 0:
                    continue

                logger.info("OPENING %s %.4f %s on %s @ ~%.4f (w=%.2f)", alloc.side.name, size, sym, ex_name, price, alloc.weight)
                fill = executor.market_order(sym, alloc.side, size)
                if fill.success:
                    new_pos = Position(
                        side=alloc.side,
                        size=fill.filled_size or size,
                        entry_price=fill.fill_price or price,
                        entry_timestamp=ts,
                    )
                    ast.position = new_pos

                    try:
                        df = self._universes[ex_name].ohlcv(sym)
                        idx = len(df) - 1
                        l2_snap = ast.feed.latest_l2 if ast.feed else None
                        bar_dict = {c: df[c].iat[idx] for c in df.columns if np.isscalar(df[c].iat[idx])}
                        stop_ctx = StopContext(
                            position=new_pos, bar_idx=idx,
                            open=df["open"].iat[idx], high=df["high"].iat[idx],
                            low=df["low"].iat[idx], close=price,
                            data=df, l2=l2_snap, bar_data=bar_dict,
                        )
                        ast.stop_loss.on_entry(new_pos, stop_ctx)
                    except Exception:
                        pass

                    trade = Trade(
                        timestamp=ts, side=alloc.side,
                        size=fill.filled_size or size,
                        entry_price=fill.fill_price or price,
                        reason_entry=alloc.reason,
                        bar_values=alloc.meta,
                        meta={"symbol": sym, "exchange": ex_name},
                    )
                    self.state.trades.append(trade)
                    ast.open_trade = trade
                    self.state.daily_trades += 1

                    if self._mode == "cross" and self.cross_strategy:
                        self.cross_strategy.on_fill(ex_name, sym, alloc.side, fill.filled_size or size, fill.fill_price or price)
                    elif ex_name in self._per_exchange_strategies:
                        self._per_exchange_strategies[ex_name].on_fill(sym, alloc.side, fill.filled_size or size, fill.fill_price or price)

    # ── Trade recording ───────────────────────────────────────────────────

    def _record_trade(self, exchange, symbol, pos, fill, ts, reason):
        key = (exchange, symbol)
        ast = self._assets.get(key)
        exit_price = fill.fill_price if fill.fill_price > 0 else (ast.bar_builder.last_close if ast else 0.0)
        direction = 1 if pos.side == Side.LONG else -1
        pnl = (exit_price - pos.entry_price) * pos.size * direction

        if ast and ast.open_trade and ast.open_trade.exit_price is None:
            t = ast.open_trade
            t.exit_price = exit_price
            t.exit_timestamp = ts
            t.pnl = pnl
            t.pnl_pct = pnl / (pos.entry_price * pos.size) if pos.entry_price * pos.size > 0 else 0
            t.fees = 0.0
            t.reason_exit = reason
            t.meta["symbol"] = symbol
            t.meta["exchange"] = exchange
            self.state.closed_trades.append(t)

        self.state.daily_pnl += pnl
        logger.info(
            "TRADE CLOSED: %s %s %.4f on %s | entry=%.4f exit=%.4f | pnl=$%.2f | %s",
            pos.side.name, symbol, pos.size, exchange, pos.entry_price, exit_price, pnl, reason,
        )
        self._write_trade_csv(self.state.closed_trades[-1] if self.state.closed_trades else None)

    def _write_trade_csv(self, trade):
        if trade is None:
            return
        file_exists = os.path.exists(self._trade_log_path)
        row = trade.to_dict()
        with open(self._trade_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ── Risk management ───────────────────────────────────────────────────

    def _check_kill_switch(self) -> bool:
        daily_state = DailyLimitState(
            daily_pnl=self.state.daily_pnl,
            starting_equity=self.state.starting_equity,
        )
        if check_daily_loss_limit(daily_state, self.config):
            self.portfolio.flatten_all(self._symbols)
            self.state.kill_switch = True
            return True
        return False

    # ── Warm-up ───────────────────────────────────────────────────────────

    def _warmup(self):
        for cred in self._creds:
            executor = self._executors[cred.exchange]
            for sym in self._symbols:
                try:
                    now_ms = int(time.time() * 1000)
                    start_ms = now_ms - self.config.warmup_bars * self.config.bar_interval_s * 1000
                    rows = executor.fetch_historical_candles(sym, "1m", start_ms, now_ms)
                    if rows:
                        df = pd.DataFrame(rows).set_index("timestamp")
                        key = (cred.exchange, sym)
                        if key in self._assets:
                            self._assets[key].bar_builder.seed(df)
                        self._universes[cred.exchange].update_asset_bars(sym, df)
                        logger.info("  %s/%s warmup: %d bars", cred.exchange, sym, len(df))
                    else:
                        logger.warning("  %s/%s: no warmup candles", cred.exchange, sym)
                except Exception as e:
                    logger.warning("  %s/%s warmup failed: %s", cred.exchange, sym, e)

    def _setup_logging(self):
        log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format=log_fmt,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config.log_dir, "multi_exchange.log"), mode="a"),
            ],
        )
