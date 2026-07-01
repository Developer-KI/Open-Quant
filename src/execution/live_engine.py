"""
execution/live_engine.py — Single-exchange live trading engine.

MultiExchangeEngine has been extracted to multi_exchange_engine.py.
LiveState and _AssetLiveState have been extracted to live_state.py.

This file retains:
  • _ManualKillSwitch — stdin listener for emergency flatten
  • _sizer_config_shim — bridges LiveConfig to BacktestConfig for sizers
  • LiveEngine — single-exchange orchestration

Import MultiExchangeEngine from execution.multi_exchange_engine (or from
execution/ which re-exports both).
"""

from __future__ import annotations

import copy
import csv
from datetime import datetime
import logging
import os
import select
import sys
import threading
import time

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

from strategy.base import Strategy, StrategyContext
from strategy.universe import Universe

from .base_executor_feed import BaseExecutor, FillResult
from .factory import create_executor, create_feed, create_bar_builder
from .live_state import _AssetLiveState, LiveState

logger = logging.getLogger(__name__)


# ── Manual kill switch ────────────────────────────────────────────────────────

KILL_KEY = "q"


class _ManualKillSwitch:
    """
    Background thread that listens for a keypress to trigger an emergency shutdown.
    Press KILL_KEY (default: 'q') then Enter.

    Silently disables itself in non-interactive environments (Docker, systemd).
    """

    def __init__(self, callback, key: str = KILL_KEY):
        self._callback = callback
        self._key = key
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        if not sys.stdin.isatty():
            logger.info("Non-interactive terminal — manual kill switch disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True, name="kill-switch")
        self._thread.start()
        logger.info("Manual kill switch active — press '%s' + Enter to flatten & shutdown", self._key)

    def stop(self):
        self._running = False

    def _listen(self):
        try:
            while self._running:
                if select.select([sys.stdin], [], [], 1.0)[0]:
                    line = sys.stdin.readline().strip().lower()
                    if line == self._key:
                        logger.critical("MANUAL KILL — '%s' pressed, flattening all positions", self._key)
                        self._callback()
                        return
        except Exception:
            pass


def _sizer_config_shim(config: LiveConfig, equity: float) -> BacktestConfig:
    """Bridge LiveConfig fields into a BacktestConfig for sizer.compute()."""
    return BacktestConfig(
        initial_capital=equity,
        risk_per_trade=config.risk_per_trade,
        max_position_pct=config.max_position_pct,
        leverage=config.leverage,
        margin_type=config.margin_type,
    )


# ── LiveEngine — single-exchange ──────────────────────────────────────────────


class LiveEngine:
    """
    Single-exchange live engine.

    Instantiates executor/feed via factory so config.exchange = "binance"
    works without any code changes.
    """

    def __init__(
        self,
        strategy: Strategy,
        config: LiveConfig | None = None,
        sizer: Sizer | dict[str, Sizer] | None = None,
        stop_loss: StopLoss | dict[str, StopLoss] | None = None,
    ):
        self.config = config or LiveConfig()
        self._sizer_spec = sizer
        self._stop_loss_spec = stop_loss
        self.strategy = strategy

        self._symbols = self.config.active_symbols
        self._is_single = len(self._symbols) == 1

        self.executor: BaseExecutor | None = None
        self._assets: dict[str, _AssetLiveState] = {}
        self.state = LiveState()
        self._lock = threading.Lock()
        self._running = False
        self._last_processed_bar: dict[str, int] = {}
        self._kill_listener = _ManualKillSwitch(self._manual_kill)

        os.makedirs(self.config.log_dir, exist_ok=True)
        self._run_log_dir: str = self.config.log_dir  # replaced in start()
        self._trade_log_path: str = ""                 # set in start()
        self._universe = Universe(symbols=self._symbols)

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

    def _get_exchange_cred(self) -> ExchangeCredentials:
        return self.config.get_credentials()[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._run_log_dir = os.path.join(self.config.log_dir, run_ts)
        os.makedirs(self._run_log_dir, exist_ok=True)
        self._trade_log_path = os.path.join(self._run_log_dir, self.config.trade_log_csv)

        self._setup_logging()
        logger.info("=" * 60)
        logger.info("LIVE ENGINE STARTING — %s on %s (%s)", self.strategy.__class__.__name__, ", ".join(self._symbols), self.config.exchange)
        logger.info("=" * 60)

        cred = self._get_exchange_cred()
        self.executor = create_executor(cred)

        for sym in self._symbols:
            self.executor.set_leverage(sym, int(self.config.leverage), cross=self.config.margin_type == "cross")

        self.state.equity = self.executor.get_equity()
        self.state.starting_equity = self.state.equity
        self.state.peak_equity = self.state.equity

        for sym in self._symbols:
            pos = self.executor.get_position(sym)
            self.state.positions[sym] = pos
            logger.info("  %s — %s %.4f @ %.4f", sym, pos.side.name, pos.size, pos.entry_price)
        logger.info("Account equity: $%.2f", self.state.equity)

        for sym in self._symbols:
            ast = _AssetLiveState(
                symbol=sym, exchange=cred.exchange,
                position=self.state.positions[sym],
                stop_loss=self._resolve_stop_loss(sym),
            )
            ast.bar_builder = create_bar_builder(
                interval_s=self.config.bar_interval_s,
                max_bars=self.config.max_bars_in_memory,
                on_bar_close=lambda data, s=sym: self._on_new_bar(s, data),
            )
            self._assets[sym] = ast

        self._warmup(cred)

        self._running = True
        for sym in self._symbols:
            ast = self._assets[sym]
            ast.feed = create_feed(exchange=cred.exchange, symbol=sym, testnet=cred.testnet, symbol_map=cred.symbol_map)
            ast.feed.start(on_trade=ast.bar_builder.on_trade, on_candle=ast.bar_builder.on_candle, on_l2=None)

        logger.info("Live engine running on %d symbol(s)", len(self._symbols))
        self._kill_listener.start()
        self._main_loop()

    def _main_loop(self):
        heartbeat_interval = 60
        last_heartbeat = time.time()
        last_bar_counts = {s: self._assets[s].bar_builder.bar_count for s in self._symbols}
        stall_ticks = {s: 0 for s in self._symbols}

        try:
            while self._running and not self.state.kill_switch:
                time.sleep(1)
                now = time.time()
                if now - last_heartbeat < heartbeat_interval:
                    continue
                last_heartbeat = now

                parts = []
                for sym in self._symbols:
                    ast = self._assets[sym]
                    bc = ast.bar_builder.bar_count
                    pos = self.state.positions.get(sym, Position())
                    parts.append(f"{sym}: {bc} bars, pos={pos.side.name} {pos.size:.4f}")

                    if bc == last_bar_counts.get(sym, 0):
                        stall_ticks[sym] += 1
                    else:
                        stall_ticks[sym] = 0
                    last_bar_counts[sym] = bc

                    if stall_ticks[sym] >= 3:
                        logger.warning("WATCHDOG: %s stalled for %d heartbeats", sym, stall_ticks[sym])

                logger.info("Heartbeat | equity=$%.2f | %s", self.state.equity, " | ".join(parts))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self._kill_listener.stop()
        for ast in self._assets.values():
            if ast.feed:
                ast.feed.stop()
        logger.info("Engine stopped | trades=%d | closed=%d", len(self.state.trades), len(self.state.closed_trades))

    def _manual_kill(self):
        logger.critical("MANUAL KILL SWITCH ACTIVATED — flattening all positions")
        for sym in self._symbols:
            try:
                self.executor.close_position(sym)
                self.executor.cancel_all(sym)
            except Exception as e:
                logger.error("Failed to flatten %s: %s", sym, e)
        self.state.kill_switch = True

    # ── Bar processing ────────────────────────────────────────────────────

    def _on_new_bar(self, trigger_symbol: str, data: pd.DataFrame):
        if not self._running or self.state.kill_switch:
            return
        bar_count = self._assets[trigger_symbol].bar_builder.bar_count
        if bar_count == self._last_processed_bar.get(trigger_symbol, -1):
            return
        self._last_processed_bar[trigger_symbol] = bar_count
        threading.Thread(
            target=self._safe_process_bar, args=(trigger_symbol,),
            daemon=True, name=f"bar-proc-{trigger_symbol}",
        ).start()

    def _safe_process_bar(self, trigger_symbol: str):
        with self._lock:
            try:
                self._process_bar(trigger_symbol)
            except Exception as e:
                logger.error("Bar processing error [%s]: %s", trigger_symbol, e, exc_info=True)

    def _process_bar(self, trigger_symbol: str):
        for sym in self._symbols:
            df = self._assets[sym].bar_builder.to_dataframe()
            if len(df) >= 2:
                self._universe.update_asset_bars(sym, df)

        trigger_df = self._assets[trigger_symbol].bar_builder.to_dataframe()
        if len(trigger_df) < 2:
            return
        ts = trigger_df.index[-1] if isinstance(trigger_df.index, pd.DatetimeIndex) else pd.Timestamp.now()

        self.state.equity = self.executor.get_equity()
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

        if self._check_kill_switch():
            return

        for sym in self._symbols:
            exchange_pos = self.executor.get_position(sym)
            local_pos = self.state.positions.get(sym, Position())
            if exchange_pos.side != local_pos.side or abs(exchange_pos.size - local_pos.size) > 1e-8:
                logger.warning("%s position mismatch — local: %s %.4f, exchange: %s %.4f", sym, local_pos.side.name, local_pos.size, exchange_pos.side.name, exchange_pos.size)
                self.state.positions[sym] = exchange_pos
                self._assets[sym].position = exchange_pos
            else:
                self._assets[sym].position = local_pos

        for sym in self._symbols:
            ast = self._assets[sym]
            pos = ast.position
            if pos.side != Side.FLAT and pos.size > 0:
                price = ast.bar_builder.last_close
                if not np.isnan(price):
                    direction = 1 if pos.side == Side.LONG else -1
                    pos.unrealized_pnl = (price - pos.entry_price) * pos.size * direction

        for sym in self._symbols:
            ast = self._assets[sym]
            pos = ast.position
            if pos.side == Side.FLAT:
                continue
            df = self._universe.ohlcv(sym) if sym in self._universe._assets else None
            if df is None or len(df) < 2:
                continue
            idx = len(df) - 1
            price = df["close"].iat[idx]
            l2_snap = ast.feed.latest_l2 if ast.feed else None
            bar_dict = {c: df[c].iat[idx] for c in df.columns if np.isscalar(df[c].iat[idx])}

            funding_snap = self.executor.fetch_funding_rate(sym)
            if funding_snap is not None:
                bar_dict["funding_rate"] = funding_snap.rate
                bar_dict["funding_rate_ann_bps"] = funding_snap.rate_annualized
                if funding_snap.oracle_price > 0:
                    bar_dict["oracle_price"] = funding_snap.oracle_price
                if funding_snap.mark_price > 0:
                    bar_dict["mark_price"] = funding_snap.mark_price

            stop_ctx = StopContext(
                position=pos, bar_idx=idx,
                open=df["open"].iat[idx], high=df["high"].iat[idx],
                low=df["low"].iat[idx], close=price,
                data=df, l2=l2_snap, bar_data=bar_dict,
            )
            ast.stop_loss.update(stop_ctx)
            stop_result = ast.stop_loss.check(stop_ctx)
            if stop_result.triggered:
                logger.info("STOP TRIGGERED on %s: %s", sym, stop_result.reason)
                fill = self._execute_close(sym, pos, stop_result.reason)
                if fill.success:
                    self._record_trade(sym, pos, fill, ts, stop_result.reason)
                    self.state.positions[sym] = Position()
                    ast.position = Position()
                    ast.open_trade = None
                    ast.stop_loss.reset()

        try:
            self.strategy.setup(self._universe)
        except Exception as e:
            logger.error("Strategy setup failed: %s", e)
            return
        self.state.strategy_setup_done = True

        bar_idx = len(self._universe.ohlcv(trigger_symbol)) - 1 if trigger_symbol in self._universe._assets else 0
        ctx = StrategyContext(
            universe=self._universe, bar_idx=bar_idx, timestamp=ts,
            equity=self.state.equity,
            positions={s: self._assets[s].position for s in self._symbols},
            trade_history=self.state.closed_trades,
        )
        target = self.strategy.generate(ctx)

        for sym in self._symbols:
            ast = self._assets[sym]
            pos = ast.position
            if pos.side == Side.FLAT:
                continue
            desired = target[sym]
            if desired.side == Side.FLAT or desired.side != pos.side:
                reason = desired.reason or "target_flat"
                fill = self._execute_close(sym, pos, reason)
                if fill.success:
                    self._record_trade(sym, pos, fill, ts, reason)
                    self.state.positions[sym] = Position()
                    ast.position = Position()
                    ast.open_trade = None
                    ast.stop_loss.reset()

        for sym in self._symbols:
            ast = self._assets[sym]
            pos = ast.position
            if pos.side != Side.FLAT:
                continue
            alloc = target[sym]
            if alloc.side == Side.FLAT or alloc.weight <= 0:
                continue
            if self.state.daily_trades >= self.config.max_daily_trades:
                continue

            price = ast.bar_builder.last_close
            if np.isnan(price) or price <= 0:
                continue
            df = self._universe.ohlcv(sym) if sym in self._universe._assets else None
            if df is None or len(df) < 2:
                continue
            idx = len(df) - 1
            l2_snap = ast.feed.latest_l2 if ast.feed else None
            bar_dict = {c: df[c].iat[idx] for c in df.columns if np.isscalar(df[c].iat[idx])}

            funding_snap = self.executor.fetch_funding_rate(sym)
            if funding_snap is not None:
                bar_dict["funding_rate"] = funding_snap.rate
                bar_dict["funding_rate_ann_bps"] = funding_snap.rate_annualized
                if funding_snap.oracle_price > 0:
                    bar_dict["oracle_price"] = funding_snap.oracle_price
                if funding_snap.mark_price > 0:
                    bar_dict["mark_price"] = funding_snap.mark_price

            sizer_cfg = _sizer_config_shim(self.config, self.state.equity)
            sizing_ctx = SizingContext(
                equity=self.state.equity, price=price,
                allocation=alloc, config=sizer_cfg,
                position=pos, data=df, bar_idx=idx,
                trade_history=self.state.closed_trades,
                l2=l2_snap, bar_data=bar_dict,
            )
            size = self._resolve_sizer(sym).compute(sizing_ctx)

            if self._is_single:
                max_notional = self.state.equity * self.config.max_position_pct * self.config.leverage
            else:
                max_notional = self.state.equity * alloc.weight * self.config.leverage
            max_size = max_notional / price if price > 0 else 0
            size = min(size, max_size)
            if size <= 0:
                continue

            logger.info("OPENING %s %.4f %s @ ~%.4f", alloc.side.name, size, sym, price)
            fill = self._execute_open(sym, alloc.side, size)
            if fill.success:
                new_pos = Position(
                    side=alloc.side,
                    size=fill.filled_size or size,
                    entry_price=fill.fill_price or price,
                    entry_timestamp=ts,
                )
                self.state.positions[sym] = new_pos
                ast.position = new_pos

                stop_ctx = StopContext(
                    position=new_pos, bar_idx=idx,
                    open=df["open"].iat[idx], high=df["high"].iat[idx],
                    low=df["low"].iat[idx], close=price,
                    data=df, l2=l2_snap, bar_data=bar_dict,
                )
                ast.stop_loss.on_entry(new_pos, stop_ctx)

                trade = Trade(
                    timestamp=ts, side=alloc.side,
                    size=fill.filled_size or size,
                    entry_price=fill.fill_price or price,
                    reason_entry=alloc.reason,
                    bar_values=alloc.meta,
                    meta={"symbol": sym, "exchange": self.config.exchange},
                )
                self.state.trades.append(trade)
                ast.open_trade = trade
                self.state.daily_trades += 1
                self.strategy.on_fill(sym, alloc.side, fill.filled_size or size, fill.fill_price or price)

    # ── Execution helpers ─────────────────────────────────────────────────

    def _execute_open(self, symbol, side, size) -> FillResult:
        if self.config.order_type == "limit":
            mid = self.executor.get_mid_price(symbol)
            offset = mid * self.config.limit_chase_bps / 1e4
            px = mid + offset if side == Side.LONG else mid - offset
            return self.executor.limit_order(symbol, side, size, px)
        return self.executor.market_order(symbol, side, size)

    def _execute_close(self, symbol, pos, reason) -> FillResult:
        close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
        return self.executor.market_order(symbol, close_side, pos.size, reduce_only=self.config.reduce_only_exits)

    # ── Trade recording ───────────────────────────────────────────────────

    def _record_trade(self, symbol, pos, fill, ts, reason):
        exit_price = fill.fill_price if fill.fill_price > 0 else self._assets[symbol].bar_builder.last_close
        direction = 1 if pos.side == Side.LONG else -1
        pnl = (exit_price - pos.entry_price) * pos.size * direction

        ast = self._assets[symbol]
        if ast.open_trade is not None and ast.open_trade.exit_price is None:
            t = ast.open_trade
            t.exit_price = exit_price
            t.exit_timestamp = ts
            t.pnl = pnl
            t.pnl_pct = pnl / (pos.entry_price * pos.size) if pos.entry_price * pos.size > 0 else 0
            t.fees = 0.0
            t.reason_exit = reason
            t.meta["symbol"] = symbol
            t.meta["exchange"] = self.config.exchange
            self.state.closed_trades.append(t)

        self.state.daily_pnl += pnl
        self.state.equity = self.executor.get_equity()
        logger.info("TRADE CLOSED: %s %s %.4f | entry=%.4f exit=%.4f | pnl=$%.2f | reason=%s", pos.side.name, symbol, pos.size, pos.entry_price, exit_price, pnl, reason)
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
            for sym in self._symbols:
                self.executor.close_position(sym)
                self.executor.cancel_all(sym)
            self.state.kill_switch = True
            return True
        return False

    # ── Warm-up ───────────────────────────────────────────────────────────

    def _warmup(self, cred: ExchangeCredentials):
        for sym in self._symbols:
            logger.info("Fetching %d warmup bars for %s...", self.config.warmup_bars, sym)
            try:
                now_ms = int(time.time() * 1000)
                start_ms = now_ms - self.config.warmup_bars * self.config.bar_interval_s * 1000
                rows = self.executor.fetch_historical_candles(sym, "1m", start_ms, now_ms)
                if rows:
                    df = pd.DataFrame(rows).set_index("timestamp")
                    self._assets[sym].bar_builder.seed(df)
                    self._universe.update_asset_bars(sym, df)
                    logger.info("  %s warmup: %d bars", sym, len(df))
                else:
                    logger.warning("  %s: no warmup candles", sym)
            except Exception as e:
                logger.warning("  %s warmup failed: %s", sym, e)

    def _setup_logging(self):
        log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format=log_fmt,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self._run_log_dir, "live_engine.log"), mode="a"),
            ],
        )
