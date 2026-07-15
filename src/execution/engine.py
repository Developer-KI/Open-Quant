"""
execution/engine.py — Unified live trading engine.

Single-exchange:
    Engine(strategy=my_strategy, config=cfg)

Multi-exchange, independent per-exchange strategies:
    Engine(per_exchange_strategies={"binance": s1, "hyperliquid": s2}, config=cfg)

Multi-exchange, cross-exchange strategy:
    Engine(cross_strategy=arb_strategy, config=cfg)
"""

from __future__ import annotations

import copy
import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from core.models import (
    BacktestConfig,
    FillResult,
    Position,
    Side,
    Trade,
    LiveConfig,
    ExchangeCredentials,
)
from core.parser import seconds_to_timeframe
from strategy.sizing import Sizer, SizingContext, default_sizer
from strategy.stops import StopLoss, StopContext, default_stop_loss
from strategy.base import (
    Strategy,
    StrategyContext,
    PortfolioTarget,
)
from core.universe import Universe
from strategy.overlay import PortfolioOverlay

from .executor import BaseExecutor
from .portfolio import MultiExchangePortfolio
from .factory import create_executor, create_feed, create_bar_builder
from .state import _AssetLiveState, LiveState, _ManualKillSwitch, _sizer_config_shim

logger = logging.getLogger(__name__)


class Engine:
    """
    Live trading engine supporting one or many exchanges.

    Single-exchange shorthand:
        Engine(strategy=my_strategy, config=cfg)

    Multi-exchange, independent strategies per exchange:
        Engine(per_exchange_strategies={"binance": s1, "hyperliquid": s2}, config=cfg)

    Multi-exchange, cross-exchange strategy (funding arb, stat arb, hedging):
        Engine(cross_strategy=arb_strategy, config=cfg)
    """

    def __init__(
        self,
        strategy: Strategy | None = None,
        cross_strategy: Strategy | None = None,
        per_exchange_strategies: dict[str, Strategy] | None = None,
        overlay: PortfolioOverlay | None = None,
        config: LiveConfig | None = None,
        sizer: Sizer | dict[str, Sizer] | None = None,
        stop_loss: StopLoss | dict[str, StopLoss] | None = None,
    ):
        n_provided = sum(x is not None for x in [strategy, cross_strategy, per_exchange_strategies])
        if n_provided == 0:
            raise ValueError(
                "Provide one of: strategy=, cross_strategy=, or per_exchange_strategies="
            )
        if n_provided > 1:
            raise ValueError(
                "Provide only one of: strategy=, cross_strategy=, or per_exchange_strategies="
            )

        self.config = config or LiveConfig()
        self._sizer_spec = sizer
        self._stop_loss_spec = stop_loss
        self.overlay = overlay

        self._creds = self.config.get_credentials()
        self._exchange_names = [c.exchange for c in self._creds]
        self._symbols = self.config.active_symbols

        if cross_strategy is not None:
            self._mode = "cross"
            self.cross_strategy = cross_strategy
            self._per_exchange_strategies: dict[str, Strategy] = {}
        else:
            self._mode = "per_exchange"
            self.cross_strategy = None
            if strategy is not None:
                if len(self._creds) != 1:
                    raise ValueError(
                        "strategy= shorthand requires exactly 1 exchange in config.exchanges; "
                        "for multi-exchange use per_exchange_strategies="
                    )
                self._per_exchange_strategies = {self._creds[0].exchange: strategy}
            else:
                self._per_exchange_strategies = per_exchange_strategies or {}

        # Runtime state — all keyed by (exchange, symbol)
        self._executors: dict[str, BaseExecutor] = {}
        self._assets: dict[tuple[str, str], _AssetLiveState] = {}
        self._universes: dict[str, Universe] = {}

        self.portfolio = MultiExchangePortfolio()
        self.state = LiveState()

        # Concurrency: bounded pool + per-exchange pending flag for backpressure
        self._pool: ThreadPoolExecutor | None = None
        self._pending_bars: dict[str, bool] = {}
        self._dedup_lock = threading.Lock()
        self._last_processed_bar: dict[str, int] = {}

        self._running = False
        self._kill_listener = _ManualKillSwitch(self._manual_kill)

        self._run_log_dir: Path | None = None
        self._trade_log_path: Path | None = None

    @property
    def primary_exchange(self) -> str:
        return self._exchange_names[0]

    @property
    def assets(self) -> dict[str, _AssetLiveState]:
        """Public view of per-(exchange, symbol) state with string keys 'exchange:symbol'."""
        return {f"{ex}:{sym}": ast for (ex, sym), ast in self._assets.items()}

    # ── Component resolution ──────────────────────────────────────────────

    def _resolve_sizer(self, symbol: str) -> Sizer:
        if isinstance(self._sizer_spec, dict):
            return copy.deepcopy(self._sizer_spec.get(symbol, default_sizer()))
        if self._sizer_spec is not None:
            return copy.deepcopy(self._sizer_spec)
        return default_sizer()

    def _resolve_stop_loss(self, symbol: str) -> StopLoss:
        if isinstance(self._stop_loss_spec, dict):
            return copy.deepcopy(self._stop_loss_spec.get(symbol, default_stop_loss()))
        if self._stop_loss_spec is not None:
            return copy.deepcopy(self._stop_loss_spec)
        return default_stop_loss()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        exchange_label = "_".join(self._exchange_names)
        self._run_log_dir = Path("logs") / "live" / exchange_label / run_ts
        self._run_log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log_path = self._run_log_dir / self.config.trade_log_csv
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("ENGINE STARTING — %s", self._describe_mode())
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

        # 3. Per-(exchange, symbol): universe, bar builders, initial positions
        for cred in self._creds:
            self._universes[cred.exchange] = Universe(symbols=self._symbols)
            self._pending_bars[cred.exchange] = False
            for sym in self._symbols:
                pos = self._executors[cred.exchange].get_position(sym)
                logger.info("  %s/%s: %s %.4f @ %.4f", cred.exchange, sym, pos.side.name, pos.size, pos.entry_price)
                ast = _AssetLiveState(
                    symbol=sym, exchange=cred.exchange,
                    position=pos,
                    stop_loss=self._resolve_stop_loss(sym),
                )
                ast.bar_builder = create_bar_builder(
                    interval_s=self.config.bar_interval_s,
                    max_bars=self.config.max_bars_in_memory,
                    on_bar_close=lambda data, e=cred.exchange, s=sym: self._on_new_bar(e, s, data),
                )
                self._assets[(cred.exchange, sym)] = ast

        self.state.positions = {
            sym: self._executors[self.primary_exchange].get_position(sym)
            for sym in self._symbols
        }

        # 4. Warm-up historical bars, then do a one-time strategy setup
        self._warmup()
        self._setup_strategies()

        # 5. Start thread pool and feeds
        n_workers = max(4, len(self._exchange_names))
        self._pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="bar-proc")
        self._running = True

        for cred in self._creds:
            for sym in self._symbols:
                ast = self._assets[(cred.exchange, sym)]
                ast.feed = create_feed(
                    exchange=cred.exchange, symbol=sym,
                    testnet=cred.testnet, symbol_map=cred.symbol_map,
                )
                ast.feed.start(
                    on_trade=ast.bar_builder.on_trade,
                    on_candle=ast.bar_builder.on_candle,
                    on_l2=None,
                )

        logger.info(
            "Engine running | %d exchange(s) × %d symbol(s) | pool workers=%d",
            len(self._exchange_names), len(self._symbols), n_workers,
        )
        self._kill_listener.start()
        self._main_loop()

    def _describe_mode(self) -> str:
        if self._mode == "cross":
            return f"CrossExchange: {self.cross_strategy.__class__.__name__}"
        strats = ", ".join(
            f"{ex}={s.__class__.__name__}" for ex, s in self._per_exchange_strategies.items()
        )
        return f"PerExchange: {strats}"

    def stop(self):
        self._running = False
        self._kill_listener.stop()
        for ast in self._assets.values():
            if ast.feed:
                try:
                    ast.feed.stop()
                except Exception:
                    pass
        if self._pool:
            self._pool.shutdown(wait=False)
        logger.info(
            "Engine stopped | trades=%d | closed=%d",
            len(self.state.trades), len(self.state.closed_trades),
        )

    def _manual_kill(self):
        logger.critical("MANUAL KILL SWITCH ACTIVATED — flattening all positions")
        self.state.kill_switch = True
        for ex_name, ex in self._executors.items():
            try:
                count = ex.close_all_positions(cancel_orders=True)
                if count == 0:
                    for sym in self._symbols:
                        try:
                            ex.close_position(sym)
                            ex.cancel_all(sym)
                        except Exception as e:
                            logger.error("close %s/%s failed: %s", ex_name, sym, e)
            except Exception as e:
                logger.error("Emergency flatten on %s failed: %s", ex_name, e)

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
                        if ast is None:
                            continue
                        bc = ast.bar_builder.bar_count if ast.bar_builder else 0
                        pos = ast.position
                        parts.append(f"{ex_name}/{sym}: {bc}bars {pos.side.name} {pos.size:.4f}")

                logger.info("Heartbeat | equity=$%.2f | %s", self.state.equity, " | ".join(parts))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
        finally:
            self.stop()

    # ── Bar processing ────────────────────────────────────────────────────

    def _on_new_bar(self, exchange: str, trigger_symbol: str, data: pd.DataFrame):
        if not self._running or self.state.kill_switch:
            return
        ast = self._assets.get((exchange, trigger_symbol))
        if ast is None or ast.bar_builder is None:
            return
        bc = ast.bar_builder.bar_count
        dedup_key = f"{exchange}:{trigger_symbol}"
        with self._dedup_lock:
            if bc == self._last_processed_bar.get(dedup_key, -1):
                return
            if self._pending_bars.get(exchange, False):
                logger.warning("Bar dropped (exchange busy): %s/%s", exchange, trigger_symbol)
                return
            self._last_processed_bar[dedup_key] = bc
            self._pending_bars[exchange] = True
        self._pool.submit(self._safe_process_bar, exchange, trigger_symbol)

    def _safe_process_bar(self, exchange: str, trigger_symbol: str):
        try:
            self._process_bar(exchange, trigger_symbol)
        except Exception as e:
            logger.error("Bar error [%s/%s]: %s", exchange, trigger_symbol, e, exc_info=True)
        finally:
            self._pending_bars[exchange] = False

    def _process_bar(self, trigger_exchange: str, trigger_symbol: str):
        # 1. Update all universes with latest bars
        for ex_name in self._exchange_names:
            for sym in self._symbols:
                ast = self._assets.get((ex_name, sym))
                if ast and ast.bar_builder:
                    df = ast.bar_builder.to_dataframe()
                    if len(df) >= 2:
                        self._universes[ex_name].update_asset_bars(sym, df)

        # Refresh indicator columns — setup_data() wrote them onto the warmup DataFrame
        # which update_asset_bars() just replaced with a new object.
        self._refresh_indicators()

        trigger_df = self._assets[(trigger_exchange, trigger_symbol)].bar_builder.to_dataframe()
        if len(trigger_df) < 2:
            return
        ts = (
            trigger_df.index[-1]
            if isinstance(trigger_df.index, pd.DatetimeIndex)
            else pd.Timestamp.now()
        )

        # 2. Refresh equity
        self.portfolio.refresh_equity()
        self.state.equity = self.portfolio.total_equity()
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

        # 3. Kill switch check
        if self._check_kill_switch():
            return

        # 4. Sync positions from all exchanges + MTM unrealized PnL
        all_positions: dict[str, dict[str, Position]] = {}
        for ex_name in self._exchange_names:
            ex = self._executors[ex_name]
            ex_pos: dict[str, Position] = {}
            for sym in self._symbols:
                pos = ex.get_position(sym)
                ex_pos[sym] = pos
                key = (ex_name, sym)
                if key in self._assets:
                    local_pos = self._assets[key].position
                    if pos.side != local_pos.side or abs(pos.size - local_pos.size) > 1e-8:
                        logger.warning(
                            "%s/%s position mismatch — local: %s %.4f, exchange: %s %.4f",
                            ex_name, sym, local_pos.side.name, local_pos.size,
                            pos.side.name, pos.size,
                        )
                    self._assets[key].position = pos
                    if pos.side != Side.FLAT and pos.size > 0:
                        price = self._assets[key].bar_builder.last_close
                        if not np.isnan(price):
                            direction = 1 if pos.side == Side.LONG else -1
                            pos.unrealized_pnl = (price - pos.entry_price) * pos.size * direction
            all_positions[ex_name] = ex_pos
        self.state.positions = all_positions.get(self.primary_exchange, {})

        # 5. Stop-loss checks
        for ex_name in self._exchange_names:
            executor = self._executors[ex_name]
            for sym in self._symbols:
                key = (ex_name, sym)
                ast = self._assets.get(key)
                if ast is None or ast.position.side == Side.FLAT:
                    continue
                try:
                    df = self._universes[ex_name].ohlcv(sym)
                except KeyError:
                    continue
                if len(df) < 2:
                    continue
                triggered, fill, reason = self._check_asset_stop(ast, df, executor)
                if triggered and fill and fill.success:
                    self._record_trade(ex_name, sym, ast.position, fill, ts, reason)
                    ast.position = Position()
                    ast.open_trade = None
                    ast.stop_loss.reset()
                    all_positions.setdefault(ex_name, {})[sym] = Position()
                    if ex_name == self.primary_exchange:
                        self.state.positions[sym] = Position()

        # 6. Generate targets
        merged_target = self._generate_targets(trigger_exchange, trigger_symbol, ts, all_positions)
        if merged_target is None:
            return

        # 7. Apply overlay
        if self.overlay:
            cross_ctx = self._build_cross_ctx(ts, all_positions)
            merged_target = self.overlay.adjust(merged_target, cross_ctx)

        # 8. Execute
        self._execute_target(merged_target, ts, all_positions)

    # ── Stop-loss check (shared across all exchange/symbol pairs) ─────────

    def _check_asset_stop(
        self,
        ast: _AssetLiveState,
        df: pd.DataFrame,
        executor: BaseExecutor,
    ) -> tuple[bool, FillResult | None, str]:
        """Check and execute stop-loss for one asset. Returns (triggered, fill, reason)."""
        pos = ast.position
        idx = len(df) - 1
        price = df["close"].iat[idx]
        l2_snap = ast.feed.latest_l2 if ast.feed else None
        bar_dict = {c: df[c].iat[idx] for c in df.columns if np.isscalar(df[c].iat[idx])}

        try:
            funding_snap = executor.fetch_funding_rate(ast.symbol)
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

        if not stop_result.triggered:
            return False, None, ""

        logger.info("STOP TRIGGERED on %s/%s: %s", ast.exchange, ast.symbol, stop_result.reason)
        close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
        fill = executor.market_order(ast.symbol, close_side, pos.size, reduce_only=True)
        return True, fill, stop_result.reason

    # ── Target generation ─────────────────────────────────────────────────

    def _generate_targets(
        self, trigger_exchange, trigger_symbol, ts, all_positions
    ) -> PortfolioTarget | None:
        if self._mode == "cross":
            return self._generate_cross(ts, all_positions)
        return self._generate_per_exchange(trigger_exchange, ts, all_positions)

    def _generate_cross(self, ts, all_positions) -> PortfolioTarget | None:
        ctx = self._build_cross_ctx(ts, all_positions)
        try:
            return self.cross_strategy.generate(ctx)
        except Exception as e:
            logger.error("cross strategy generate() failed at %s: %s", ts, e, exc_info=True)
            return None

    def _generate_per_exchange(
        self, trigger_exchange, ts, all_positions
    ) -> PortfolioTarget | None:
        per_exchange_targets: dict[str, PortfolioTarget] = {}
        for ex_name, strat in self._per_exchange_strategies.items():
            universe = self._universes.get(ex_name)
            if universe is None:
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
            try:
                per_exchange_targets[ex_name] = strat.generate(ctx)
            except Exception as e:
                logger.error(
                    "%s strategy generate() failed at %s: %s", ex_name, ts, e, exc_info=True
                )
        return PortfolioTarget.from_exchange_targets(per_exchange_targets)

    def _build_cross_ctx(self, ts, all_positions) -> StrategyContext:
        bar_idx = 0
        for u in self._universes.values():
            for sym in self._symbols:
                try:
                    bar_idx = max(bar_idx, len(u.ohlcv(sym)) - 1)
                except KeyError:
                    pass
        return StrategyContext(
            universes=self._universes,
            bar_idx=bar_idx,
            timestamp=ts,
            equity_by_exchange=self.portfolio.equity_breakdown(),
            all_positions=all_positions,
            trade_history=self.state.closed_trades,
        )

    # ── Execution ─────────────────────────────────────────────────────────

    def _execute_target(self, target: PortfolioTarget, ts, all_positions):
        # Phase 1: closes
        for ex_name in self._exchange_names:
            executor = self._executors[ex_name]
            for sym in self._symbols:
                key = (ex_name, sym)
                ast = self._assets.get(key)
                if ast is None or ast.position.side == Side.FLAT:
                    continue
                pos = ast.position
                desired = target[(ex_name, sym)]
                if not (desired.side == Side.FLAT or desired.side != pos.side):
                    continue
                reason = desired.reason or "target_flat"
                logger.info("CLOSING %s %s %.4f on %s — %s", pos.side.name, sym, pos.size, ex_name, reason)
                fill = self._execute_close(executor, sym, pos)
                if fill.success:
                    self._record_trade(ex_name, sym, pos, fill, ts, reason)
                    ast.position = Position()
                    ast.open_trade = None
                    ast.stop_loss.reset()
                    if ex_name == self.primary_exchange:
                        self.state.positions[sym] = Position()

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

                # Gather bar context for sizer
                df = None
                idx = 0
                l2_snap = None
                bar_dict: dict = {}
                universe = self._universes.get(ex_name)
                if universe:
                    try:
                        df = universe.ohlcv(sym)
                        idx = len(df) - 1
                        l2_snap = ast.feed.latest_l2 if ast.feed else None
                        bar_dict = {c: df[c].iat[idx] for c in df.columns if np.isscalar(df[c].iat[idx])}
                        try:
                            fs = executor.fetch_funding_rate(sym)
                            if fs is not None:
                                bar_dict["funding_rate"] = fs.rate
                                bar_dict["funding_rate_ann_bps"] = fs.rate_annualized
                                if fs.oracle_price > 0:
                                    bar_dict["oracle_price"] = fs.oracle_price
                                if fs.mark_price > 0:
                                    bar_dict["mark_price"] = fs.mark_price
                        except Exception:
                            pass
                    except (KeyError, Exception):
                        pass

                # Size: start from weight-based notional, cap at max_position_pct for single-exchange
                max_notional = self.state.equity * alloc.weight * self.config.leverage
                if len(self._per_exchange_strategies) <= 1 and self._mode == "per_exchange":
                    max_notional = min(
                        max_notional,
                        self.state.equity * self.config.max_position_pct * self.config.leverage,
                    )
                size = max_notional / price if price > 0 else 0.0

                if df is not None and len(df) >= 2:
                    try:
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
                    except Exception as e:
                        logger.warning("Sizer failed for %s/%s: %s", ex_name, sym, e)

                if size <= 0:
                    continue

                logger.info(
                    "OPENING %s %.4f %s on %s @ ~%.4f (w=%.2f)",
                    alloc.side.name, size, sym, ex_name, price, alloc.weight,
                )
                fill = self._execute_open(executor, sym, alloc.side, size)
                if fill.success:
                    new_pos = Position(
                        side=alloc.side,
                        size=fill.filled_size or size,
                        entry_price=fill.fill_price or price,
                        entry_timestamp=ts,
                    )
                    ast.position = new_pos
                    if ex_name == self.primary_exchange:
                        self.state.positions[sym] = new_pos

                    if df is not None and len(df) >= 2:
                        try:
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
                        self.cross_strategy.on_fill(sym, alloc.side, fill.filled_size or size, fill.fill_price or price, exchange=ex_name)
                    elif ex_name in self._per_exchange_strategies:
                        self._per_exchange_strategies[ex_name].on_fill(sym, alloc.side, fill.filled_size or size, fill.fill_price or price, exchange=ex_name)

    # ── Execution helpers ─────────────────────────────────────────────────

    def _execute_open(self, executor: BaseExecutor, symbol: str, side: Side, size: float) -> FillResult:
        if self.config.order_type == "limit":
            mid = executor.get_mid_price(symbol)
            offset = mid * self.config.limit_chase_bps / 1e4
            px = mid + offset if side == Side.LONG else mid - offset
            return executor.limit_order(symbol, side, size, px)
        return executor.market_order(symbol, side, size)

    def _execute_close(self, executor: BaseExecutor, symbol: str, pos: Position) -> FillResult:
        close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
        return executor.market_order(symbol, close_side, pos.size, reduce_only=self.config.reduce_only_exits)

    # ── Trade recording ───────────────────────────────────────────────────

    def _record_trade(self, exchange, symbol, pos, fill, ts, reason):
        key = (exchange, symbol)
        ast = self._assets.get(key)
        exit_price = fill.fill_price if fill.fill_price > 0 else (
            ast.bar_builder.last_close if ast and ast.bar_builder else pos.entry_price
        )
        direction = 1 if pos.side == Side.LONG else -1
        pnl = (exit_price - pos.entry_price) * pos.size * direction

        if ast and ast.open_trade and ast.open_trade.exit_price is None:
            t = ast.open_trade
            t.exit_price = exit_price
            t.exit_timestamp = ts
            t.pnl = pnl
            t.pnl_pct = pnl / (pos.entry_price * pos.size) if pos.entry_price * pos.size > 0 else 0.0
            # Preserve entry fees already recorded on the trade; do not zero them out.
            # FillResult carries no fee field — exchange-reported fees would need to
            # come from the executor's order status query and can be added here when
            # the executor supports it.
            t.reason_exit = reason
            t.meta["symbol"] = symbol
            t.meta["exchange"] = exchange
            self.state.closed_trades.append(t)

        self.state.daily_pnl += pnl
        self.state.equity = self.portfolio.total_equity()
        logger.info(
            "TRADE CLOSED: %s %s %.4f on %s | entry=%.4f exit=%.4f | pnl=$%.2f | %s",
            pos.side.name, symbol, pos.size, exchange,
            pos.entry_price, exit_price, pnl, reason,
        )
        if self.state.closed_trades:
            self._write_trade_csv(self.state.closed_trades[-1])

    def _write_trade_csv(self, trade):
        if trade is None or self._trade_log_path is None:
            return
        file_exists = self._trade_log_path.exists()
        row = trade.to_dict()
        with open(self._trade_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ── Risk management ───────────────────────────────────────────────────

    def _check_kill_switch(self) -> bool:
        if self.state.check_daily_loss_limit(self.config):
            self.portfolio.flatten_all(self._symbols)
            self.state.kill_switch = True
            return True
        return False

    # ── Warm-up ───────────────────────────────────────────────────────────

    def _warmup(self):
        try:
            bar_tf = seconds_to_timeframe(self.config.bar_interval_s)
        except ValueError:
            bar_tf = "1m"
            logger.warning(
                "bar_interval_s=%d has no exact timeframe label; fetching warmup as 1m",
                self.config.bar_interval_s,
            )
        for cred in self._creds:
            executor = self._executors[cred.exchange]
            for sym in self._symbols:
                logger.info("Fetching %d warmup bars for %s/%s (%s)...",
                            self.config.warmup_bars, cred.exchange, sym, bar_tf)
                try:
                    now_ms = int(time.time() * 1000)
                    start_ms = now_ms - self.config.warmup_bars * self.config.bar_interval_s * 1000
                    rows = executor.fetch_historical_candles(sym, bar_tf, start_ms, now_ms)
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

    def _refresh_indicators(self):
        """Re-apply setup_data() so indicator columns survive update_asset_bars() replacement."""
        if self._mode == "cross":
            strat = self.cross_strategy
            if not hasattr(strat, "setup_data"):
                return
            for ex_name in self._exchange_names:
                universe = self._universes.get(ex_name)
                if universe is None:
                    continue
                for sym in self._symbols:
                    try:
                        strat.setup_data(universe.ohlcv(sym), universe.l2(sym))
                    except Exception:
                        pass
        else:
            for ex_name, strat in self._per_exchange_strategies.items():
                if not hasattr(strat, "setup_data"):
                    continue
                universe = self._universes.get(ex_name)
                if universe is None:
                    continue
                for sym in self._symbols:
                    try:
                        strat.setup_data(universe.ohlcv(sym), universe.l2(sym))
                    except Exception:
                        pass

    def _setup_strategies(self):
        """Call strategy.setup() once after warmup, not on every bar."""
        if self._mode == "cross":
            try:
                self.cross_strategy.setup(self._universes)
            except Exception as e:
                logger.error("CrossStrategy initial setup failed: %s", e)
        else:
            for ex_name, strat in self._per_exchange_strategies.items():
                universe = self._universes.get(ex_name)
                if universe is None:
                    continue
                try:
                    strat.setup(universe)
                except Exception as e:
                    logger.error("%s strategy initial setup failed: %s", ex_name, e)

    def _setup_logging(self):
        log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format=log_fmt,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self._run_log_dir / "engine.log", mode="a"),
            ],
        )
