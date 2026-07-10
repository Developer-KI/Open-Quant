"""Streamlit form widgets that return ready-to-use strategy/risk objects."""
from __future__ import annotations

import inspect

import streamlit as st


# ── Helpers ───────────────────────────────────────────────────────────────────

def _num(container, label: str, value: float, key: str,
         step: float = 0.01, min_val: float | None = None, max_val: float | None = None) -> float:
    return float(container.number_input(
        label, value=float(value), step=step,
        min_value=min_val, max_value=max_val, key=key,
    ))


@st.cache_resource
def _load_signal_modules():
    """Import known strategy modules once so their @register_strategy decorators fire."""
    for mod in [
        "strategy.built_in",
        "trading.backtest_demo",
        "trading.alpaca_livetest_demo",
    ]:
        try:
            __import__(mod)
        except Exception:
            pass


# ── Signal form ───────────────────────────────────────────────────────────────

def signal_form(container, key_prefix: str = "sig") -> tuple[type | None, dict]:
    """
    Strategy selector + auto-generated param fields.
    Returns (strategy_cls, params_dict) or (None, {}) if no strategies registered.
    """
    _load_signal_modules()
    from strategy.base import list_strategies, get_strategy

    signals = list_strategies()
    if not signals:
        container.warning("No strategies registered. Add strategies with @register_strategy.")
        return None, {}

    name = container.selectbox("Signal", signals, key=f"{key_prefix}_name")
    signal_cls = get_strategy(name)

    # Reflect constructor defaults for simple scalar params
    sig = inspect.signature(signal_cls.__init__)
    defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if k not in ("self",) and v.default is not inspect.Parameter.empty
    }
    simple_defaults = {k: v for k, v in defaults.items() if isinstance(v, (int, float, str, bool))}

    params: dict = {}
    if simple_defaults:
        container.markdown("**Signal parameters**")
        for k, default in simple_defaults.items():
            wkey = f"{key_prefix}_p_{k}"
            if isinstance(default, bool):
                params[k] = container.checkbox(k, value=default, key=wkey)
            elif isinstance(default, int):
                params[k] = int(_num(container, k, float(default), wkey, step=1.0))
            elif isinstance(default, float):
                params[k] = _num(container, k, default, wkey, step=0.01)
            else:
                params[k] = container.text_input(k, value=str(default), key=wkey)

    return signal_cls, params


# ── Sizer form ────────────────────────────────────────────────────────────────

def sizer_form(container, key_prefix: str = "sizer"):
    """
    Sizer selector + param fields.
    Returns an instantiated Sizer.
    """
    from strategy.sizing import (
        FixedFractionalSizer, FixedNotionalSizer,
        VolatilityTargetSizer, KellySizer, CompositeSizer,
    )

    OPTIONS = ["Fixed Fractional", "Fixed Notional", "Volatility Target", "Kelly", "Vol + Kelly"]
    choice = container.radio("Sizer", OPTIONS, key=f"{key_prefix}_choice")

    if choice == "Fixed Fractional":
        rf = _num(container, "Risk fraction", 0.02, f"{key_prefix}_rf", step=0.005, min_val=0.001, max_val=0.5)
        return FixedFractionalSizer(risk_frac=rf)

    elif choice == "Fixed Notional":
        use_fixed = container.checkbox("Fixed $ notional (not equity %)", value=False, key=f"{key_prefix}_use_notional")
        if use_fixed:
            notional = _num(container, "Notional ($)", 1000.0, f"{key_prefix}_notional", step=100.0, min_val=10.0)
            return FixedNotionalSizer(notional=notional)
        else:
            pct = _num(container, "Equity %", 0.10, f"{key_prefix}_ep", step=0.01, min_val=0.01, max_val=1.0)
            return FixedNotionalSizer(equity_pct=pct)

    elif choice == "Volatility Target":
        tv = _num(container, "Target annual vol", 0.15, f"{key_prefix}_tv", step=0.01, min_val=0.01)
        lb = int(_num(container, "Lookback bars", 20.0, f"{key_prefix}_lb", step=1.0, min_val=5.0))
        return VolatilityTargetSizer(target_vol=tv, lookback=lb)

    elif choice == "Kelly":
        kf = _num(container, "Kelly fraction", 0.5, f"{key_prefix}_kf", step=0.1, min_val=0.1, max_val=1.0)
        mt = int(_num(container, "Min trades for Kelly", 20.0, f"{key_prefix}_mt", step=1.0, min_val=5.0))
        return KellySizer(kelly_frac=kf, min_trades=mt)

    else:  # Vol + Kelly
        tv = _num(container, "Target vol", 0.15, f"{key_prefix}_cvtv", step=0.01)
        kf = _num(container, "Kelly fraction", 0.5, f"{key_prefix}_cvkf", step=0.1)
        mode = container.selectbox("Combine mode", ["avg", "min", "max"], key=f"{key_prefix}_cvmode")
        return CompositeSizer(
            sizers=[VolatilityTargetSizer(target_vol=tv), KellySizer(kelly_frac=kf)],
            mode=mode,
        )


# ── Stop form ─────────────────────────────────────────────────────────────────

def stop_form(container, key_prefix: str = "stop"):
    """
    Stop-loss selector + param fields.
    Returns an instantiated StopLoss.
    """
    from strategy.stops import FixedPercentStop, ATRStop, TrailingStop, RiskRewardStop

    OPTIONS = ["Fixed Percent", "ATR", "Trailing", "Risk/Reward"]
    choice = container.radio("Stop loss", OPTIONS, key=f"{key_prefix}_choice")

    if choice == "Fixed Percent":
        sl = _num(container, "SL %", 2.0, f"{key_prefix}_sl", step=0.1, min_val=0.1)
        use_tp = container.checkbox("Set take profit", value=False, key=f"{key_prefix}_use_tp")
        tp = _num(container, "TP %", 4.0, f"{key_prefix}_tp", step=0.1, min_val=0.1) if use_tp else None
        return FixedPercentStop(sl_pct=sl, tp_pct=tp)

    elif choice == "ATR":
        asl = _num(container, "ATR mult (SL)", 2.0, f"{key_prefix}_asl", step=0.5, min_val=0.5)
        atp = _num(container, "ATR mult (TP)", 3.0, f"{key_prefix}_atp", step=0.5, min_val=0.5)
        return ATRStop(atr_mult_sl=asl, atr_mult_tp=atp)

    elif choice == "Trailing":
        trail = _num(container, "Trail %", 2.0, f"{key_prefix}_trail", step=0.1, min_val=0.1)
        act = _num(container, "Activation profit %", 0.0, f"{key_prefix}_act", step=0.1, min_val=0.0)
        return TrailingStop(trail_pct=trail, activation_pct=act)

    else:  # Risk/Reward
        sl = _num(container, "SL %", 1.5, f"{key_prefix}_rrsl", step=0.1, min_val=0.1)
        rr = _num(container, "R/R ratio", 2.0, f"{key_prefix}_rr", step=0.5, min_val=0.5)
        return RiskRewardStop(sl_pct=sl, rr_ratio=rr)


# ── Backtest config form ──────────────────────────────────────────────────────

def backtest_config_form(container, key_prefix: str = "btcfg"):
    """BacktestConfig form. Returns a BacktestConfig instance."""
    from core.models import BacktestConfig

    container.markdown("**Capital & risk**")
    cap = _num(container, "Initial capital ($)", 10_000.0, f"{key_prefix}_cap", step=1000.0, min_val=100.0)
    mpp = _num(container, "Max position %", 0.25, f"{key_prefix}_mpp", step=0.05, min_val=0.01, max_val=1.0)
    lev = _num(container, "Leverage", 1.0, f"{key_prefix}_lev", step=0.5, min_val=1.0, max_val=20.0)

    return BacktestConfig(
        initial_capital=cap,
        max_position_pct=mpp,
        leverage=lev,
    )
