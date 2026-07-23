"""
Option-chain half of the Market tab: IV surface, smile, term structure and Greeks.

Inverts every live contract's bid/ask mid under Black-Scholes to an implied volatility,
then draws the smile, term structure, 3-D surface and Greeks off that surface. Mid is used
rather than the last traded price because a last print can be hours stale against a live
spot, which pushes deep-ITM quotes below intrinsic and leaves holes in the surface.

The underlying is whichever asset is selected in the Market tab, so the price charts and
the vol surface above/below each other always describe the same instrument.

Split in two because the controls live in the sidebar and the output in the tab:
`sidebar_controls()` renders the inputs and returns state, `render_section()` draws from it.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from components import ui
from components.charts import (
    greeks_chart, iv_smile_chart, iv_surface_chart, term_structure_chart,
)
from components.lse_data import fetch_option_underlyings, load_chain_cached
from core.derivatives import IVSurface


@st.cache_data(show_spinner="Loading LSE options catalogue…")
def _option_symbols(api_key: str) -> set[str]:
    """Symbols with listed options on LSE. Cached — one API call per session."""
    return {row.get("symbol", "") for row in (fetch_option_underlyings(api_key) or [])}


def sidebar_controls(symbol: str | None, api_key: str) -> dict:
    """
    Render the chain controls into the current container (the app sidebar).

    `symbol` is the asset selected in the Market tab — there is no separate underlying
    picker, so price and option views always describe the same instrument.

    Returns the state `render_section()` consumes. `status` is "ok" when a chain can be
    built; otherwise it names why not, so the tab body can show a matching empty state
    instead of the sidebar shouting about it.
    """
    if not symbol:
        st.caption("Load a universe to enable.")
        return {"status": "no_universe"}

    listed = _option_symbols(api_key)
    if not listed:
        st.caption("Options catalogue unavailable.")
        return {"status": "no_catalogue"}

    if symbol not in listed:
        st.caption(f"{symbol} has no listed options.")
        return {"status": "no_options", "symbol": symbol}

    st.caption(f"Underlying: **{symbol}**")

    today = date.today()
    one_month_ahead = today + timedelta(days=30)
    exp_range = st.date_input(
        "Expiry window",
        value=(today, one_month_ahead),
        min_value=today,
        key="opt_expiry",
    )
    if isinstance(exp_range, (list, tuple)) and len(exp_range) == 2:
        d_from, d_to = exp_range
    else:  # user mid-selection: single date → treat as the upper bound
        d_from, d_to = today, (
            exp_range if isinstance(exp_range, date) else one_month_ahead
        )

    _rc, _qc = st.columns(2)
    r_bps = _rc.number_input("Rate (bps)", value=400, step=25, min_value=0, key="opt_r")
    q_bps = _qc.number_input("Div. (bps)", value=0, step=25, min_value=0, key="opt_q")

    load = st.button(f"Load {symbol} chain", type="primary", use_container_width=True,
                     key="opt_load")

    return {
        "status": "ok",
        "symbol": symbol,
        "min_dte": max((d_from - today).days, 0),
        "max_dte": max((d_to - today).days, max((d_from - today).days, 0)),
        "r_bps": r_bps, "q_bps": q_bps,
        "r": r_bps / 1e4, "q": q_bps / 1e4,
        "load": load,
    }


def render_section(state: dict, api_key: str) -> dict | None:
    """
    Render the option-chain half of the Market tab from `sidebar_controls()` state.

    A section rather than a tab: it sits under the price and indicator charts for the
    same asset, so one screen covers spot and vol together.

    Returns `{"underlying", "table"}` for the built chain, or None when there is
    nothing to show. The caller renders that table under its own "Data" heading, next
    to the other tables and downloads, rather than stranding it up here.
    """
    ui.section(
        "Option chain",
        "Black-Scholes implied volatility inverted from the bid/ask mid of every live "
        "contract — the smile, term structure and Greeks all read off that surface.",
    )

    status = state.get("status")
    if status == "no_universe":
        ui.empty_state("Load one or more assets from the sidebar first.")
        return None
    if status == "no_catalogue":
        st.error("Could not load the LSE options catalogue. Check connectivity, then reload.")
        return None
    if status == "no_options":
        ui.empty_state(
            f"**{state.get('symbol', 'This asset')}** has no listed options on LSE — "
            "select an asset with a listed chain to build a surface."
        )
        return None

    symbol = state["symbol"]
    min_dte, max_dte = state["min_dte"], state["max_dte"]
    r, q = state["r"], state["q"]
    r_bps, q_bps = state["r_bps"], state["q_bps"]

    # ── Chain acquisition ────────────────────────────────────────────────────
    # Keyed by symbol: the picker switches underlyings without reloading the chain.
    chain_key = f"_options_chain_{symbol}"
    if state["load"]:
        st.session_state[chain_key] = load_chain_cached(
            symbol, min_dte=min_dte, max_dte=max_dte, api_key=api_key,
        )
        # A reload brings fresh chain data, so every cached surface is now stale — including
        # one a settings-change rerun may have just built from the *old* chain under the key
        # this reload will land on. Drop them all so the surface re-fits from the new chain
        # instead of returning that stale entry.
        for _k in [k for k in st.session_state if str(k).startswith("_surf_")]:
            del st.session_state[_k]
    chain = st.session_state.get(chain_key)

    if chain is None or not chain.contracts:
        ui.empty_state(
            f"Set the expiry window in the sidebar, then press **Load {symbol} chain**."
        )
        return None

    # Providers keep expired contracts in the "current" chain, frozen at their last traded
    # state. Time to expiry is zero, so nothing can be reconstructed from them — drop them
    # rather than rendering rows of blank IV and Greeks.
    n_expired = chain.n_expired
    if n_expired:
        chain = chain.drop_expired()
    if not chain.contracts:
        ui.verdict("warn", f"All {n_expired} contracts in this chain have already expired — "
                           "the provider has no live chain for this underlying.")
        return None

    # ── Reconstruct surface (cached by parameter signature) ──────────────────
    surf_key = (
        f"_surf_{chain.underlying}_{min_dte}_{max_dte}"
        f"_{r_bps}_{q_bps}"
    )
    if surf_key in st.session_state:
        surface = st.session_state[surf_key]
    else:
        try:
            with st.spinner("Reconstructing IV surface…"):
                # Black-Scholes inversion of the bid/ask mid — the only mode the tab
                # offers, so the surface always reflects market quotes rather than a
                # fitted model, and never a stale last-traded print.
                surface = IVSurface.from_chain(
                    chain, r=r, q=q, price="mid",
                )
            st.session_state[surf_key] = surface
        except (ValueError, RuntimeError) as e:
            st.error(f"Could not build the IV surface: {e}")
            return None

    gdf = surface.iv_df
    n_total = len(chain.contracts)
    n_valid = int(gdf["iv"].notna().sum())

    # Headline vol numbers lead: what the surface says comes before how it is drawn.
    # The underlying is not repeated here — the asset picker above already names it.
    _atm = surface.atm_vol(surface.expiries[0]) if surface.expiries else float("nan")
    _skew = surface.skew(surface.expiries[0]) if surface.expiries else float("nan")
    ui.metric_row([
        {"label": "Spot", "value": f"{chain.spot:,.2f}" if pd.notna(chain.spot) else "—",
         "help": "Live underlying price from the option feed — not the last bar close."},
        {"label": "Front ATM IV", "value": f"{_atm * 100:.1f}%" if pd.notna(_atm) else "—",
         "help": "At-the-forward implied vol (log-moneyness 0) of the nearest expiry."},
        {"label": "Front skew ±0.10", "value": f"{_skew * 100:.1f}%" if pd.notna(_skew) else "—",
         "help": "IV at −0.1 log-moneyness minus IV at +0.1 (put-side minus call-side). "
                 "Positive means downside is bid."},
        {"label": "Contracts", "value": f"{n_total:,}",
         "delta": f"−{n_expired} expired" if n_expired else None,
         "help": "Live contracts in the requested expiry window."},
        {"label": "Expiries", "value": f"{len(surface.expiries)}"},
    ])

    # One quality line rather than a metric tile: how many quotes inverted, and why
    # the rest did not. Every blank IV has a reason, so none of them read as a bug.
    _quality = f"{n_valid:,}/{n_total:,} quotes inverted to an implied vol."
    if "iv_status" in gdf.columns:
        _unsolved = gdf.loc[gdf["iv"].isna(), "iv_status"]
        if len(_unsolved):
            _reasons = ", ".join(f"{n} {why}" for why, n in _unsolved.value_counts().items())
            _quality += f" Skipped — {_reasons} (see `iv_status` in the chain table)."
    if surface.svi:
        _rmse = pd.Series([p.rmse for p in surface.svi.values()]).median()
        _quality += (f" SVI smiles fit on {len(surface.svi)}/{len(surface.expiries)} expiries "
                     f"(median fit error {_rmse * 100:.2f} vol pts).")
    st.caption(_quality)

    # ── Surface + term structure ─────────────────────────────────────────────
    # Equal heights: with the ATM/skew tiles moved up into the headline row, the
    # right column is a single chart and the two columns line up on their own.
    _CHART_H = 520
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(iv_surface_chart(surface, height=_CHART_H), use_container_width=True)
    with right:
        st.plotly_chart(term_structure_chart(surface, height=_CHART_H),
                        use_container_width=True)

    st.plotly_chart(iv_smile_chart(surface), use_container_width=True)

    # ── Greeks per expiry ────────────────────────────────────────────────────
    # Only offer expiries with reconstructed data (skips e.g. 0-DTE rows that don't solve).
    solved_expiries = surface.expiries or []
    if solved_expiries:
        exp_labels = [pd.Timestamp(e).strftime("%Y-%m-%d") for e in solved_expiries]
        # Selector beside its heading rather than full width above the chart, so it
        # reads as "Greeks for <expiry>" instead of a stray dropdown.
        _gh, _gs = st.columns([3, 1])
        with _gh:
            ui.section("Greeks by strike")
        sel = _gs.selectbox("Expiry", exp_labels, index=0, key="opt_greeks_expiry",
                            label_visibility="collapsed")
        st.plotly_chart(greeks_chart(gdf, expiry=pd.Timestamp(sel)), use_container_width=True)
    else:
        ui.section("Greeks by strike")
        ui.empty_state("No expiries with solvable IV in this window.")

    # ── Chain table, handed to the caller ────────────────────────────────────
    show = gdf.copy()
    show["iv_%"] = show["iv"] * 100.0
    cols = ["expiry", "strike", "option_type", "mid", "last", "underlying_price",
            "T", "iv_%", "iv_status", "delta", "gamma", "vega", "theta", "rho",
            "volume", "open_interest"]
    cols = [c for c in cols if c in show.columns]
    return {"underlying": chain.underlying, "table": show[cols]}


def chain_table_expander(built: dict) -> None:
    """Chain table + CSV download, rendered wherever the caller groups its data."""
    table = built["table"]
    with st.expander(f"Chain table — {len(table):,} contracts, with IV & Greeks"):
        st.dataframe(table, use_container_width=True, height=420)
        st.download_button(
            "Download chain CSV", table.to_csv(index=False).encode(),
            file_name=f"{built['underlying']}_chain.csv", mime="text/csv",
        )
