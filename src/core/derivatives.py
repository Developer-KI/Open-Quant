"""
core/derivatives.py — Options analytics: pricing, implied vol, Greeks, IV surface.

Self-contained derivatives layer. Given an options chain for an underlying, this module
reconstructs the implied volatility of every contract, computes the full Greek set, and
builds an interpolable IV surface (smile + term structure + 3-D grid).

Design mirrors indicators.py / sizing.py: small stateless functions plus a few dataclasses,
importing numpy + scipy. scipy is a hard dependency (declared in pyproject): it supplies the
vectorized normal CDF, the Brent root-find, and the grid interpolation, all of which have no
equal-quality pure-numpy substitute — so there are no fallbacks.

Pricing model:
  • Black-Scholes (European), continuous dividend yield q — vectorized.

Greek conventions (raw, per unit — scale in the UI as needed):
  • delta : ∂V/∂S            (per $1 of underlying)
  • gamma : ∂²V/∂S²
  • vega  : ∂V/∂σ            (per 1.00 = 100 vol points; divide by 100 for per 1%)
  • theta : ∂V/∂t            (per year; divide by 365 for per calendar day)
  • rho   : ∂V/∂r            (per 1.00 = 100 rate points; divide by 100 for per 1%)
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr as _ndtr  # vectorized standard-normal CDF
from scipy.optimize import brentq as _brentq
from scipy.optimize import least_squares as _least_squares

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_YEAR_SECONDS = 365.25 * 24 * 3600  # calendar-year fraction denominator


def _to_utc(ts) -> pd.Timestamp:
    """Coerce a timestamp to UTC-aware (assume UTC when tz-naive)."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ── Standard-normal helpers ───────────────────────────────────────────────────

def _norm_cdf(x):
    """Standard-normal CDF, vectorized (scipy `ndtr`)."""
    return _ndtr(x)


def _norm_pdf(x):
    """Standard-normal PDF, vectorized (pure numpy)."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


# ── Data models ───────────────────────────────────────────────────────────────

class OptionType(enum.Enum):
    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: "OptionType | str") -> "OptionType":
        if isinstance(value, cls):
            return value
        s = str(value).strip().lower()
        if s in ("c", "call"):
            return cls.CALL
        if s in ("p", "put"):
            return cls.PUT
        raise ValueError(f"Unrecognized option type: {value!r}")


@dataclass
class OptionContract:
    """A single option quote/observation."""

    underlying: str
    expiry: pd.Timestamp
    strike: float
    option_type: OptionType
    bid: float = np.nan
    ask: float = np.nan
    last: float = np.nan
    volume: float = 0.0
    open_interest: float = 0.0
    underlying_price: float = np.nan
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mid(self) -> float:
        """Mid of bid/ask; falls back to last when either side is missing."""
        if np.isfinite(self.bid) and np.isfinite(self.ask) and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    def price(self, basis: str = "mid") -> float:
        return self.last if basis == "last" else self.mid


# Common column-name aliases for provider-agnostic ingestion.
_FIELD_ALIASES = {
    "expiry": ["expiry", "expiration", "expiration_date", "expiry_date", "exp"],
    "strike": ["strike", "strike_price", "k"],
    "option_type": ["option_type", "contract_type", "type", "right", "call_put", "cp"],
    "bid": ["bid", "bid_price"],
    "ask": ["ask", "ask_price", "offer"],
    "last": ["last", "last_price", "price", "close", "mark"],
    "volume": ["volume", "volume_today", "vol"],
    "open_interest": ["open_interest", "oi", "openinterest"],
    # note: bare "underlying" is deliberately excluded — some feeds put the ticker there.
    "underlying_price": ["underlying_price", "underlying_last", "underlying_close",
                          "spot_price", "spot"],
}


def _pick(row: dict, keys: list[str], default=np.nan):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


@dataclass
class OptionChain:
    """A snapshot of the option chain for one underlying at one point in time."""

    underlying: str
    asof: pd.Timestamp
    contracts: list[OptionContract] = field(default_factory=list)

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def from_records(
        cls,
        records: list[dict],
        underlying: str,
        asof: pd.Timestamp | str | None = None,
    ) -> "OptionChain":
        """Build from a list of dicts (e.g. LSE `client.options()` rows). Tolerant of
        varying provider field names via `_FIELD_ALIASES`."""
        asof_ts = pd.Timestamp(asof) if asof is not None else pd.Timestamp.utcnow()
        contracts: list[OptionContract] = []
        for row in records:
            try:
                otype = OptionType.parse(_pick(row, _FIELD_ALIASES["option_type"]))
            except (ValueError, KeyError):
                continue
            expiry = _pick(row, _FIELD_ALIASES["expiry"])
            strike = _pick(row, _FIELD_ALIASES["strike"])
            if expiry is None or not np.isfinite(float(strike if strike is not None else np.nan)):
                continue
            contracts.append(OptionContract(
                underlying=underlying,
                expiry=pd.Timestamp(expiry),
                strike=float(strike),
                option_type=otype,
                bid=float(_pick(row, _FIELD_ALIASES["bid"])),
                ask=float(_pick(row, _FIELD_ALIASES["ask"])),
                last=float(_pick(row, _FIELD_ALIASES["last"])),
                volume=float(_pick(row, _FIELD_ALIASES["volume"], 0.0) or 0.0),
                open_interest=float(_pick(row, _FIELD_ALIASES["open_interest"], 0.0) or 0.0),
                underlying_price=float(_pick(row, _FIELD_ALIASES["underlying_price"])),
                meta={k: v for k, v in row.items()},
            ))
        return cls(underlying=underlying, asof=asof_ts, contracts=contracts)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        underlying: str,
        asof: pd.Timestamp | str | None = None,
    ) -> "OptionChain":
        return cls.from_records(df.to_dict("records"), underlying, asof)

    # ── views ────────────────────────────────────────────────────────────
    def to_df(self) -> pd.DataFrame:
        rows = []
        for c in self.contracts:
            rows.append({
                "underlying": c.underlying,
                "expiry": c.expiry,
                "strike": c.strike,
                "option_type": c.option_type.value,
                "bid": c.bid,
                "ask": c.ask,
                "last": c.last,
                "mid": c.mid,
                "volume": c.volume,
                "open_interest": c.open_interest,
                "underlying_price": c.underlying_price,
            })
        return pd.DataFrame(rows)

    @property
    def expiries(self) -> list[pd.Timestamp]:
        return sorted({c.expiry for c in self.contracts})

    @property
    def strikes(self) -> list[float]:
        return sorted({c.strike for c in self.contracts})

    @property
    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type is OptionType.CALL]

    @property
    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type is OptionType.PUT]

    def for_expiry(self, expiry: pd.Timestamp | str) -> list[OptionContract]:
        exp = pd.Timestamp(expiry)
        return [c for c in self.contracts if c.expiry == exp]

    @property
    def spot(self) -> float:
        """Best estimate of the underlying price (median of non-null quotes)."""
        vals = [c.underlying_price for c in self.contracts if np.isfinite(c.underlying_price)]
        return float(np.median(vals)) if vals else np.nan

    def drop_expired(self) -> "OptionChain":
        """Chain without already-expired contracts.

        Providers keep expired rows in the "current" chain until they age out, frozen at
        their last-traded state. Nothing can be reconstructed from them (T = 0 ⇒ no IV, no
        Greeks), so they are dropped rather than carried as blank rows.
        """
        asof = _to_utc(self.asof)
        live = [c for c in self.contracts if _to_utc(c.expiry) > asof]
        return OptionChain(underlying=self.underlying, asof=self.asof, contracts=live)

    @property
    def n_expired(self) -> int:
        asof = _to_utc(self.asof)
        return sum(1 for c in self.contracts if _to_utc(c.expiry) <= asof)

    def year_fractions(self) -> np.ndarray:
        """Time-to-expiry in calendar years for each contract, aligned to `contracts`.

        Normalizes tz-naive/aware mismatches (provider expiry dates are often naive while
        `asof` is UTC-aware) by coercing both operands to UTC before subtracting.
        """
        asof = _to_utc(self.asof)
        return np.array([
            max((_to_utc(c.expiry) - asof).total_seconds() / _YEAR_SECONDS, 0.0)
            for c in self.contracts
        ])


# ── Black-Scholes pricing ─────────────────────────────────────────────────────

def _d1_d2(S, K, T, r, sigma, q):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vsqrt = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrt
        d2 = d1 - vsqrt
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type, q=0.0):
    """European option price under Black-Scholes with continuous dividend yield q.

    Scalars or numpy arrays (broadcast) are both accepted. Handles the T→0 limit by
    returning discounted intrinsic value.
    """
    is_call = OptionType.parse(option_type) is OptionType.CALL
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if is_call:
        price = S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:
        price = K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)

    # T→0 or σ→0: collapse to intrinsic value.
    degenerate = (T <= 0) | (sigma <= 0)
    price = np.where(degenerate, intrinsic, price)
    return float(price) if np.ndim(price) == 0 else price


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Black-Scholes vega (∂price/∂σ, per 1.00 vol). Type-independent."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    vega = S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T)
    vega = np.where((T <= 0) | (np.asarray(sigma) <= 0), 0.0, vega)
    return float(vega) if np.ndim(vega) == 0 else vega


# ── Implied volatility ────────────────────────────────────────────────────────

def _sanitize_iv_inputs(price, S, K, T, r, q, is_call):
    """Validate one contract's inputs against the Black-Scholes no-arbitrage bounds.

    Returns `(price, status)`. `status` is `"ok"` when `price` is usable, otherwise a short
    reason the contract cannot be inverted — surfaced as `iv_status` so blank IV/Greek cells
    are explainable rather than mysterious. Quotes outside the intrinsic/upper bounds are
    rejected outright (no slack): they carry no well-defined implied vol.
    """
    if not np.isfinite(T) or T <= 0:
        return np.nan, "expired"
    if not np.isfinite(S) or S <= 0 or not np.isfinite(K) or K <= 0:
        return np.nan, "no spot"
    if not np.isfinite(price) or price <= 0:
        return np.nan, "no price"

    # No-arbitrage bounds: price must sit between intrinsic and the underlying (call) /
    # discounted strike (put).
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    lo_bound = max(S * disc_q - K * disc_r, 0.0) if is_call else max(K * disc_r - S * disc_q, 0.0)
    hi_bound = S * disc_q if is_call else K * disc_r
    if price < lo_bound:
        return np.nan, "below intrinsic"
    if price > hi_bound:
        return np.nan, "above bound"
    return price, "ok"


def _solve_bracketed(f, tol, lo=1e-6):
    """Root-find `f` over vol, growing the upper bracket until it straddles the root.

    The Black-Scholes price is monotone increasing in σ and bounded above by the
    no-arbitrage ceiling that `_sanitize_iv_inputs` already enforced, so a finite root is
    guaranteed to exist and doubling the bracket from 1.0 will reach it. The doubling count
    is bounded only as a runaway guard (2**64 vol is astronomically past any real quote),
    not as a cap on admissible volatility.
    """
    flo = f(lo)
    if not np.isfinite(flo):
        return np.nan, "unsolvable"
    hi = 1.0
    for _ in range(64):
        fhi = f(hi)
        if not np.isfinite(fhi):
            return np.nan, "unsolvable"
        if flo * fhi <= 0:
            try:
                return float(_brentq(f, lo, hi, xtol=tol, maxiter=200)), "ok"
            except Exception:
                return np.nan, "unsolvable"
        hi *= 2.0
    return np.nan, "unsolvable"


def _bs_iv_scalar(price, S, K, T, r, q, is_call, tol=1e-8, max_iter=100):
    """Invert Black-Scholes for one contract. Newton with a bracketed fallback.

    Returns `(iv, status)`.
    """
    otype = OptionType.CALL if is_call else OptionType.PUT
    price, status = _sanitize_iv_inputs(price, S, K, T, r, q, is_call)
    if status != "ok":
        return np.nan, status

    # Brenner-Subrahmanyam seed for at-the-money-ish contracts.
    sigma = math.sqrt(2 * math.pi / T) * price / S
    for _ in range(max_iter):
        model = black_scholes_price(S, K, T, r, sigma, otype, q)
        diff = model - price
        if abs(diff) < tol:
            return sigma, "ok"
        v = bs_vega(S, K, T, r, sigma, q)
        if v < 1e-10:
            break  # vega collapsed — hand off to the bracketed solver
        sigma -= diff / v
        if sigma <= 0 or not np.isfinite(sigma):
            break  # left the valid domain — hand off to the bracketed solver

    return _solve_bracketed(
        lambda sig: black_scholes_price(S, K, T, r, sig, otype, q) - price, tol)


def implied_vol_detail(price, S, K, T, r, option_type, q=0.0) -> tuple[float, str]:
    """`(iv, status)` for one contract — `status` explains any NaN instead of raising."""
    is_call = OptionType.parse(option_type) is OptionType.CALL
    return _bs_iv_scalar(price, S, K, T, r, q, is_call)


def implied_vol(price, S, K, T, r, option_type, q=0.0) -> float:
    """Implied volatility for one contract. Returns np.nan for arbitrage-violating /
    unsolvable inputs rather than raising.

    Use `implied_vol_detail` when you also want the reason for a NaN.
    """
    return implied_vol_detail(price, S, K, T, r, option_type, q)[0]


def implied_vol_chain(chain: OptionChain, r: float, q: float = 0.0,
                      price: str = "mid") -> pd.DataFrame:
    """Return the chain as a DataFrame with added `T` (years), `iv` and `iv_status` columns."""
    df = chain.to_df()
    if df.empty:
        df["T"] = []
        df["iv"] = []
        df["iv_status"] = []
        return df
    T = chain.year_fractions()
    df["T"] = T
    ivs, statuses = [], []
    for c, t in zip(chain.contracts, T):
        px = c.price(price)
        S = c.underlying_price if np.isfinite(c.underlying_price) else chain.spot
        iv, status = implied_vol_detail(px, S, c.strike, t, r, c.option_type, q)
        ivs.append(iv)
        statuses.append(status)
    df["iv"] = ivs
    df["iv_status"] = statuses
    return df


# ── Greeks ────────────────────────────────────────────────────────────────────

@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return {"delta": self.delta, "gamma": self.gamma, "vega": self.vega,
                "theta": self.theta, "rho": self.rho}


def _bs_greeks(S, K, T, r, sigma, is_call, q=0.0) -> OptionGreeks:
    if T <= 0 or sigma <= 0:
        return OptionGreeks(np.nan, np.nan, np.nan, np.nan, np.nan)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    d1 = float(d1); d2 = float(d2)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf = float(_norm_pdf(d1))
    gamma = disc_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T)
    if is_call:
        delta = disc_q * float(_norm_cdf(d1))
        theta = (-(S * disc_q * pdf * sigma) / (2 * math.sqrt(T))
                 - r * K * disc_r * float(_norm_cdf(d2))
                 + q * S * disc_q * float(_norm_cdf(d1)))
        rho = K * T * disc_r * float(_norm_cdf(d2))
    else:
        delta = -disc_q * float(_norm_cdf(-d1))
        theta = (-(S * disc_q * pdf * sigma) / (2 * math.sqrt(T))
                 + r * K * disc_r * float(_norm_cdf(-d2))
                 - q * S * disc_q * float(_norm_cdf(-d1)))
        rho = -K * T * disc_r * float(_norm_cdf(-d2))
    return OptionGreeks(delta, gamma, vega, theta, rho)


def greeks(S, K, T, r, sigma, option_type, q=0.0) -> OptionGreeks:
    """Full analytic Black-Scholes Greek set for one contract."""
    is_call = OptionType.parse(option_type) is OptionType.CALL
    return _bs_greeks(S, K, T, r, sigma, is_call, q)


def greeks_chain(chain: OptionChain, r: float, q: float = 0.0,
                 price: str = "mid") -> pd.DataFrame:
    """Chain DataFrame with `T`, `iv`, and delta/gamma/vega/theta/rho columns."""
    df = implied_vol_chain(chain, r, q, price)
    cols = {"delta": [], "gamma": [], "vega": [], "theta": [], "rho": []}
    for c, t, iv in zip(chain.contracts, df["T"], df["iv"]):
        S = c.underlying_price if np.isfinite(c.underlying_price) else chain.spot
        if not np.isfinite(iv):
            g = OptionGreeks(np.nan, np.nan, np.nan, np.nan, np.nan)
        else:
            g = greeks(S, c.strike, t, r, iv, c.option_type, q)
        for k, v in g.as_dict().items():
            cols[k].append(v)
    for k, v in cols.items():
        df[k] = v
    return df


# ── SVI parametric smile (Gatheral raw parameterization) ──────────────────────

# Minimum distinct log-moneyness points to identify the 5-parameter SVI slice.
_SVI_MIN_POINTS = 5


@dataclass
class SVIParams:
    """Raw-SVI parameters (Gatheral) for one expiry slice, in total-variance space.

    Total implied variance as a function of forward log-moneyness k:

        w(k) = a + b · ( ρ·(k − m) + √((k − m)² + σ²) )

    and the Black-Scholes implied vol is √(w(k) / T). Parameters: `a` overall level, `b ≥ 0`
    wing slope, `|ρ| < 1` skew/rotation, `m` horizontal shift of the minimum, `σ > 0`
    curvature (ATM smoothness).
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    rmse: float = float("nan")  # RMS fit error in IV (vol) units

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m)
                                  + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def iv(self, k):
        """Black-Scholes implied vol implied by the slice at log-moneyness k."""
        if not (self.T > 0):
            return np.full(np.shape(k), np.nan)
        w = np.maximum(self.total_variance(k), 0.0)
        return np.sqrt(w / self.T)

    def as_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m, "sigma": self.sigma}


def fit_svi(k, iv, T) -> "SVIParams | None":
    """Least-squares fit of raw SVI to one expiry's (log-moneyness, IV) points.

    Fits in total-variance space (w = iv²·T), which is the space SVI is linear-ish and well
    conditioned in. Returns None when there are too few distinct points to identify the five
    parameters, T is non-positive, or scipy fails to converge.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    good = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[good], iv[good]
    if len(np.unique(k)) < _SVI_MIN_POINTS or not (T > 0):
        return None

    w = iv ** 2 * T
    k_lo, k_hi = float(k.min()), float(k.max())
    span = max(k_hi - k_lo, 1e-3)
    w_max = float(w.max())

    # Seeds: level near the variance floor, mild equity skew, minimum at the observed low.
    a0 = max(float(w.min()) * 0.5, 1e-6)
    m0 = float(k[np.argmin(w)])
    lb = [-w_max,     1e-8, -0.999, k_lo - span, 1e-4]
    ub = [2 * w_max,  10.0,  0.999, k_hi + span, 2.0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in
          zip([a0, 0.1, -0.5, m0, 0.1], lb, ub)]

    def resid(p):
        a, b, rho, m, sig = p
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))
        return model - w

    try:
        sol = _least_squares(resid, p0, bounds=(lb, ub), method="trf",
                             max_nfev=2000, xtol=1e-10, ftol=1e-10)
    except Exception:
        return None

    params = SVIParams(*sol.x, T=float(T))
    params.rmse = float(np.sqrt(np.nanmean((params.iv(k) - iv) ** 2)))
    return params


# ── Implied-volatility surface ────────────────────────────────────────────────

@dataclass
class IVSurface:
    """Reconstructed IV surface for one underlying.

    The X-axis is **forward log-moneyness** k = ln(K / F), where F = S·e^{(r−q)T} is the
    per-contract forward — so k = ln(K/spot) − (r−q)·T. The at-the-forward point is k = 0 for
    every expiry, which stacks the smiles vertically instead of letting them drift with r, q,
    and T (as a spot-based K/S axis would).

    `points` is the finite-IV cloud used for smile / term-structure / grid views.
    `iv_df` is the full per-contract table (every contract, incl. IV + BS Greeks) used
    for display. IVs are inverted directly from market prices via Black-Scholes.

    Each expiry slice is fit with a **raw-SVI** smile (`svi`, one `SVIParams` per expiry),
    so smile / ATM / skew / surface queries evaluate a smooth, arbitrage-aware curve rather
    than linearly interpolating the raw quote cloud. Interpolation across maturities is done
    in total-variance space between the two bracketing SVI slices.
    """

    underlying: str
    spot: float
    points: pd.DataFrame  # columns: expiry, T, strike, log_moneyness, option_type, iv
    iv_df: pd.DataFrame | None = None
    svi: dict[pd.Timestamp, SVIParams] = field(default_factory=dict)

    @classmethod
    def from_chain(cls, chain: OptionChain, r: float, q: float = 0.0,
                   price: str = "mid") -> "IVSurface":
        spot = chain.spot
        T = chain.year_fractions()
        base = chain.to_df()
        base["T"] = T

        n = len(chain.contracts)
        ivs = np.full(n, np.nan)
        statuses = ["ok"] * n

        # Invert market prices contract by contract.
        for i, (c, t) in enumerate(zip(chain.contracts, T)):
            S = c.underlying_price if np.isfinite(c.underlying_price) else spot
            ivs[i], statuses[i] = implied_vol_detail(
                c.price(price), S, c.strike, t, r, c.option_type, q)

        base["iv"] = ivs
        base["iv_status"] = statuses
        # Forward log-moneyness: k = ln(K/F) = ln(K/spot) − (r−q)·T. Baked in at build time so
        # every downstream view (ATM at k=0, skew wings) reads straight off the column.
        if np.isfinite(spot) and spot > 0:
            base["log_moneyness"] = np.log(base["strike"] / spot) - (r - q) * base["T"]
        else:
            base["log_moneyness"] = np.nan

        # BS Greeks at the reconstructed IV.
        gcols = {k: np.full(n, np.nan) for k in ("delta", "gamma", "vega", "theta", "rho")}
        for i, (c, t, iv) in enumerate(zip(chain.contracts, T, ivs)):
            if np.isfinite(iv) and t > 0:
                g = _bs_greeks(spot, c.strike, t, r, iv, c.option_type is OptionType.CALL, q)
                for k, v in g.as_dict().items():
                    gcols[k][i] = v
        for k, v in gcols.items():
            base[k] = v

        pts = base[np.isfinite(base["iv"]) & (base["T"] > 0)][
            ["expiry", "T", "strike", "log_moneyness", "option_type", "iv"]
        ].reset_index(drop=True)

        # Fit one raw-SVI slice per expiry (call/put duplicates averaged at each k), eagerly
        # so the surface arrives fully cooked. Slices too thin to identify are simply omitted;
        # the affected views fall back to linear interpolation of the raw points.
        svi: dict[pd.Timestamp, SVIParams] = {}
        for exp in pts["expiry"].unique():
            sub = pts[pts["expiry"] == exp]
            grp = sub.groupby("log_moneyness")["iv"].mean()
            fit = fit_svi(grp.index.to_numpy(), grp.to_numpy(), float(sub["T"].iloc[0]))
            if fit is not None:
                svi[pd.Timestamp(exp)] = fit

        return cls(underlying=chain.underlying, spot=spot, points=pts, iv_df=base, svi=svi)

    def _x(self) -> np.ndarray:
        return self.points["log_moneyness"].to_numpy()

    @property
    def x_label(self) -> str:
        return "log-moneyness"

    @property
    def expiries(self) -> list[pd.Timestamp]:
        return sorted(self.points["expiry"].unique())

    def smile(self, expiry: pd.Timestamp | str) -> tuple[np.ndarray, np.ndarray]:
        """(k, iv) market points for one expiry, averaged across call/put duplicates at each
        log-moneyness. This is the raw quote cloud; use `smile_curve` for the fitted SVI line."""
        exp = pd.Timestamp(expiry)
        sub = self.points[self.points["expiry"] == exp]
        grp = sub.groupby("log_moneyness")["iv"].mean().sort_index()
        return grp.index.to_numpy(), grp.to_numpy()

    def smile_curve(self, expiry: pd.Timestamp | str, n: int = 100
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Dense (k, iv) SVI curve spanning the expiry's observed strikes.

        Returns empty arrays when that slice could not be fit (too few points)."""
        exp = pd.Timestamp(expiry)
        p = self.svi.get(exp)
        sub = self.points[self.points["expiry"] == exp]
        if p is None or sub.empty:
            return np.array([]), np.array([])
        ks = np.linspace(float(sub["log_moneyness"].min()),
                         float(sub["log_moneyness"].max()), n)
        return ks, p.iv(ks)

    def term_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """(T, ATM-iv) — one point per expiry, SVI IV at the forward (log-moneyness 0)."""
        Ts, ivs = [], []
        for exp in self.expiries:
            atm = self.atm_vol(exp)
            if not np.isfinite(atm):
                continue
            Ts.append(float(self.points[self.points["expiry"] == exp]["T"].iloc[0]))
            ivs.append(atm)
        order = np.argsort(Ts)
        return np.array(Ts)[order], np.array(ivs)[order]

    def atm_vol(self, expiry: pd.Timestamp | str) -> float:
        """At-the-forward IV for one expiry — the SVI slice evaluated at log-moneyness 0."""
        exp = pd.Timestamp(expiry)
        p = self.svi.get(exp)
        if p is not None:
            return float(p.iv(0.0))
        x, iv = self.smile(exp)  # fallback: linear on raw points
        if len(x) == 0:
            return np.nan
        return float(np.interp(0.0, x, iv)) if len(x) > 1 else float(iv[0])

    def skew(self, expiry: pd.Timestamp | str, lo: float = -0.1, hi: float = 0.1) -> float:
        """IV(lo) − IV(hi) across log-moneyness for one expiry (put-side minus call-side).

        Evaluated on the SVI slice; defaults probe the ±0.1 wings. Positive means downside is
        bid. Falls back to linear interpolation of the raw points when the slice is unfit.
        """
        exp = pd.Timestamp(expiry)
        p = self.svi.get(exp)
        if p is not None:
            return float(p.iv(lo) - p.iv(hi))
        x, iv = self.smile(exp)
        if len(x) < 2:
            return np.nan
        return float(np.interp(lo, x, iv) - np.interp(hi, x, iv))

    def _slices(self) -> list[SVIParams]:
        """Fitted SVI slices ordered by maturity."""
        return sorted(self.svi.values(), key=lambda p: p.T)

    def _iv_at(self, k: np.ndarray, T: float) -> np.ndarray:
        """SVI IV over a log-moneyness array at maturity T.

        Between the two bracketing slices, total variance w = σ²·T is linearly interpolated in
        T (calendar-time interpolation) and converted back to vol; outside the fitted range the
        nearest slice is held flat.
        """
        slices = self._slices()
        if not slices:
            return np.full(np.shape(k), np.nan)
        if T <= slices[0].T:
            return slices[0].iv(k)
        if T >= slices[-1].T:
            return slices[-1].iv(k)
        lo = hi = slices[-1]
        for i in range(1, len(slices)):
            if slices[i].T >= T:
                lo, hi = slices[i - 1], slices[i]
                break
        wl = np.maximum(lo.total_variance(k), 0.0)
        wh = np.maximum(hi.total_variance(k), 0.0)
        w = wl + (wh - wl) * (T - lo.T) / (hi.T - lo.T)
        return np.sqrt(np.maximum(w, 0.0) / T)

    def interpolate(self, x: float, T: float) -> float:
        """SVI-model IV at (log-moneyness x, time-to-expiry T years)."""
        return float(self._iv_at(np.array([x], dtype=float), float(T))[0])

    def grid(self, n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mesh (X, Y, Z) of the fitted SVI surface for 3-D plotting.

        X = log-moneyness axis, Y = time-to-expiry (years), Z = SVI IV (per-slice smile,
        total-variance interpolation across maturities).
        """
        px = self._x()
        if len(self.svi) == 0 or len(px) == 0:
            empty = np.zeros((n, n))
            return empty, empty, empty
        py = self.points["T"].to_numpy()
        xi = np.linspace(px.min(), px.max(), n)
        yi = np.linspace(py.min(), py.max(), n)
        X, Y = np.meshgrid(xi, yi)
        Z = np.empty_like(X)
        for j in range(n):
            Z[j, :] = self._iv_at(xi, float(yi[j]))
        return X, Y, Z
