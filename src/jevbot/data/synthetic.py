"""`SyntheticProvider` (DESIGN.md 5.10, D20) - a seeded, offline, deterministic `ChainProvider` for tests and smoke runs.

Only `data.synthetic.mode = "offline"` is implemented: a deterministic price path per underlying (mean-reverting
log-vol, cross-correlated shocks), Black-76 priced chains with a FLAT (no-skew) smile, and the side series
`DataView` needs beyond chains. `mode = "d20"` (Black-Scholes seeded from real bars / a real vol index / a real bill
rate) is NOT implemented here - the constructor raises `ConfigError` for it. This is a deliberate scope cut (see the
module docstrings of `cycle.py` / `backtest.py` for the full list of what this minimal engine skips).

Every snapshot is slot `eod`, `ts = knowable_at =` that session's calendar close, `spot_measure = "synthetic"`,
`div_unmodelled = True`, `fidelity = Fidelity.SYNTHETIC`, `source = "synthetic"` - SYNTHETIC DATA / MOCK DECIDER,
never evidence of anything. Chains are priced with a flat-in-strike smile (`price_chain`) and then run through the
SAME `data.surface.parity_forwards` / `data.surface.enrich` code real data uses, so the own-IV solve, the parity
forward and `data.surface.chain_nodes` are exercised exactly as they would be on mirror data.

Deviation from the literal 5.10 method list: this module does NOT implement its own `SyntheticProvider.tables()`.
Per the coordinator's pluggability requirement, table assembly (`bars:<U>`, `daily:<U>`, `volidx:<NAME>`, `rates`) is
centralised in ONE provider-agnostic function, `jevbot.data.market.build_market_data`, which any `ChainProvider` can
feed through the small `daily_closes(underlying) -> pd.Series` duck-typed extra method this class also implements
(the same shape a future `MirrorParquetProvider` will carry). `generate_vol_index` below is this provider's own
optional vol-index generator, passed into `build_market_data(..., vol_index=...)` by the caller (`backtest.py` /
the CLI) - it is NOT part of the `ChainProvider` contract, so nothing downstream needs to know this provider exists.

Deterministic: two `SyntheticProvider(cfg, calendar)` built from the same `cfg` (same `run.seed`, same
`universe.underlyings`, same `[data.synthetic]`, same `run.start` / `run.end`) produce byte-identical
`content_hash`es and the same `manifest_hash()`.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from jevbot import bs
from jevbot.cal import year_fraction
from jevbot.config import Config
from jevbot.data import surface
from jevbot.errors import ConfigError, DataUnavailable
from jevbot.types import CHAIN_COLUMNS, Cents, ChainSnapshot, Fidelity, Right, SnapshotKey, Slot

if TYPE_CHECKING:
    from jevbot.protocols import Calendar

__all__ = ["SyntheticProvider", "generate_paths", "generate_vol_index", "price_chain"]

_MILLI_PER_CENT: Final = 10
_STRIKE_SNAPS: Final = (50, 100, 250, 500, 1000)  # cents; step_c snaps to the nearest of these (5.10)
_LOOKBACK_DAYS: Final = 470  # calendar days of history generated BEFORE run.start, so percentile features (iv_rank,
# vix_pctile, rv20_pctile: data.min_history_sessions = 126 sessions by default) are populated once the backtest's
# own window begins. ~470 calendar days is comfortably > 252 trading sessions (features.YEAR_SESSIONS).
_SMOKE_BANNER: Final = "SYNTHETIC DATA / MOCK DECIDER - NOT EVIDENCE OF ANY SKILL"


def _rng(seed: int, tag: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|synthetic|{tag}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _snap_step(spot0_c: float, pct: float) -> int:
    raw = max(1.0, spot0_c * pct)
    return min(_STRIKE_SNAPS, key=lambda s: abs(s - raw))


# ======================================================================================================================
# 5.10 offline price paths
# ======================================================================================================================


def generate_paths(cfg: Config, calendar: "Calendar") -> dict[str, pd.DataFrame]:
    """A deterministic OHLC + vol-state path per `universe.underlyings` (5.10 offline mode).

    Columns: `session, open, high, low, close, vol_state` (cents for OHLC; `vol_state` is the underlying's own
    mean-reverting instantaneous vol level `s_t`, decimal). Sessions run from `run.start - LOOKBACK` to `run.end`
    (`_LOOKBACK_DAYS` calendar days), so percentile-based features have history once the run window begins.
    """
    scfg = cfg.data.synthetic
    if scfg.mode != "offline":
        raise ConfigError(f"data/synthetic.py: only [data.synthetic] mode 'offline' is implemented here, got {scfg.mode!r}")
    sessions = calendar.sessions(cfg.run.start - timedelta(days=_LOOKBACK_DAYS), cfg.run.end)
    if len(sessions) < 2:
        raise ConfigError("data/synthetic.py: the calendar window is too short to generate a synthetic path")
    n = len(sessions)
    seed = cfg.run.seed
    common = _rng(seed, "common")
    z_common = common.standard_normal(n)
    xi = common.standard_normal(n)

    out: dict[str, pd.DataFrame] = {}
    for underlying in cfg.universe.underlyings:
        rng = _rng(seed, underlying)
        z_u = rng.standard_normal(n)
        eps = math.sqrt(scfg.cross_corr) * z_common + math.sqrt(max(0.0, 1.0 - scfg.cross_corr)) * z_u
        z_open = rng.standard_normal(n)
        z_hl = rng.standard_normal(n)

        opens = np.empty(n)
        highs = np.empty(n)
        lows = np.empty(n)
        closes = np.empty(n)
        vol_state = np.empty(n)

        log_base = math.log(scfg.base_vol)
        log_s = log_base
        prev_close = float(cfg.data.synthetic.start_price_usd.get(underlying, 100.0)) * 100.0  # cents
        window: list[float] = []
        for i in range(n):
            log_s = log_s + scfg.vol_mean_revert * (log_base - log_s) + scfg.vol_of_vol * float(xi[i])
            s_t = math.exp(log_s)
            vol_state[i] = s_t
            g = _trend_signal(window)
            m_t = (scfg.planted_drift_bp / 1e4) * g
            r_t = m_t + (s_t / math.sqrt(252.0)) * float(eps[i])
            close_t = prev_close * math.exp(r_t)
            open_t = prev_close * math.exp(0.25 * abs(r_t) * (1.0 if z_open[i] >= 0 else -1.0))
            spread = 0.3 * (s_t / math.sqrt(252.0)) * abs(float(z_hl[i]))
            highs[i] = round(max(open_t, close_t) * math.exp(spread))
            lows[i] = round(min(open_t, close_t) * math.exp(-spread))
            opens[i] = round(open_t)
            closes[i] = round(close_t)
            window.append(close_t)
            prev_close = close_t
        out[underlying] = pd.DataFrame(
            {
                "session": pd.to_datetime(sessions),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "vol_state": vol_state,
            }
        )
    return out


def _trend_signal(closes: list[float]) -> float:
    """`g_{t-1} = sign(MA20 - MA50)` when `|MA20/MA50 - 1| > 0.005`, else 0 (the 5.10 planted-skill lever)."""
    if len(closes) < 50:
        return 0.0
    ma20 = sum(closes[-20:]) / 20.0
    ma50 = sum(closes[-50:]) / 50.0
    if ma50 <= 0.0 or abs(ma20 / ma50 - 1.0) <= 0.005:
        return 0.0
    return 1.0 if ma20 > ma50 else -1.0


def generate_vol_index(cfg: Config, calendar: "Calendar", paths: Mapping[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """The 5.10 side volatility tables, simplified: `VIX`/`VXN`/`RVX` etc. come from each underlying's OWN `vol_state`
    (`universe.iv_proxy`); `VIX9D`/`VIX3M`/`VVIX`/`SKEW` are derived once from the FIRST underlying's path (a
    simplification: the real design ties them to the market-wide "VIX" complex only - fine here, since none of them
    is a REQUIRED feature, 5.3). Every series is indexed by session (naive `datetime64[ns]`), values as levels
    (not bp): `build_market_data` attaches the D22 `knowable_at = next session open` gate.
    """
    scfg = cfg.data.synthetic
    out: dict[str, pd.Series] = {}
    for underlying, path in paths.items():
        proxy = cfg.universe.iv_proxy.get(underlying)
        if not proxy:
            continue
        level = 100.0 * path["vol_state"].to_numpy() * (1.0 + scfg.implied_premium)
        out[proxy] = pd.Series(level, index=pd.DatetimeIndex(path["session"]), name=proxy)
    if cfg.universe.underlyings:
        base = out.get(cfg.universe.iv_proxy.get(cfg.universe.underlyings[0], ""))
        if base is not None:
            rng = _rng(cfg.run.seed, "indices")
            n = len(base)
            out.setdefault("VIX9D", 0.97 * base)
            out.setdefault("VIX3M", 1.06 * base)
            out.setdefault("VVIX", pd.Series(90.0 + 10.0 * rng.standard_normal(n), index=base.index, name="VVIX"))
            out.setdefault("SKEW", pd.Series(125.0 + 3.0 * rng.standard_normal(n), index=base.index, name="SKEW"))
    return out


# ======================================================================================================================
# The pricer (offline AND, were it built, d20 mode)
# ======================================================================================================================


def _listed_fridays(session: date, max_dte: int, calendar: "Calendar") -> list[date]:
    out: list[date] = []
    friday = session + timedelta(days=(4 - session.weekday()) % 7 or 7)
    while (friday - session).days <= max_dte + 1:
        out.append(calendar.prev_or_same_session(friday))
        friday += timedelta(days=7)
    return sorted(set(out))


def price_chain(
    session: date, ts: datetime, spot_c: int, v: float, rate: float, cfg: Config, calendar: "Calendar", step_c: int, underlying: str
) -> pd.DataFrame:
    """RAW (pre-enrichment) quotes: `expiry, right, strike_milli, bid, ask, bid_size, ask_size, oi_prev, iv_vendor,
    quote_ts, occ` for every listed strike within `data.moneyness_window` of the parity forward, per expiry.

    Deviation from the literal 5.10 signature: `underlying` is an added keyword-free trailing parameter (needed to
    fill the `occ` column here rather than in a second pass over the caller). Flat-in-strike vol per expiry
    (`sigma_E = v * clip(1 + term_slope * ln(dte/30), 0.5, 1.5)`, no skew): the skew maths belongs to
    `tests/fixtures/chain_factory.py`, not here (5.10).
    """
    scfg = cfg.data.synthetic
    max_dte = cfg.data.max_dte
    window = cfg.data.moneyness_window
    blocks: list[pd.DataFrame] = []
    for expiry in _listed_fridays(session, max_dte, calendar):
        last = calendar.prev_or_same_session(expiry)
        dte = (last - session).days
        if not 1 <= dte <= max_dte:
            continue
        tau = year_fraction(ts, calendar.open_close(last)[1])
        if tau <= 0.0:
            continue
        sigma_e = v * min(max(1.0 + scfg.term_slope * math.log(dte / 30.0), 0.5), 1.5)
        if sigma_e <= 0.0:
            continue
        df = math.exp(-rate * tau)
        fwd = spot_c * math.exp(rate * tau)
        lo = math.ceil(fwd * math.exp(-window) / step_c)
        hi = math.floor(fwd * math.exp(window) / step_c)
        if hi < lo:
            continue
        strike_c = (np.arange(lo, hi + 1, dtype=np.int64) * step_c).astype(np.int64)
        n = strike_c.size
        for right, is_call in ((Right.CALL, True), (Right.PUT, False)):
            call_mask = np.full(n, is_call, dtype=np.bool_)
            theo = bs.b76_price(float(fwd), strike_c.astype(np.float64), tau, sigma_e, df, call_mask)
            mid = np.maximum(1.0, np.round(theo)).astype(np.int64)
            half = np.maximum(1, np.ceil(scfg.spread_pct * mid.astype(np.float64) / 2.0)).astype(np.int64)
            bid = np.maximum(mid - half, 0)
            ask = mid + half
            strike_milli = strike_c * _MILLI_PER_CENT
            occ = [f"{underlying}{expiry:%y%m%d}{right.value}{int(m):08d}" for m in strike_milli]
            blocks.append(
                pd.DataFrame(
                    {
                        "occ": occ,
                        "expiry": np.full(n, np.datetime64(expiry, "ns")),
                        "right": right.value,
                        "strike_milli": strike_milli,
                        "bid": bid,
                        "ask": ask,
                        "bid_size": np.full(n, scfg.quote_size, dtype=np.int64),
                        "ask_size": np.full(n, scfg.quote_size, dtype=np.int64),
                        "oi_prev": np.full(n, scfg.open_interest, dtype=np.int64),
                        "iv_vendor": np.full(n, np.nan, dtype=np.float64),
                    }
                )
            )
    if not blocks:
        return pd.DataFrame(
            columns=["occ", "expiry", "right", "strike_milli", "bid", "ask", "bid_size", "ask_size", "oi_prev", "iv_vendor"]
        )
    table = pd.concat(blocks, ignore_index=True)
    table["quote_ts"] = pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns, UTC]")
    return table


_CHAIN_DTYPES: Final[Mapping[str, str]] = {
    "strike_milli": "int64",
    "dte": "int64",
    "bid": "int64",
    "ask": "int64",
    "fwd": "int64",
    "bid_size": "Int64",
    "ask_size": "Int64",
    "oi_prev": "Int64",
    "iv": "float64",
    "delta": "float64",
    "vega": "float64",
    "iv_vendor": "float64",
}


# ======================================================================================================================
# SyntheticProvider
# ======================================================================================================================


class SyntheticProvider:
    """Implements `protocols.ChainProvider` (3.2) over `generate_paths` / `price_chain` - see the module docstring.

    `fidelity = Fidelity.SYNTHETIC`, `source = "synthetic"`. Every snapshot is precomputed at construction time (the
    whole run's history is cheap to build: a few hundred sessions, a handful of underlyings) and served from memory.
    """

    def __init__(
        self,
        cfg: Config,
        calendar: "Calendar",
        *,
        bars: Mapping[str, pd.DataFrame] | None = None,
        vol_index: Mapping[str, pd.Series] | None = None,
        rates: pd.Series | None = None,
    ) -> None:
        if cfg.data.synthetic.mode != "offline":
            raise ConfigError("SyntheticProvider: only [data.synthetic] mode 'offline' is implemented in this build")
        if bars is not None or vol_index is not None or rates is not None:
            raise ConfigError("SyntheticProvider: mode 'offline' generates everything itself (bars/vol_index/rates must be None)")
        self._cfg = cfg
        self._calendar = calendar
        self._paths = generate_paths(cfg, calendar)
        step_by_underlying = {
            u: _snap_step(float(path["close"].iloc[0]), cfg.data.synthetic.strike_step_pct) for u, path in self._paths.items()
        }
        self._chains: dict[str, dict[SnapshotKey, ChainSnapshot]] = {}
        for underlying, path in self._paths.items():
            per_session: dict[SnapshotKey, ChainSnapshot] = {}
            for row in path.itertuples():
                session = pd.Timestamp(row.session).date()
                spot_c = int(row.close)
                v = float(row.vol_state) * (1.0 + cfg.data.synthetic.implied_premium)
                per_session[SnapshotKey(session=session, slot=Slot.EOD)] = self._build_snapshot(
                    underlying, session, spot_c, v, step_by_underlying[underlying]
                )
            self._chains[underlying] = per_session
        material = {
            "mode": cfg.data.synthetic.mode,
            "seed": cfg.run.seed,
            "underlyings": list(cfg.universe.underlyings),
            "synthetic": _synthetic_table(cfg),
            "calendar_bounds": [d.isoformat() for d in calendar.bounds] if hasattr(calendar, "bounds") else [],
        }
        text = json.dumps(material, sort_keys=True, separators=(",", ":"))
        self._manifest_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _build_snapshot(self, underlying: str, session: date, spot_c: int, v: float, step_c: int) -> ChainSnapshot:
        ts = self._calendar.open_close(session)[1]  # eod: ts = knowable_at = the session's calendar close (5.10)
        rate = self._cfg.data.synthetic.rate_bp / 1e4
        raw = price_chain(session, ts, spot_c, v, rate, self._cfg, self._calendar, step_c, underlying)
        forwards = surface.parity_forwards(raw, rate, ts, self._calendar) if not raw.empty else {}
        enriched = surface.enrich(raw, forwards, rate, ts, self._calendar, session=session)
        enriched = enriched.astype(dict(_CHAIN_DTYPES))[CHAIN_COLUMNS]
        enriched = enriched.sort_values(["expiry", "right", "strike_milli"], kind="mergesort").reset_index(drop=True)
        digest = hashlib.sha256(enriched.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
        return ChainSnapshot(
            underlying=underlying,
            key=SnapshotKey(session=session, slot=Slot.EOD),
            ts=ts,
            knowable_at=ts,
            spot=spot_c,
            spot_measure="synthetic",
            div_unmodelled=True,
            rate=rate,
            table=enriched,
            fidelity=Fidelity.SYNTHETIC,
            source="synthetic",
            content_hash=digest,
        )

    # --- ChainProvider ----------------------------------------------------------------------------------------------

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.SYNTHETIC

    @property
    def source(self) -> str:
        return "synthetic"

    def underlyings(self) -> tuple[str, ...]:
        return tuple(self._chains)

    def keys(self, underlying: str, start: date, end: date) -> list[SnapshotKey]:
        per_session = self._chains.get(underlying, {})
        return sorted((k for k in per_session if start <= k.session <= end), key=lambda k: (k.session, k.slot.value))

    def get_chain(self, underlying: str, key: SnapshotKey) -> ChainSnapshot | None:
        return self._chains.get(underlying, {}).get(key)

    def manifest_hash(self) -> str:
        return self._manifest_hash

    # --- extra (duck-typed by data.market.build_market_data; the analogue of a future MirrorParquetProvider) --------

    def daily_closes(self, underlying: str) -> pd.Series:
        """This underlying's official close series (cents), index = session date - what `build_market_data` needs
        besides chains (the same shape a `MirrorParquetProvider.daily_closes` will carry)."""
        path = self._paths.get(underlying)
        if path is None:
            raise DataUnavailable(f"SyntheticProvider: no path for {underlying!r}")
        return pd.Series(path["close"].to_numpy(dtype="int64"), index=pd.DatetimeIndex(path["session"]), name="close_c")

    def vol_index(self) -> dict[str, pd.Series]:
        """This provider's own generated Cboe-style index series (5.10, simplified) - pass to
        `build_market_data(..., vol_index=provider.vol_index())`; not part of the `ChainProvider` contract."""
        return generate_vol_index(self._cfg, self._calendar, self._paths)

    @property
    def rate_bp(self) -> int:
        return int(self._cfg.data.synthetic.rate_bp)


def _synthetic_table(cfg: Config) -> dict[str, object]:
    s = cfg.data.synthetic
    return {
        "mode": s.mode,
        "base_vol": s.base_vol,
        "vol_mean_revert": s.vol_mean_revert,
        "vol_of_vol": s.vol_of_vol,
        "cross_corr": s.cross_corr,
        "implied_premium": s.implied_premium,
        "planted_drift_bp": s.planted_drift_bp,
        "term_slope": s.term_slope,
        "spread_pct": s.spread_pct,
        "strike_step_pct": s.strike_step_pct,
        "quote_size": s.quote_size,
        "open_interest": s.open_interest,
        "rate_bp": s.rate_bp,
        "start_price_usd": dict(s.start_price_usd),
        "run_start": cfg.run.start.isoformat(),
        "run_end": cfg.run.end.isoformat(),
    }


if TYPE_CHECKING:
    from jevbot.protocols import ChainProvider

    def _synthetic_provider_is_a_chain_provider(p: SyntheticProvider) -> "ChainProvider":
        return p
