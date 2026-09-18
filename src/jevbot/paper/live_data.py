"""`AlpacaLiveProvider`: one live snapshot of the Alpaca option chains, as a `ChainProvider` (DESIGN 11.7, lean v1).

Read-only: market-data endpoints plus the broker clock; it never touches an order endpoint.

What one snapshot holds, per underlying:
- the option chain for expiries in `[session + 1, session + data.max_dte]`, converted to the raw quote table and enriched
  by the SAME `data.surface.parity_forwards` / `enrich` code the historical and synthetic providers use;
- `daily_closes` from Alpaca daily bars (split-adjusted, SIP, >= 16 minutes old; the current session is excluded
  while it trades);
- an `iv_proxy` IV history (V10): the underlying's Cboe index (`universe.iv_proxy`) scaled so its last knowable value
  equals today's own-chain 30-day IV. Those rows are tagged `source = "proxy"`.

LEAN v1 (2026-09-18). Not done yet, compared with 11.7:
- no open-interest fetch (`oi_prev` is <NA>, which `liquidity.allow_missing_open_interest = true` allows);
- no news, corporate actions / ex-dividend dates, or FOMC events;
- no recorder integration.
Fetch failures raise: no value is ever invented.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import urllib.request
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import OptionChainRequest, StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

from jevbot.config import Config
from jevbot.data import surface
from jevbot.errors import DataUnavailable
from jevbot.types import CHAIN_COLUMNS, Cents, ChainSnapshot, Fidelity, Slot, SnapshotKey

if TYPE_CHECKING:
    from jevbot.paper.alpaca_client import AlpacaClients
    from jevbot.protocols import Calendar

__all__ = ["AlpacaLiveProvider", "current_session", "fetch_bill_rate", "fetch_cboe_index"]

_NY: Final = ZoneInfo("America/New_York")
_CBOE: Final = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
_TREASURY: Final = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
    "{year}/all?type=daily_treasury_bill_rates&field_tdr_date_value={year}&page&_format=csv"
)
_UA: Final = {"User-Agent": "Mozilla/5.0 (jevbot; personal research)"}
_HISTORY_DAYS: Final = 800  # calendar days of daily bars: > 252 sessions of rank window + rv warm-up
_SIP_DELAY: Final = timedelta(minutes=16)  # free plan: SIP data only when at least 15 minutes old
_CHAIN_DTYPES: Final[dict[str, str]] = {
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


def current_session(calendar: Calendar, now: datetime) -> tuple[date, Slot]:
    """The session a snapshot taken at `now` belongs to: the trading session (`dec`) or the last closed one (`eod`)."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    s = calendar.prev_or_same_session(now.astimezone(_NY).date())
    opened, closed = calendar.open_close(s)
    if now < opened:
        return calendar.prev_session(s), Slot.EOD
    return (s, Slot.DEC) if now <= closed else (s, Slot.EOD)


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(request, timeout=30) as response:
        text: str = response.read().decode("utf-8")
    return text


def fetch_cboe_index(sym: str) -> pd.Series:
    """Cboe daily close history for an index (VIX, VXN, RVX, ...): index = session date, values = index points."""
    out: dict[pd.Timestamp, float] = {}
    for row in csv.DictReader(io.StringIO(_get(_CBOE.format(sym=sym)))):
        value = row.get("CLOSE") or row.get(sym)
        if not value:
            continue
        out[pd.Timestamp(datetime.strptime(row["DATE"], "%m/%d/%Y").date())] = float(value)  # noqa: DTZ007 - only the date is kept
    if not out:
        raise DataUnavailable(f"cboe: empty history for {sym}")
    return pd.Series(out).sort_index()


def fetch_bill_rate(today: date) -> float:
    """Latest 13-week T-bill coupon-equivalent rate (decimal), from the US Treasury daily bill-rates CSV."""
    for year in (today.year, today.year - 1):
        rows = list(csv.DictReader(io.StringIO(_get(_TREASURY.format(year=year)))))
        values = [(r["Date"], r.get("13 WEEKS COUPON EQUIVALENT")) for r in rows]
        dated = sorted((datetime.strptime(d, "%m/%d/%Y").date(), float(v)) for d, v in values if v)  # noqa: DTZ007 - only the date is kept
        dated = [(d, v) for d, v in dated if d < today]  # EOD series: usable from the next session
        if dated:
            return dated[-1][1] / 100.0
    raise DataUnavailable("treasury: no 13-week bill rate found")


def _cents(x: float | None) -> int | None:
    return None if x is None or not math.isfinite(x) else round(x * 100)


def _raw_chain(snaps: dict[str, Any]) -> pd.DataFrame:
    """Alpaca `OptionsSnapshot`s -> the raw quote table `surface.parity_forwards` / `enrich` consume."""
    rows: list[dict[str, Any]] = []
    for occ, snap in snaps.items():
        q = snap.latest_quote
        if q is None:
            continue
        bid, ask = _cents(q.bid_price), _cents(q.ask_price)
        if bid is None or ask is None or ask <= 0 or bid < 0 or ask < bid:  # empty or crossed: unusable
            continue
        tail = occ[-15:]  # yymmdd + C/P + 8-digit strike in milli-dollars
        rows.append(
            {
                "occ": occ,
                "expiry": np.datetime64(datetime.strptime(tail[:6], "%y%m%d").date(), "ns"),  # noqa: DTZ007 - only the date is kept
                "right": tail[6],
                "strike_milli": int(tail[7:]),
                "bid": bid,
                "ask": ask,
                "bid_size": int(q.bid_size) if q.bid_size is not None else None,
                "ask_size": int(q.ask_size) if q.ask_size is not None else None,
                "iv_vendor": snap.implied_volatility,
                "quote_ts": q.timestamp,
            }
        )
    if not rows:
        raise DataUnavailable("alpaca: option chain has no usable two-sided quotes")
    table = pd.DataFrame(rows)
    table["oi_prev"] = pd.array([pd.NA] * len(table), dtype="Int64")
    table["bid_size"] = table["bid_size"].astype("Int64")
    table["ask_size"] = table["ask_size"].astype("Int64")
    table["iv_vendor"] = table["iv_vendor"].astype("float64")
    table["quote_ts"] = pd.to_datetime(table["quote_ts"], utc=True)
    return table


class AlpacaLiveProvider:
    """A single live snapshot (one `SnapshotKey`) of every configured underlying. Build it with `snapshot()`."""

    def __init__(
        self,
        *,
        key: SnapshotKey,
        chains: dict[str, ChainSnapshot],
        closes: dict[str, pd.Series],
        vol_index: dict[str, pd.Series],
        iv_proxy: dict[str, pd.Series],
        rate: float,
    ) -> None:
        self._key = key
        self._chains = chains
        self._closes = closes
        self._vol_index = vol_index
        self._iv_proxy = iv_proxy
        self._rate = rate
        joined = "|".join(f"{u}:{c.content_hash}" for u, c in sorted(chains.items()))
        self._manifest = hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @classmethod
    def snapshot(cls, clients: AlpacaClients, cfg: Config, calendar: Calendar, now: datetime) -> AlpacaLiveProvider:
        session, slot = current_session(calendar, now)
        key = SnapshotKey(session=session, slot=slot)
        underlyings = tuple(cfg.universe.underlyings)
        rate = fetch_bill_rate(session)
        trades = clients.stocks.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=list(underlyings)))
        quotes = clients.stocks.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=list(underlyings)))
        vol_index: dict[str, pd.Series] = {}
        chains: dict[str, ChainSnapshot] = {}
        closes: dict[str, pd.Series] = {}
        iv_proxy: dict[str, pd.Series] = {}
        for u in underlyings:
            spot = _spot(quotes.get(u), trades.get(u))
            chain = _chain(clients, cfg, calendar, u, key, spot, rate, now)
            chains[u] = chain
            closes[u] = _daily_closes(clients, u, session, slot, now, calendar)
            proxy_sym = cfg.universe.iv_proxy[u]
            if proxy_sym not in vol_index:
                history = fetch_cboe_index(proxy_sym)
                # Cboe archives extend beyond our calendar and include exchange holidays.
                # Keep only completed sessions in the requested history window.
                history = history[
                    (history.index >= pd.Timestamp((now - timedelta(days=_HISTORY_DAYS)).date())) & (history.index < pd.Timestamp(session))
                ]
                vol_index[proxy_sym] = history[[calendar.is_session(d.date()) for d in history.index]]
            iv_proxy[u] = _scaled_proxy(vol_index[proxy_sym], chain, calendar, session)
        return cls(key=key, chains=chains, closes=closes, vol_index=vol_index, iv_proxy=iv_proxy, rate=rate)

    # --- ChainProvider (3.2) --------------------------------------------------------------------------------------------

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.LIVE_INDICATIVE

    @property
    def source(self) -> str:
        return "alpaca_live"

    def underlyings(self) -> tuple[str, ...]:
        return tuple(self._chains)

    def keys(self, underlying: str, start: date, end: date) -> list[SnapshotKey]:
        ok = underlying in self._chains and start <= self._key.session <= end
        return [self._key] if ok else []

    def get_chain(self, underlying: str, key: SnapshotKey) -> ChainSnapshot | None:
        return self._chains.get(underlying) if key == self._key else None

    def manifest_hash(self) -> str:
        return self._manifest

    # --- extras consumed by data.market.build_market_data ---------------------------------------------------------------

    @property
    def key(self) -> SnapshotKey:
        return self._key

    @property
    def rate(self) -> float:
        return self._rate

    def daily_closes(self, underlying: str) -> pd.Series:
        return self._closes[underlying]

    def vol_index(self) -> dict[str, pd.Series]:
        return dict(self._vol_index)

    def iv_proxy(self) -> dict[str, pd.Series]:
        return dict(self._iv_proxy)


def _spot(quote: Any, trade: Any) -> Cents:
    """Decision-time reference price: the underlying's quote mid when two-sided and sane, else the last trade."""
    if quote is not None and quote.bid_price and quote.ask_price and 0 < quote.bid_price <= quote.ask_price:
        mid = (quote.bid_price + quote.ask_price) / 2.0
        if math.isfinite(mid) and (trade is None or not trade.price or abs(mid / trade.price - 1.0) < 0.01):
            return round(float(mid) * 100)
    if trade is not None and trade.price and math.isfinite(trade.price) and trade.price > 0:
        return round(float(trade.price) * 100)
    raise DataUnavailable("alpaca: no usable underlying quote or trade")


def _chain(
    clients: AlpacaClients, cfg: Config, calendar: Calendar, u: str, key: SnapshotKey, spot: Cents, rate: float, now: datetime
) -> ChainSnapshot:
    snaps = clients.options.get_option_chain(
        OptionChainRequest(
            underlying_symbol=u,
            expiration_date_gte=key.session + timedelta(days=1),
            expiration_date_lte=key.session + timedelta(days=cfg.data.max_dte),
        )
    )
    raw = _raw_chain(dict(snaps))
    ts = datetime.now(UTC)  # receipt time, never the timestamp from before the network request
    forwards = surface.parity_forwards(raw, rate, ts, calendar)
    enriched = surface.enrich(raw, forwards, rate, ts, calendar, session=key.session)
    keep = (enriched["dte"] >= 1) & (enriched["dte"] <= cfg.data.max_dte)
    fwd = enriched["fwd"].astype("float64")
    moneyness = np.log((enriched["strike_milli"].astype("float64") / 10.0) / fwd.where(fwd > 0))
    keep &= moneyness.abs() <= cfg.data.moneyness_window
    enriched = enriched[keep].astype(_CHAIN_DTYPES)[CHAIN_COLUMNS]
    if enriched.empty:
        raise DataUnavailable(f"alpaca: no usable contracts after filtering {u}")
    enriched = enriched.sort_values(["expiry", "right", "strike_milli"], kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256(enriched.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    return ChainSnapshot(
        underlying=u,
        key=key,
        ts=ts,
        knowable_at=ts,  # live: the local receive time
        spot=spot,
        spot_measure="live_mid",
        div_unmodelled=True,  # v1 fetches no dividend schedule
        rate=rate,
        table=enriched,
        fidelity=Fidelity.LIVE_INDICATIVE,
        source="alpaca_live",
        content_hash=digest,
    )


def _daily_closes(clients: AlpacaClients, u: str, session: date, slot: Slot, now: datetime, calendar: Calendar) -> pd.Series:
    """Split-adjusted daily closes in integer cents, indexed by session date; the live session is excluded."""
    bars = clients.stocks.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=u,
            timeframe=TimeFrame.Day,
            start=now - timedelta(days=_HISTORY_DAYS),
            end=now - _SIP_DELAY,
            adjustment=Adjustment.SPLIT,
            feed=DataFeed.SIP,
        )
    )
    if isinstance(bars, dict):
        raise DataUnavailable(f"alpaca: unexpected raw bars payload for {u}")
    out: dict[pd.Timestamp, int] = {}
    for bar in bars.data.get(u, []):
        d = bar.timestamp.astimezone(_NY).date()
        if d > session or not calendar.is_session(d):
            continue
        if (slot is Slot.DEC and d == session) or now - _SIP_DELAY < calendar.open_close(d)[1]:
            # Even just after the close the delayed SIP bar is still partial.
            continue
        out[pd.Timestamp(d)] = round(bar.close * 100)
    if len(out) < 60:
        raise DataUnavailable(f"alpaca: only {len(out)} daily bars for {u}")
    return pd.Series(out, dtype="int64").sort_index()


def _scaled_proxy(index: pd.Series, chain: ChainSnapshot, calendar: Calendar, session: date) -> pd.Series:
    """V10: the Cboe index scaled so its last knowable close equals today's own-chain 30-day IV; values in bp."""
    history = index[index.index < pd.Timestamp(session)]
    if history.empty:
        raise DataUnavailable("proxy: no index history before the session")
    scale = 1.0
    try:
        own_iv30 = surface.const_maturity_iv(surface.term_of(surface.chain_nodes(chain, calendar)), 30, min_dte=7)[0]
        scale = own_iv30 / (float(history.iloc[-1]) / 100.0)
    except DataUnavailable:
        pass  # no own IV30 today: the unscaled index is still a usable RANK series
    return (history * scale * 100.0).round().astype("int64")
