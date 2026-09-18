"""Offline regression tests for the live-data dry run; no credentials or network."""

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from jevbot.cli.main import app
from jevbot.config import Config
from jevbot.data.market import _bars_table, build_market_data
from jevbot.jev.mock import MockJev
from jevbot.paper import live_data
from jevbot.paper.decide import decide_snapshot
from jevbot.paper.live_data import AlpacaLiveProvider, _daily_closes, _raw_chain, current_session


@pytest.fixture
def snapshot(factory_chain, xnys_calendar):
    chain = factory_chain
    sessions = xnys_calendar.sessions(date(2023, 1, 3), chain.key.session)
    idx = pd.to_datetime(sessions)
    closes = pd.Series(np.round(43000 + np.arange(len(idx)) * 12 + 200 * np.sin(np.arange(len(idx)))), index=idx, dtype="int64")
    proxy = pd.Series(2200 + np.arange(len(idx)) % 400, index=idx)
    return AlpacaLiveProvider(
        key=chain.key, chains={"SPY": chain}, closes={"SPY": closes}, vol_index={}, iv_proxy={"SPY": proxy}, rate=0.04
    )


def one_underlying():
    cfg = Config()
    return msgspec.structs.replace(cfg, universe=msgspec.structs.replace(cfg.universe, underlyings=("SPY",)))


def test_real_pipeline(snapshot, xnys_calendar):
    decider = Mock(wraps=MockJev())
    decider.model = "mock-1"
    chain = snapshot.get_chain("SPY", snapshot.key)
    result = decide_snapshot(one_underlying(), snapshot, xnys_calendar, decider, chain.knowable_at)
    assert result["dry_run"] is True
    assert result["orders_submitted"] == 0
    assert result["diagnostics"]["SPY"]["missing_required"] == []
    assert result["diagnostics"]["SPY"]["iv_history_proxy_pct"] > 90
    assert decider.decide.call_count == 1
    assert any(r["kind"] == "decision" and r["payload"]["requests"] for r in result["records"])


def test_insufficient_history_does_not_call_decider(snapshot, xnys_calendar):
    snapshot._iv_proxy = {}
    decider = Mock(wraps=MockJev())
    decider.model = "mock-1"
    result = decide_snapshot(one_underlying(), snapshot, xnys_calendar, decider, snapshot.get_chain("SPY", snapshot.key).knowable_at)
    assert "iv_rank" in result["diagnostics"]["SPY"]["missing_required"]
    decider.decide.assert_not_called()
    assert result["hypothetical_orders"] == 0


def test_proxy_never_overwrites_snapshot_or_leaks_at_previous_close(snapshot, xnys_calendar):
    cfg = one_underlying()
    market = build_market_data(cfg, xnys_calendar, snapshot, iv_proxy=snapshot.iv_proxy())
    daily = market.tables["daily:SPY"]
    frame = daily.frame()
    own = frame[frame.session == pd.Timestamp(snapshot.key.session)]
    assert len(own) == 1
    assert own.iloc[0].source != "proxy"
    previous = xnys_calendar.prev_session(snapshot.key.session)
    at_close = daily.asof(xnys_calendar.open_close(previous)[1])
    assert pd.Timestamp(previous) not in set(at_close.session)


def test_close_only_bar_does_not_invent_known_open(xnys_calendar):
    session = date(2024, 5, 17)
    opened, closed = xnys_calendar.open_close(session)
    table = _bars_table("SPY", pd.Series([50000], index=[pd.Timestamp(session)]), xnys_calendar)
    assert table.asof(opened + timedelta(minutes=2)).empty
    assert len(table.asof(xnys_calendar.next_open_after(closed))) == 1


@pytest.mark.parametrize(
    "stamp, expected, slot",
    [
        ("2024-05-20T12:00:00+00:00", date(2024, 5, 17), "eod"),
        ("2024-05-20T15:00:00+00:00", date(2024, 5, 20), "dec"),
        ("2024-05-20T21:00:00+00:00", date(2024, 5, 20), "eod"),
        ("2024-05-19T15:00:00+00:00", date(2024, 5, 17), "eod"),
    ],
)
def test_session_selection(xnys_calendar, stamp, expected, slot):
    session, actual = current_session(xnys_calendar, datetime.fromisoformat(stamp))
    assert (session, actual.value) == (expected, slot)


def test_naive_clock_refused(xnys_calendar):
    with pytest.raises(ValueError, match="timezone-aware"):
        current_session(xnys_calendar, datetime(2024, 5, 20))  # noqa: DTZ001


def test_sip_partial_bar_after_close_is_excluded(xnys_calendar):
    session = date(2024, 5, 17)
    sessions = xnys_calendar.sessions(date(2024, 1, 2), session)
    bars = [
        SimpleNamespace(timestamp=datetime.combine(d, datetime.min.time(), tzinfo=UTC) + timedelta(hours=4), close=500.0) for d in sessions
    ]
    clients = SimpleNamespace(stocks=SimpleNamespace(get_stock_bars=lambda req: SimpleNamespace(data={"SPY": bars})))
    closed = xnys_calendar.open_close(session)[1]
    _, slot = current_session(xnys_calendar, closed + timedelta(minutes=1))
    partial = _daily_closes(clients, "SPY", session, slot, closed + timedelta(minutes=1), xnys_calendar)
    complete = _daily_closes(clients, "SPY", session, slot, closed + timedelta(minutes=17), xnys_calendar)
    assert pd.Timestamp(session) not in partial.index
    assert pd.Timestamp(session) in complete.index


def test_raw_quotes_skip_crossed_and_missing():
    quote = SimpleNamespace(bid_price=1.0, ask_price=1.1, bid_size=3, ask_size=4, timestamp=datetime(2024, 5, 17, 20, tzinfo=UTC))
    good = SimpleNamespace(latest_quote=quote, implied_volatility=0.2)
    crossed = SimpleNamespace(latest_quote=SimpleNamespace(**{**vars(quote), "bid_price": 2.0}), implied_volatility=0.2)
    frame = _raw_chain(
        {"SPY240621C00500000": good, "SPY240621P00500000": crossed, "SPY240621C00510000": SimpleNamespace(latest_quote=None)}
    )
    assert len(frame) == 1
    assert frame.iloc[0].bid == 100
    assert frame.iloc[0].strike_milli == 500000
    assert frame.oi_prev.isna().all()


def test_cli_json_and_no_broker_mutations(monkeypatch, snapshot, tmp_path):
    import json

    from jevbot.paper import alpaca_client

    class ReadOnlyClient:
        _session = SimpleNamespace(close=lambda: None)

        def __getattr__(self, name):
            raise AssertionError(f"Unexpected broker call: {name}")

    monkeypatch.setattr(alpaca_client, "make_clients", lambda *args: (ReadOnlyClient(),))
    monkeypatch.setattr(live_data.AlpacaLiveProvider, "snapshot", lambda *args: snapshot)
    result = CliRunner().invoke(
        app, ["paper", "decide", "--mock", "--json", "-o", 'universe.underlyings=["SPY"]', "-o", f"paths.env_file={tmp_path / 'absent'}"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["orders_submitted"] == 0
    assert payload["model"] == "mock-1"


def test_help_does_not_fetch():
    result = CliRunner().invoke(app, ["paper", "decide", "--help"])
    assert result.exit_code == 0
    assert "Never submit" in result.output


def test_decider_outage_is_a_no_trade(snapshot, xnys_calendar):
    from jevbot.errors import DeciderTransportError

    decider = Mock(wraps=MockJev())
    decider.model = "mock-1"
    decider.decide.side_effect = DeciderTransportError("unavailable")
    result = decide_snapshot(one_underlying(), snapshot, xnys_calendar, decider, snapshot.get_chain("SPY", snapshot.key).knowable_at)
    decisions = [r["payload"] for r in result["records"] if r["kind"] == "decision"]
    assert decisions[0]["rules"]["action"] == "no_trade"
    assert decisions[0]["requests"][0]["error"] == "DeciderTransportError"
    assert result["hypothetical_orders"] == 0


@pytest.mark.parametrize("price", [-1.0, float("inf"), float("nan")])
def test_invalid_spot_refused(price):
    from jevbot.errors import DataUnavailable

    with pytest.raises(DataUnavailable):
        live_data._spot(None, SimpleNamespace(price=price))


def test_infinite_quote_falls_back_to_trade():
    assert live_data._spot(SimpleNamespace(bid_price=1.0, ask_price=float("inf")), SimpleNamespace(price=500.0)) == 50000


def test_cli_live_uses_shared_cache_and_paper_spend(monkeypatch, snapshot, tmp_path):
    from jevbot.jev import live
    from jevbot.paper import alpaca_client

    observed = {}

    def fake_live(cfg, cache, spend, **kwargs):
        observed.update(cache=cache.path, scope=spend.scope, mode=kwargs["mode"])
        return MockJev()

    monkeypatch.setattr(live, "LiveJev", fake_live)
    monkeypatch.setattr(alpaca_client, "make_clients", lambda *args: ())
    monkeypatch.setattr(live_data.AlpacaLiveProvider, "snapshot", lambda *args: snapshot)
    result = CliRunner().invoke(
        app,
        [
            "paper",
            "decide",
            "--json",
            "-o",
            'universe.underlyings=["SPY"]',
            "-o",
            "decider.kind=live",
            "-o",
            f"paths.env_file={tmp_path / 'absent'}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert observed["scope"] == "paper"
    assert observed["mode"].value == "record"
    assert observed["cache"].name == "decisions.sqlite"
