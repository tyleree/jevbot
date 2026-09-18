"""`paper/alpaca_client.py`: the single construction site and every layer of DESIGN 11.1 (INV-01, INV-02, INV-08).

The construction site itself is exercised end to end by pointing the module's five client names at the doubles of
`tests/fixtures/fake_alpaca.py`: `make_clients` then runs unchanged - the literal `paper=True`, the base-URL assertion, the
global `_retry = 0`, the forced timeout and the account check all happen for real, offline.
"""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from alpaca.common.enums import BaseURL

from jevbot.config import Secrets
from jevbot.errors import BrokerError, PaperGuardError
from jevbot.paper import alpaca_client
from jevbot.paper.alpaca_client import (
    DEFAULT_HTTP_TIMEOUT,
    MIN_OPTIONS_LEVEL,
    PAPER_BASE_URL,
    REQUIRED_CLIENT_ATTRS,
    account_snapshot,
    assert_client_surface,
    assert_no_forbidden_env,
    assert_paper_base_url,
    assert_tradable_account,
    base_url_of,
    disable_sdk_retries,
    install_default_timeout,
    make_clients,
    paper_credentials,
    to_cents,
)
from tests.fixtures.fake_alpaca import FakeDataClient, FakeRestClient, FakeTradingClient

# the host make_clients must refuse; spelled here (a test file, never scanned by the 11.1 grep) so the refusal is real
LIVE_HOST = "https://api.alpaca.markets"

PAPER_SECRETS = Secrets(alpaca_paper_key="PKTESTDUMMY", alpaca_paper_secret="SKTESTDUMMYSECRET")


class _Recorder:
    """Stands in for one SDK class: records the keyword arguments `make_clients` passed and returns a double."""

    def __init__(self, factory: type[FakeRestClient], **overrides: object) -> None:
        self.factory = factory
        self.overrides = overrides
        self.calls: list[dict[str, object]] = []
        self.instances: list[FakeRestClient] = []

    def __call__(self, **kwargs: object) -> FakeRestClient:
        self.calls.append(dict(kwargs))
        kwargs.pop("paper", None)
        built = self.factory(**{**kwargs, **self.overrides})  # type: ignore[arg-type]
        self.instances.append(built)
        return built


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Recorder]:
    """Point the five construction names at doubles; yield the trading recorder."""
    trading = _Recorder(FakeTradingClient)
    monkeypatch.setattr(alpaca_client, "TradingClient", trading)
    for name in ("OptionHistoricalDataClient", "StockHistoricalDataClient", "NewsClient", "CorporateActionsClient"):
        monkeypatch.setattr(alpaca_client, name, _Recorder(FakeDataClient))
    yield trading


# ======================================================================================================================
# The construction site, end to end
# ======================================================================================================================


def test_make_clients_builds_all_five_and_applies_every_layer(wired: _Recorder) -> None:
    clients = make_clients(PAPER_SECRETS, DEFAULT_HTTP_TIMEOUT)

    assert [type(c).__name__ for c in clients] == ["FakeTradingClient", *["FakeDataClient"] * 4]
    assert base_url_of(clients.trading) == PAPER_BASE_URL
    # step 5: alpaca-py's blind retry loop is off on EVERY client, not just the trading one (G5, INV-08)
    assert [c._retry for c in clients] == [0, 0, 0, 0, 0]
    # step 6: every session always sends the (connect, read) pair
    for client in clients:
        client._session.request("GET", "/whatever")
        assert client._session.calls[-1][2]["timeout"] == DEFAULT_HTTP_TIMEOUT


def test_the_trading_client_is_constructed_with_the_paper_flag_and_no_url_override(wired: _Recorder) -> None:
    make_clients(PAPER_SECRETS, DEFAULT_HTTP_TIMEOUT)

    (call,) = wired.calls
    assert call["paper"] is True  # INV-01: the literal True, never a variable
    assert set(call) == {"api_key", "secret_key", "paper"}  # the override-URL keyword is never passed


def test_make_clients_forwards_the_paper_credentials_only(wired: _Recorder) -> None:
    make_clients(PAPER_SECRETS, DEFAULT_HTTP_TIMEOUT)

    (call,) = wired.calls
    assert call["api_key"] == "PKTESTDUMMY"
    assert call["secret_key"] == "SKTESTDUMMYSECRET"


def test_make_clients_refuses_a_non_paper_base_url_before_the_account_is_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    trading = _Recorder(FakeTradingClient, base_url=LIVE_HOST)
    monkeypatch.setattr(alpaca_client, "TradingClient", trading)
    for name in ("OptionHistoricalDataClient", "StockHistoricalDataClient", "NewsClient", "CorporateActionsClient"):
        monkeypatch.setattr(alpaca_client, name, _Recorder(FakeDataClient))

    with pytest.raises(PaperGuardError, match="paper host"):
        make_clients(PAPER_SECRETS, DEFAULT_HTTP_TIMEOUT)

    built = trading.instances[0]
    assert isinstance(built, FakeTradingClient)
    assert built.calls == []  # the guard fires before a single endpoint is touched


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"options_trading_level": 2}, "options trading level is 2"),
        ({"options_trading_level": None}, "options trading level is 0"),
        ({"trading_blocked": True}, "trading_blocked=True"),
        ({"account_blocked": True}, "account_blocked=True"),
    ],
)
def test_make_clients_refuses_an_account_that_cannot_trade(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], needle: str
) -> None:
    monkeypatch.setattr(alpaca_client, "TradingClient", _Recorder(FakeTradingClient, **overrides))
    for name in ("OptionHistoricalDataClient", "StockHistoricalDataClient", "NewsClient", "CorporateActionsClient"):
        monkeypatch.setattr(alpaca_client, name, _Recorder(FakeDataClient))

    with pytest.raises(PaperGuardError, match=needle):
        make_clients(PAPER_SECRETS, DEFAULT_HTTP_TIMEOUT)


def test_make_clients_reads_the_process_environment_by_default(wired: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = "ALPACA" + "_API_KEY"  # assembled so this file carries no legacy credential name
    monkeypatch.setenv(legacy, "AKLIVE")

    with pytest.raises(PaperGuardError, match=legacy):
        make_clients(PAPER_SECRETS, DEFAULT_HTTP_TIMEOUT)


# ======================================================================================================================
# Step 1-2: credentials and environment
# ======================================================================================================================


def test_forbidden_env_names_are_refused() -> None:
    for name in ("ALPACA" + "_API_KEY", "ALPACA" + "_SECRET_KEY", "AP" + "CA_API_BASE_URL", "AP" + "CA_ANYTHING_ELSE"):
        with pytest.raises(PaperGuardError, match="ambiguous Alpaca credentials"):
            assert_no_forbidden_env({name: "x"})


def test_the_two_names_we_do_read_are_not_forbidden() -> None:
    assert_no_forbidden_env({"ALPACA_PAPER_KEY": "PK1", "ALPACA_PAPER_SECRET": "SK1", "PATH": "/usr/bin"})


@pytest.mark.parametrize(
    "secrets",
    [
        Secrets(alpaca_paper_key=None, alpaca_paper_secret="SK"),
        Secrets(alpaca_paper_key="   ", alpaca_paper_secret="SK"),  # blank is absent (D29)
        Secrets(alpaca_paper_key="PK", alpaca_paper_secret=""),
    ],
)
def test_absent_or_blank_credentials_are_refused(secrets: Secrets) -> None:
    with pytest.raises(PaperGuardError, match="absent or blank"):
        paper_credentials(secrets)


def test_a_live_looking_key_is_refused_and_never_echoed() -> None:
    with pytest.raises(PaperGuardError) as caught:
        paper_credentials(Secrets(alpaca_paper_key="AKLIVEKEY0001", alpaca_paper_secret="SKLIVESECRET"))

    message = str(caught.value)
    assert "'AK'" in message  # the two-character hint is enough to diagnose it
    assert "AKLIVEKEY0001" not in message and "SKLIVESECRET" not in message


def test_a_paper_key_passes() -> None:
    assert paper_credentials(PAPER_SECRETS) == ("PKTESTDUMMY", "SKTESTDUMMYSECRET")


# ======================================================================================================================
# Step 4-6: base URL, SDK surface, retries, timeout
# ======================================================================================================================


def test_base_url_of_handles_the_enum_and_a_plain_string() -> None:
    assert base_url_of(FakeRestClient(base_url=BaseURL.TRADING_PAPER)) == PAPER_BASE_URL
    assert base_url_of(FakeRestClient(base_url=PAPER_BASE_URL)) == PAPER_BASE_URL
    assert base_url_of(FakeRestClient(base_url=LIVE_HOST)) == LIVE_HOST


def test_assert_paper_base_url_accepts_the_paper_host_and_refuses_anything_else() -> None:
    assert_paper_base_url(FakeRestClient(base_url=BaseURL.TRADING_PAPER))
    with pytest.raises(PaperGuardError, match="INV-01"):
        assert_paper_base_url(FakeRestClient(base_url=LIVE_HOST))


@pytest.mark.parametrize("attribute", REQUIRED_CLIENT_ATTRS)
def test_a_missing_private_attribute_fails_closed(attribute: str) -> None:
    client = FakeRestClient()
    delattr(client, attribute)
    with pytest.raises(PaperGuardError, match=attribute):
        assert_client_surface(client)


def test_the_pinned_wheel_itself_still_matches_what_section_11_1_assumes() -> None:
    """The adapter-verification step of 11.1 as a regression test, against the REAL class (no request is made)."""
    from alpaca.trading.client import TradingClient

    real = TradingClient(api_key="PKTESTDUMMY", secret_key="SKTESTDUMMYSECRET", paper=True)

    assert base_url_of(real) == PAPER_BASE_URL  # `paper=True` alone selects the paper host
    assert all(hasattr(real, name) for name in REQUIRED_CLIENT_ATTRS)
    assert real._retry > 0  # the default we must turn off; `disable_sdk_retries` is what makes it exactly one request
    assert 429 in real._retry_codes  # the statuses alpaca-py would blindly repeat a POST on (G5)
    disable_sdk_retries(real)
    assert real._retry == 0
    # the account fields section 11.1 asked us to verify, on the real model
    from alpaca.trading.models import TradeAccount

    assert {"last_equity", "options_trading_level", "trade_suspended_by_user"} <= set(TradeAccount.model_fields)


def test_disable_sdk_retries_sets_exactly_one_request() -> None:
    client = FakeRestClient()
    assert client._retry > 0  # the wheel's own default, which would blindly repeat a POST on 429 / 504
    disable_sdk_retries(client)
    assert client._retry == 0


def test_the_timeout_wrapper_is_installed_once_and_defaults_every_call() -> None:
    client = FakeRestClient()
    install_default_timeout(client, (1.5, 4.0))
    wrapped = client._session.request
    install_default_timeout(client, (9.0, 9.0))  # idempotent: the first wrapper stays
    assert client._session.request is wrapped

    client._session.request("GET", "/account")
    assert client._session.calls[-1][2]["timeout"] == (1.5, 4.0)


def test_an_explicit_timeout_still_wins() -> None:
    client = FakeRestClient()
    install_default_timeout(client, (1.5, 4.0))
    client._session.request("POST", "/orders", timeout=(0.5, 0.5))
    assert client._session.calls[-1][2]["timeout"] == (0.5, 0.5)


def test_the_wrapper_passes_the_payload_through_untouched() -> None:
    client = FakeRestClient()
    install_default_timeout(client, DEFAULT_HTTP_TIMEOUT)
    client._session.request("POST", "/orders", json={"qty": 1}, headers={"h": "v"}, allow_redirects=False)
    method, url, kwargs = client._session.calls[-1]
    assert (method, url) == ("POST", "/orders")
    assert kwargs["json"] == {"qty": 1}
    assert kwargs["headers"] == {"h": "v"}
    assert kwargs["allow_redirects"] is False


@pytest.mark.parametrize("timeout", [(0.0, 10.0), (-1.0, 10.0), (3.05, float("nan")), (3.05, float("inf")), (3.05,), "x"])
def test_an_unusable_timeout_is_refused(timeout: object) -> None:
    with pytest.raises(PaperGuardError, match="INV-08"):
        install_default_timeout(FakeRestClient(), timeout)  # type: ignore[arg-type]


# ======================================================================================================================
# Step 7 and the account mapping
# ======================================================================================================================


def test_assert_tradable_account_returns_the_snapshot_at_level_three() -> None:
    snapshot = assert_tradable_account(FakeTradingClient(), now=datetime(2026, 9, 17, 19, 45, tzinfo=UTC))
    assert snapshot.options_level == MIN_OPTIONS_LEVEL
    assert snapshot.equity == 10_000_000  # $100,000.00 -> cents, hand-computed
    assert snapshot.last_equity == 10_200_000  # $102,000.00: the PRIOR session's close (9.5)
    assert snapshot.ts == datetime(2026, 9, 17, 19, 45, tzinfo=UTC)


def test_account_snapshot_maps_every_field_we_depend_on() -> None:
    client = FakeTradingClient(
        equity="12345.67",
        cash="-89.10",
        options_buying_power="2500.00",
        last_equity="12000.01",
        options_trading_level=4,
        suspended=True,
    )
    snapshot = account_snapshot(client.get_account(), now=datetime(2026, 9, 17, tzinfo=UTC))

    assert snapshot.equity == 1_234_567
    assert snapshot.cash == -8_910
    assert snapshot.options_buying_power == 250_000
    assert snapshot.last_equity == 1_200_001
    assert snapshot.options_level == 4
    assert snapshot.suspended is True  # the account-side mirror of AccountConfiguration.suspend_trade
    assert snapshot.trading_blocked is False and snapshot.account_blocked is False


def test_a_broker_without_last_equity_maps_to_none_rather_than_zero() -> None:
    # 9.5 takes the WORSE of the book loss and the broker loss; a zero here would fake a -100% broker day
    snapshot = account_snapshot(FakeTradingClient(last_equity=None).get_account())
    assert snapshot.last_equity is None


def test_a_missing_required_money_field_is_an_error_not_a_zero() -> None:
    client = FakeTradingClient()
    client.account_raw["options_buying_power"] = None
    with pytest.raises(BrokerError, match="options_buying_power"):
        account_snapshot(client.get_account())


@pytest.mark.parametrize(
    ("text", "cents"),
    [("123.45", 12_345), ("-99.99", -9_999), ("0.005", 1), ("-0.005", -1), ("100000", 10_000_000), ("0", 0)],
)
def test_money_strings_become_integer_cents(text: str, cents: int) -> None:
    assert to_cents(text, "equity") == cents


def test_a_non_numeric_money_field_is_refused() -> None:
    with pytest.raises(BrokerError, match="not a number"):
        to_cents("about a hundred", "equity")
