"""THE single Alpaca construction site, the paper guards and the transport layers (DESIGN.md 11.1; D2 / G7 / D18; INV-01, INV-02, INV-08).

`make_clients` is the only function in the repository that constructs an Alpaca client. Every layer of 11.1 is a floor, not a
preference: the paper host is asserted, the SDK's blind retry loop is disabled globally, a `(connect, read)` timeout is forced
onto every request, and the account is refused unless it can actually trade options at level 3.

Adapter verification against the pinned wheel (alpaca-py 0.44.0), the first task of WP08. Everything below was read out of
`.venv/lib/python3.12/site-packages/alpaca/`, not assumed:

| DESIGN assumes | verified in 0.44.0 |
|---|---|
| `TradingClient(api_key, secret_key, paper=True)` | `alpaca/trading/client.py::TradingClient.__init__`; `paper=True` selects `BaseURL.TRADING_PAPER` and sets `sandbox=True` |
| the override-URL keyword | present on `TradingClient.__init__` and on every data client; NEVER passed here (INV-01) |
| `_base_url`, `_session`, `_retry`, `_retry_wait`, `_retry_codes` | all set in `alpaca/common/rest.py::RESTClient.__init__` (defaults 3 / 3 s / `[429, 504]`) |
| `_retry = 0` means exactly one request | `_request` loops `while retry >= 0` and `_one_request` re-raises `RetryException` only `if retry > 0`, so 0 = one attempt |
| the SDK sets no timeout | `_one_request` calls `self._session.request(method, url, **opts)` with `opts` = headers / params / json only |
| `TradeAccount.last_equity` | present, `Optional[str]` - the prior-session closing equity that feeds the daily-loss halt (9.5) |
| `TradeAccount.options_trading_level` | present, `Optional[int]` (`options_approved_level` is the separate approval field) |
| `TradeAccount.trade_suspended_by_user` | present: the account-side mirror of `AccountConfiguration.suspend_trade` |
| PDT fields | `pattern_day_trader` / `daytrade_count` still exist but are never read (critique correction 10) |
| `get_clock` / `get_calendar` | `Clock(timestamp, is_open, next_open, next_close)`; `Calendar(date, open, close)` with NAIVE Eastern `open` / `close` |
| `get/set_account_configurations` | the setter takes a FULL `AccountConfiguration` and PATCHes `model_dump()`, so it must be read first |
| account activities | NOT wrapped by `TradingClient` (only `BrokerClient.get_account_activities` exists), so `paper/broker.py` uses the client's own authenticated `get("/account/activities")` |

The data clients (`OptionHistoricalDataClient`, `StockHistoricalDataClient`, `NewsClient`, `CorporateActionsClient`) all derive
from the same `RESTClient` and point at the market-data host, which is identical for paper and live accounts and can place no
order; they carry the same five private attributes and get the same retry / timeout treatment.
"""

import functools
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Final, NamedTuple

from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from jevbot.config import (
    _FORBIDDEN_ALPACA_PREFIX,  # the single source of truth for the legacy-prefix guard (INV-02); no credential literal lives here
    ENV_ALPACA_PAPER_KEY,
    ENV_ALPACA_PAPER_SECRET,
    FORBIDDEN_ALPACA_ENV,
    Secrets,
)
from jevbot.errors import BrokerError, PaperGuardError
from jevbot.types import AccountSnapshot, Cents

__all__ = [
    "DEFAULT_HTTP_TIMEOUT",
    "MIN_OPTIONS_LEVEL",
    "PAPER_BASE_URL",
    "REQUIRED_CLIENT_ATTRS",
    "AlpacaClients",
    "account_snapshot",
    "assert_client_surface",
    "assert_no_forbidden_env",
    "assert_paper_base_url",
    "assert_tradable_account",
    "base_url_of",
    "disable_sdk_retries",
    "install_default_timeout",
    "make_clients",
    "paper_credentials",
    "to_cents",
]

PAPER_BASE_URL: Final = "https://paper-api.alpaca.markets"
"""The only host an Alpaca trading client of this repository may ever talk to (INV-01)."""

DEFAULT_HTTP_TIMEOUT: Final[tuple[float, float]] = (3.05, 10.0)
"""(connect, read) seconds forced onto every request; `orders.http_timeout_s` carries the same default (G5, INV-08)."""

MIN_OPTIONS_LEVEL: Final = 3
"""Level 3 = defined-risk multi-leg spreads. Below it the strategy cannot be traded at all (11.1 step 7)."""

REQUIRED_CLIENT_ATTRS: Final[tuple[str, ...]] = ("_retry", "_retry_wait", "_retry_codes", "_session", "_base_url")
"""Private attributes this module depends on; an SDK change that drops one must fail closed, never silently."""

_PAPER_KEY_PREFIX: Final = "PK"
_TIMEOUT_MARK: Final = "_jevbot_default_timeout"


class AlpacaClients(NamedTuple):
    """The five clients of the paper runner. Built once, at boot, by `make_clients`."""

    trading: TradingClient
    options: OptionHistoricalDataClient
    stocks: StockHistoricalDataClient
    news: NewsClient
    corporate_actions: CorporateActionsClient


# ======================================================================================================================
# Guards (each one usable on its own so that `tests/guards/test_paper_only.py` can exercise it against a fake client)
# ======================================================================================================================


def assert_no_forbidden_env(env: Mapping[str, str]) -> None:
    """INV-02, step 1. `config.secrets()` has already refused these names; `make_clients` refuses them again because it is the
    last gate before a credential reaches an SDK, and a probe or a test harness may build `Secrets` by hand."""
    names = sorted(
        name
        for name in env
        if name in FORBIDDEN_ALPACA_ENV
        or (name.startswith(_FORBIDDEN_ALPACA_PREFIX) and name not in (ENV_ALPACA_PAPER_KEY, ENV_ALPACA_PAPER_SECRET))
    )
    if names:
        raise PaperGuardError(
            f"ambiguous Alpaca credentials in the environment: unset {', '.join(names)} - "
            f"only {ENV_ALPACA_PAPER_KEY} / {ENV_ALPACA_PAPER_SECRET} are ever read (INV-02)"
        )


def paper_credentials(secrets: Secrets) -> tuple[str, str]:
    """Steps 1-2: both values present and non-blank, and the key id carries the paper prefix (a hint check, D2)."""
    key, secret = secrets.alpaca_paper_key, secrets.alpaca_paper_secret
    missing = [name for name, value in ((ENV_ALPACA_PAPER_KEY, key), (ENV_ALPACA_PAPER_SECRET, secret)) if not value]
    if missing or key is None or secret is None:
        raise PaperGuardError(f"{' and '.join(missing)} is absent or blank: paper trading needs both (INV-02)")
    if not key.startswith(_PAPER_KEY_PREFIX):
        raise PaperGuardError(
            f"the Alpaca key id starts with {key[:2]!r}, not {_PAPER_KEY_PREFIX!r}: this looks like a live key and "
            f"{__name__} only ever talks to {PAPER_BASE_URL} (INV-01, INV-02)"
        )
    return key, secret


def base_url_of(client: object) -> str:
    """The client's base URL as a plain string, whether the SDK stored the enum member or a raw `str` (11.1 step 4)."""
    raw = getattr(client, "_base_url", None)
    return str(getattr(raw, "value", raw))


def assert_paper_base_url(client: object) -> None:
    """Step 4: the trading client points at the paper host - the one assertion that makes INV-01 observable at runtime."""
    url = base_url_of(client)
    if url != PAPER_BASE_URL:
        raise PaperGuardError(f"the trading client points at {url!r}, not the paper host {PAPER_BASE_URL!r} (INV-01)")


def assert_client_surface(client: object) -> None:
    """Step 5a: fail closed when the pinned SDK no longer carries an attribute the transport layers depend on."""
    missing = [name for name in REQUIRED_CLIENT_ATTRS if not hasattr(client, name)]
    if missing:
        raise PaperGuardError(
            f"{type(client).__name__} has no {', '.join(missing)}: the pinned alpaca-py surface changed, so the retry and "
            "timeout layers of 11.1 cannot be installed (INV-08)"
        )


def disable_sdk_retries(client: Any) -> None:
    """Step 5b: `_retry = 0` GLOBALLY at construction.

    The SDK's loop is `while retry >= 0` with the re-raise guarded by `retry > 0`, so 0 means exactly one request. Left at the
    default 3 it would blindly repeat a POST on 429 / 504 and could open the same position twice (G5, INV-08). It is never
    toggled per call: that would race with the deadline worker threads of `paper/broker.py`.
    """
    client._retry = 0


def install_default_timeout(client: Any, timeout: tuple[float, float]) -> None:
    """Step 6: wrap `_session.request` so a `(connect, read)` timeout is always passed (the SDK sets none; G5, INV-08).

    Idempotent: a client that is already wrapped keeps its wrapper. An explicit `timeout=` from a caller still wins.
    """
    connect, read = _valid_timeout(timeout)
    session = client._session
    original = session.request
    if getattr(original, _TIMEOUT_MARK, None) is not None:
        return

    @functools.wraps(original)
    def request(method: Any, url: Any, /, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", (connect, read))
        return original(method, url, *args, **kwargs)

    setattr(request, _TIMEOUT_MARK, (connect, read))
    session.request = request


def _valid_timeout(timeout: tuple[float, float]) -> tuple[float, float]:
    try:
        connect, read = timeout
    except (TypeError, ValueError):
        raise PaperGuardError(f"timeout must be a (connect, read) pair of seconds, got {timeout!r} (INV-08)") from None
    for name, value in (("connect", connect), ("read", read)):
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
            raise PaperGuardError(f"the {name} timeout must be a positive finite number of seconds, got {value!r} (INV-08)")
    return (float(connect), float(read))


def assert_tradable_account(trading: Any, *, now: datetime | None = None) -> AccountSnapshot:
    """Step 7: `get_account()` must report options level >= 3 and neither block flag set, else `PaperGuardError` (exit 4)."""
    snapshot = account_snapshot(trading.get_account(), now=now)
    if snapshot.options_level < MIN_OPTIONS_LEVEL:
        raise PaperGuardError(
            f"the account's options trading level is {snapshot.options_level}, but defined-risk multi-leg spreads need "
            f"level {MIN_OPTIONS_LEVEL} (11.1 step 7)"
        )
    if snapshot.trading_blocked or snapshot.account_blocked:
        raise PaperGuardError(
            f"the account cannot trade: trading_blocked={snapshot.trading_blocked}, account_blocked={snapshot.account_blocked}"
        )
    return snapshot


# ======================================================================================================================
# Account mapping (the one place a vendor account object becomes our `AccountSnapshot`)
# ======================================================================================================================


def to_cents(value: object, field: str) -> Cents:
    """A vendor money string ("100000.42") -> integer cents. Decimal, never float: money is integer everywhere (Conventions)."""
    if value is None or value == "":
        raise BrokerError(f"the broker account is missing {field}")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise BrokerError(f"the broker reported {field}={value!r}, which is not a number") from None
    return int((amount * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _optional_cents(value: object, field: str) -> Cents | None:
    return None if value is None or value == "" else to_cents(value, field)


def account_snapshot(raw: Any, *, now: datetime | None = None) -> AccountSnapshot:
    """`alpaca.trading.models.TradeAccount` -> `AccountSnapshot` (2.4).

    `last_equity` is the prior-session closing equity the daily-loss halt compares against (9.5); it stays `None` when the
    broker does not report it, and `risk.py` then falls back to the book's own `day_start_equity` alone. `suspended` is the
    account-side mirror of `AccountConfiguration.suspend_trade` (`trade_suspended_by_user`), so reading the account is enough
    for reconcile R4. The PDT fields are deliberately not read (critique correction 10).
    """
    level = getattr(raw, "options_trading_level", None)
    return AccountSnapshot(
        equity=to_cents(getattr(raw, "equity", None), "equity"),
        cash=to_cents(getattr(raw, "cash", None), "cash"),
        options_buying_power=to_cents(getattr(raw, "options_buying_power", None), "options_buying_power"),
        last_equity=_optional_cents(getattr(raw, "last_equity", None), "last_equity"),
        options_level=0 if level is None else int(level),
        trading_blocked=bool(getattr(raw, "trading_blocked", False)),
        account_blocked=bool(getattr(raw, "account_blocked", False)),
        suspended=bool(getattr(raw, "trade_suspended_by_user", False)),
        ts=now if now is not None else datetime.now(UTC),
    )


# ======================================================================================================================
# THE construction site (INV-01: `tests/guards/test_paper_only.py` asserts by AST that no other function builds a client)
# ======================================================================================================================


def make_clients(secrets: Secrets, timeout: tuple[float, float], *, env: Mapping[str, str] | None = None) -> AlpacaClients:
    """Build the five Alpaca clients of the paper runner, with every guard and layer of 11.1 applied, in that order.

    1. the credential environment is unambiguous, both paper values are present;
    2. the key id carries the paper prefix;
    3. `TradingClient(..., paper=True)` - the literal `True`, no variable - and the override-URL keyword is never passed;
    4. the resulting base URL is the paper host, else `PaperGuardError` (exit 4);
    5. every client carries the private attributes the transport layers need, and its SDK retry loop is disabled;
    6. every client's session always sends a `(connect, read)` timeout;
    7. the account is reachable, options level >= 3 and neither block flag is set.

    `env` defaults to the process environment; it is a parameter only so the guard suite can drive step 1 directly.
    """
    assert_no_forbidden_env(os.environ if env is None else env)
    key, secret = paper_credentials(secrets)
    layers = _valid_timeout(timeout)

    trading = TradingClient(api_key=key, secret_key=secret, paper=True)
    clients = AlpacaClients(
        trading=trading,
        options=OptionHistoricalDataClient(api_key=key, secret_key=secret),
        stocks=StockHistoricalDataClient(api_key=key, secret_key=secret),
        news=NewsClient(api_key=key, secret_key=secret),
        corporate_actions=CorporateActionsClient(api_key=key, secret_key=secret),
    )

    assert_paper_base_url(trading)
    for client in clients:
        assert_client_surface(client)
        disable_sdk_retries(client)
        install_default_timeout(client, layers)
    assert_tradable_account(trading)
    return clients
