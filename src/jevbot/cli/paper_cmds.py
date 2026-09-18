"""Read-only live-market diagnostics. The paper trading service is not implemented."""

from typing import Annotated

import typer

from jevbot.cli.main import get_globals

app = typer.Typer(no_args_is_help=True, help="Paper-market diagnostics (no orders).", pretty_exceptions_enable=False)


@app.callback()
def paper() -> None:
    """Paper-market commands."""


@app.command("decide")
def decide(
    ctx: typer.Context, mock: Annotated[bool, typer.Option("--mock", help="Use the offline decider on live market data.")] = False
) -> None:
    """Fetch live data and preview an entry cycle. Never submit or cancel orders."""
    import json
    from contextlib import ExitStack
    from datetime import UTC, datetime
    from uuid import uuid4

    from jevbot import ids
    from jevbot.cal import XnysCalendar
    from jevbot.errors import ConfigError, DataUnavailable
    from jevbot.jev.cache import SqliteDecisionCache
    from jevbot.jev.live import LiveJev
    from jevbot.jev.mock import MockJev
    from jevbot.jev.spend import SpendGuard
    from jevbot.paper.alpaca_client import make_clients
    from jevbot.paper.decide import decide_snapshot
    from jevbot.paper.live_data import AlpacaLiveProvider
    from jevbot.protocols import Decider
    from jevbot.types import CacheMode

    globals_ = get_globals(ctx)
    loaded = globals_.load()
    cfg = loaded.cfg
    if cfg.news.enabled == "on":
        raise ConfigError("paper decide has no news feed; use news.enabled=off or auto")
    if cfg.decider.kind == "replay" and not mock:
        raise ConfigError("paper decide cannot replay a new live snapshot; select live or mock")
    use_mock = mock or cfg.decider.kind == "mock" or (cfg.decider.kind == "auto" and not loaded.secrets.has_typesafe_key)
    if not use_mock and not loaded.secrets.has_typesafe_key:
        raise ConfigError("live paper decide requires TYPESAFE_API_KEY")
    with ExitStack() as stack:
        decider: Decider
        if use_mock:
            decider = MockJev(cfg.decider.mock_profile)
        else:
            cache = stack.enter_context(SqliteDecisionCache(loaded.data_dir / "cache" / "decisions.sqlite"))
            namespace = ids.namespace(cfg.run.experiment, cfg.jev.model, cfg.jev.refresh_generation)
            cache.ensure_namespace(namespace, cfg.jev.model, cfg.jev.model_release_date, refresh=False)
            spend = stack.enter_context(SpendGuard(loaded.data_dir / "state" / "spend.sqlite", scope="paper", cfg=cfg.jev.spend))
            decider = LiveJev(
                cfg.jev, cache, spend, api_key=loaded.secrets.typesafe_api_key or "", run_id=f"decide-{uuid4().hex}", mode=CacheMode.RECORD
            )
        stack.callback(decider.close)
        clients = make_clients(loaded.secrets, cfg.orders.http_timeout_s)
        for client in clients:
            stack.callback(client._session.close)
        calendar = XnysCalendar()
        try:
            provider = AlpacaLiveProvider.snapshot(clients, cfg, calendar, datetime.now(UTC))
        except Exception as exc:
            # Vendor exceptions may contain authentication headers or response bodies.
            raise DataUnavailable(f"live snapshot failed ({type(exc).__name__})") from None
        result = decide_snapshot(cfg, provider, calendar, decider, datetime.now(UTC))
    if globals_.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"DRY RUN | {result['session']} {result['slot']} | {result['model']} | no orders submitted")
        typer.echo(f"Hypothetical empty portfolio: ${result['hypothetical_equity_usd']:,}; indicative data; news off.")
        for record in result["records"]:
            typer.echo(json.dumps(record))
        typer.echo("Diagnostics: " + json.dumps(result["diagnostics"]))
        typer.echo("This preview omits paper-service checks. Use --json for the full limitations and results.")
