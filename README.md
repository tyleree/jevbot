# jevbot

An options trading bot whose decision core is [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
a "System One" model that answers typed questions about a market state with calibrated probabilities.
Code does all the maths, sizing and risk; Jev only picks from closed lists and supplies probabilities.

**Paper trading and historical backtests only.** There is no live-money code path, by design: Alpaca clients are
constructed in exactly one place with `paper=True`, and tests scan the source to keep it that way.

## Status: under construction

| Layer | State |
|---|---|
| Foundation (types, contracts, config, option maths, calendar) | done, reviewed |
| Data core, state builder, Jev client + decision cache, rules, risk engine + kill switch, fills + ledger, evaluation maths, Alpaca paper broker | implemented and tested; review partial, see [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) |
| Synthetic provider, shared market tables, live Alpaca snapshot provider, lean decision cycle | implemented; diagnostic use |
| `paper decide` | runnable dry-run entry preview; never submits orders |
| Historical mirror provider, report writer, **backtest engine**, recorder, baselines, paper service | not built yet |

There is **no runnable backtest or autonomous paper loop yet**. The read-only `paper decide` command now connects live
market data to the state builder, decider, rules, candidate generator and risk engine.

## Preview a decision

Set the paper credentials and `JEVBOT_DATA` as described in `.env.example`. The data directory must be outside this
repository and private (mode `700` on Linux). Then, from the repository:

```bash
uv run jevbot paper decide --mock
uv run jevbot paper decide --json
```

Both commands fetch live Alpaca market data, Cboe volatility-index history and Treasury bill rates. `--mock` uses the
deterministic offline decider; without it, `decider.kind=auto` uses Jev when `TYPESAFE_API_KEY` is available and otherwise
uses the mock. Explicit `live` requires that key; `replay` is refused for a new live snapshot. Live Jev calls use the
shared decision cache and the paper token budget. `--json` writes machine-readable output to stdout and the paper-only
banner to stderr. Keep output private; it is not committed to the repository.

This is an **entry diagnostic**, sized against an empty hypothetical portfolio using `run.initial_equity_usd`.
It does not manage existing positions, reconcile the account, or submit/cancel orders. The output includes decisions,
risk verdicts, missing required features and the share of IV history derived from a scaled index proxy. It uses indicative
quotes, has no open-interest/news/event feeds, and omits perturbation confirmation and paper-service health gates.
`news.enabled=on` is refused rather than silently ignored. A hypothetical approval is not an executable trading approval.

## Develop

Python 3.12, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check . && uv run mypy && uv run pytest
```

## Documents

- [`docs/design/DESIGN.md`](docs/design/DESIGN.md) — the binding specification (work packages, invariants, contracts)
- [`docs/research/03-decisions.md`](docs/research/03-decisions.md) — v1 decisions and defaults
- [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — open review findings

## Terms

- Results stay private. TypeSafe's customer agreement (2.3(f)) forbids publishing benchmarks or performance
  information about the service, so no backtest or paper-trading results are ever committed here.
- No surrogate or distilled models are trained on Jev outputs (agreement 2.3(b)).
- Market data used by the bot is for personal use; vendor data is never committed.

Nothing here is investment advice.
