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
| Data providers (historical mirror, synthetic), report writer, **backtest engine**, recorder, baselines, paper runner | not built yet |

There is **no runnable backtest or paper loop yet**. What exists is a tested library of the parts they are built from
(about 3,200 tests, `mypy --strict` clean).

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
