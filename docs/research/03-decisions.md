# Project decisions (v1)

Date: 2026-09-17. These are the orchestrator's defaults for the open questions in
`01-build-brief.md` section 9 and the gaps in `02-critique-and-corrections.md`.
Precedence when documents disagree: **this file > 02-critique > 01-brief**.
Everything marked *(user-tunable)* lives in config, not code.

## Scope

- D1. Project name and Python package: `jevbot`. Repo: `/home/tyler/jevbot` (WSL2 Ubuntu-24.04, Python 3.12, `uv`).
  Runtime data lives OUTSIDE git in `$JEVBOT_DATA` (default `/home/tyler/jevbot-data`).
- D2. Modes: historical **backtest** and Alpaca **PAPER** trading only. There is no live-trading code path at all.
  Paper-only enforcement is layered exactly as critique G7 describes (single construction site, literal `paper=True`,
  never `url_override`, startup assertion on the paper base URL, `PK` key-prefix hint check, env vars named
  `ALPACA_PAPER_KEY` / `ALPACA_PAPER_SECRET`, and a test that greps the source tree for forbidden strings).
- D3. Universe *(user-tunable)*: SPY, QQQ, IWM. Structures: long call, long put, call/put debit vertical,
  call/put credit vertical, iron condor. Defined-risk only. No naked shorts, no calendars/diagonals, no single names.
- D4. Cadence: one decision cycle per session near the close. Backtest headline rule: decide on snapshot D,
  fill on snapshot D+1. Sensitivity: same-snapshot fill at the worst-case band. All times are offsets from that
  day's calendar close (early closes handled), never hard-coded clock times.
- D5. Expiry policy (critique G4): never hold any position into its expiration day. Exit at the last decision
  snapshot with DTE >= 1. Minimum DTE at entry and a management "time exit" *(user-tunable, default exit at DTE <= 7
  for short-premium structures)*. This rule is identical in backtest and paper.

## Jev

- D6. `typesafe-sdk==0.6.0` pinned; model pinned to `jev-1.13.0`; a run FAILS CLOSED if `resp.model` differs.
  Questions are authored in the raw-dict form. The SDK is hidden behind our own `Decider` interface.
- D7. Deciders: `LiveJev` (SDK), `ReplayJev` (cache only, a miss is a hard error), `MockJev` (deterministic,
  rule-based on the state buckets; doubles as the "Jev-off" baseline and is the default when no API key is present).
  The LLM-adapter control arm is out of scope for v1 (leave the seam, do not implement).
- D8. Decision cache (critique G3): always send the FULL question batch for a request type; cache key =
  sha256(canonical JSON of {v, model, state, question_set_hash, question}). No `sample` index, no uid in state.
  Robustness is measured with deterministic meaning-preserving perturbations (permuted Choice option order,
  permuted state key order, bucket-only vs value+bucket rendering); each variant is content-addressed.
  Modes: `record`, `replay`, `refresh`(new namespace). SQLite WAL, never evicted, manifest hash in every report.
- D9. Spend guard (critique G12): per-run and per-day input-token ceilings computed from `usage.input_tokens`
  *(user-tunable)*; hard stop when exceeded. Bounded concurrency + token bucket under the documented rate limits.
- D10. "Step 0" live probes (determinism of identical requests, batch-composition independence, option-order and
  key-order sensitivity) ship as a scripted command that records its outputs. It cannot run until the user has a key.
- D11. What Jev is for (critique G9): judgement over inputs code cannot judge. News/event TEXT is a first-class
  v1 input behind `news.enabled` (default: on when Alpaca keys are present, otherwise off), entity-masked,
  sanitised, size-capped, treated as hostile input. Text-derived answers may veto or rank, never raise limits.
  The ablation "Jev on buckets-only vs deterministic mapping of the same buckets" is a registered baseline.

## Evaluation

- D12. Evidence tiers C/B/A exactly as brief 7.1, enforced in code; every Tier C report carries the
  "not evidence of model skill" banner. Primary pre-registered endpoint = calibration of Jev probabilities
  against realised outcomes and against the option-implied probability (Brier skill score). P&L is secondary.
  Short-horizon (1-5 session) machine-resolvable evaluation-only questions are included so outcomes accumulate
  faster (critique G1). A forced model-version change starts a new experiment namespace.
- D13. Fill model: ORATS headline + worst-case (cross the spread) + best-case (mid) bands, always all three.
  Alpaca paper fills are a plumbing test only; Tier A P&L is computed from SHADOW fills using our own fill model
  on quotes we recorded at decision time (critique G2). No extra haircut on Alpaca paper P&L.
- D14. Fees applied in our ledger for both backtest and paper *(user-tunable, defaults from critique G11)*:
  ORF 0.015, OCC 0.025 per contract per side; TAF 0.00329 and SEC fee on sells; commissions 0.
- D15. Baselines 1-7 of brief 7.6 run through the identical engine. Stationary bootstrap, reliability curves, ECE,
  Brier decomposition and Deflated Sharpe are implemented in-repo (numpy only); no scikit-learn / arch dependency.
- D16. Results stay private (TypeSafe MCA 2.3(f)). README states this. No surrogate/distilled models (MCA 2.3(b)).

## Risk *(all user-tunable; conservative starting values)*

- D17. Max defined loss per trade 1% of equity; aggregate open max-loss 10%; max open structures 6; max new
  structures per day 2; one structure per underlying per direction; daily loss 2% => halt new entries for the day;
  peak-to-trough drawdown 8% => kill switch. Kill-switch sequence exactly as critique G6 (stop loop, cancel orders,
  close each structure with ONE mleg order using *_to_close intents, verify flat, then `suspend_trade=true`);
  re-arming requires a manual flag file. Other triggers: ledger/broker reconcile mismatch, stale quotes or clock
  skew, repeated Jev/broker errors, `resp.model` mismatch.
- D18. Broker client safety (critique G5): deterministic `client_order_id` from (decision_id, action, attempt);
  on any ambiguous failure look the order up by client id before resubmitting; enforced HTTP timeouts on every
  broker call; repricing a spread = cancel + resubmit (mleg replace is disabled).
- D19. The RiskEngine always runs AFTER the decider and nothing the decider outputs can bypass it. Jev never outputs
  a quantity, strike, price or limit. Any decider failure => no new positions; existing risk is managed purely in code.

## Data

- D20. `ChainProvider` interface with a `fidelity` tag on every run. v1 providers:
  `MirrorParquetProvider` (EOD_QUOTES; `anahatsingh-ui/options-dataset-hist`, MIT per its README, private use,
  checksummed on fetch, LICENSE kept alongside), `SyntheticProvider` (Black-Scholes from underlying + vol index;
  tests and smoke runs ONLY, never reported as evidence), `AlpacaLiveProvider` + a forward **snapshot recorder**
  that archives chains/quotes/news with local `received_at` so paper days can be replayed in the backtester.
  ThetaData and DoltHub are documented upgrade paths, not implemented in v1.
- D21. IV rank / IV-vs-realised / term structure / skew / expected move are computed in code from the chain itself
  (ATM-interpolated IV series), so no third-party IV-history dependency is needed.
- D22. Cboe vol-index CSVs (VIX, VIX9D, VIX3M, VVIX, SKEW) and the US Treasury bill-rate CSV are fetched by a
  `jevbot data fetch` command and cached as parquet with `knowable_at` = next session open. FRED is NOT used.
- D23. Scheduled events: only sources whose dates can be fetched and verified are included. v1 = FOMC decision
  dates fetched from federalreserve.gov by script (stored as CSV with source URL and fetch time). CPI/NFP and
  ETF ex-dividend dates are supported by the schema but left EMPTY unless a verified source is available
  (Alpaca corporate-actions API when keys exist). Never fabricate or approximate a date.
- D24. One `DataView(as_of)` gates every read; any access with `knowable_at > as_of` raises. StateBuilder is a pure
  deterministic function of the DataView. State holds only str/int/bool/None/list/dict (no floats, timestamps,
  tickers, absolute dates or raw price levels).
- D25. Market calendar: `exchange_calendars` (XNYS) for backtests; in paper mode time comes from Alpaca
  `get_clock()`/`get_calendar()` and the run refuses to trade if local and broker clocks differ by more than 5 s.

## Engineering

- D26. `src/` layout, `uv` project, `ruff` + `mypy` + `pytest`. Lean dependencies: numpy, pandas, pyarrow, scipy,
  msgspec (config + structs), exchange-calendars, typer, matplotlib, typesafe-sdk==0.6.0, alpaca-py==0.44.0.
  All network clients are behind interfaces and every test runs offline (`httpx2.MockTransport` for Jev, a fake
  broker for Alpaca). CI-style check: `uv run ruff check && uv run mypy && uv run pytest`.
- D27. Ledger: hash-chained, append-only (SQLite). Every decision records state hash, question hashes, cache keys,
  risk verdict, order ids, plus a provenance sidecar that is never sent to Jev.
- D28. Ops (critique G10): systemd user-service unit templates + a WSL checklist (idle timeouts, clock drift) in
  `docs/ops.md`; startup always reconciles positions/orders from the broker first; heartbeat file.
- D29. Never set `TYPESAFE_LOG_LEVEL=debug`; never add the `pypi.typesafe.ai` index; never pass an empty API key.
  Secrets come from environment / a git-ignored `.env`; nothing secret is ever logged.
- D30. The TypeSafe Claude Code plugin/skill advertised in the vendor docs is NOT installed (user's call).
