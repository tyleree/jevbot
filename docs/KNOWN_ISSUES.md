# Known issues (open review findings)

Generated 2026-09-17 from the implementation workflow journal. These findings were produced by adversarial
reviewers, but the fixer agents died on a usage limit before applying them. **None of these are fixed yet**
unless marked RESOLVED. Nothing here is live-money risk: jevbot has no live path and no runnable engine yet.

## Coverage gaps

| Package | State |
|---|---|
| WP00 foundation | implemented, reviewed, fixed |
| WP01a data core, WP02 state, WP03 jev, WP05 risk | implemented + tested, **never reviewed** |
| WP04 rules, WP06 fills/ledger, WP07a/b eval, WP08 broker | implemented + tested + reviewed, **findings below not applied** |
| WP01b data providers, WP07c report, WP09-WP13 | **not implemented** |

## WP04:1

### [MAJOR] `src/jevbot/rules.py:527` — RESOLVED

`news_align` for a neutral_range structure is `1.0 - abs(tone)`, which evaluates to 1.0 — not 0.5 — when `tone == 0`. DESIGN 7.5 prints the three branches and then annotates the line `; = 0.5 when tone == 0`, i.e. all three must agree at tone 0 so that `S_rank = (1-w)*S_core + w*news_align` stays a monotone transform of `S_core` whenever text is absent or immaterial. As written, every news-off run (`text is None` -> tone 0, the default when `news.enabled` resolves to off) gives iron condors news_align 1.0 and every directional structure 0.5, a fixed +0.03 rank advantage for condors, and `rank()` can put a strictly lower-`S_core` condor ahead of a higher-`S_core` vertical when `risk.max_new_per_day` (2) binds. The same value is carried into phase 2 via `confirm_entry`'s `base.features_ppm.get("news_align", ...)` (rules.py:412).

**Evidence:** DESIGN.md:3157 `news_align = 0.5 + 0.5*tone (bullish) | 0.5 - 0.5*tone (bearish) | 1 - abs(tone) (neutral_range)      ; = 0.5 when tone == 0`. Probe /tmp/wp04probe/p1.py with text=None: put_credit_spread news_align 500000, call_credit_spread 500000, iron_condor 1000000. Probe /tmp/wp04probe/p6.py, news off for both: condor S_core 620000 -> S_rank 639000; put_credit S_core 630050 -> S_rank 623548; `DecisionRules.rank([put_credit(SPY), condor(QQQ)], ("SPY","QQQ"))` returns ['QQQ','SPY'] — the lower-scoring condor wins the daily slot. TEST GAP that let this through: tests/property/test_prop_text_monotone.py:331-332 explicitly `continue`s past NEUTRAL structures in E3, and tests/unit/test_rules.py:669 (`test_news_align_for_bearish_and_neutral_structures`) exercises neutral only at tone 0.6 (asserting 400000), so no test covers tone == 0 for neutral_range.

**Fix:** Take the DESIGN 7.5 contradiction back to WP00 as a contract request. The reading consistent with the annotation and with 'News-off and news-on runs therefore gate and size on the same scale' is `news_align = 0.5 - 0.5 * abs(tone)` for neutral_range (0.5 at tone 0, 0.0 at |tone| 1). Whichever is chosen, add a unit case pinning news_align for all three directions at tone == 0 (and with `text is None`), and drop the NEUTRAL `continue` in the E3 property so the branch is actually covered.

### [MAJOR] `src/jevbot/rules.py:593`

The decider-down code-default branch (`core is None`) returns `watch_text=0`, discarding `pos.watch_text`, while the very same return preserves `exit_latch=pos.exit_latch`. DESIGN 7.7 says 'The latch and the counter are stored on the Position through the ledger ... and restarts keep them', and its `watch_text = 0` reset belongs to step 4 (the text rule), which this branch never reaches — no manage_text result is consulted at all here. Consequence: a single decider outage silently erases the unconfirmed-text alert progress, so the `RISK_EVENT{text_watch}` operator alert that must fire when the counter reaches `rules.text_watch_alert_sessions` (2) is pushed back, and under alternating outages it never fires. The alert is the ONLY signal the operator gets for a persistent or hostile headline that the code-side confirmation never confirms (7.7, D11, INV-16), so losing it is a real safety regression.

**Evidence:** rules.py:583-595 — `exit_latch=pos.exit_latch` on line 591 but `watch_text=0` on line 593. Probe /tmp/wp04probe/p7.py: a position with `watch_text=1` (one hostile unconfirmed session already recorded) goes through one decider-down session (`core=None`, `short_dist_code='far'`) -> `action hold watch_text 0`; the next hostile session then yields `watch_text 1` instead of 2, so `rules.text_watch_alert(decision, cfg)` stays False and no alert is raised. No test in tests/unit/test_rules.py or tests/property/test_prop_text_monotone.py exercises watch_text across a decider outage (`test_the_code_default_when_the_decider_is_down` only checks action/reason/source/pressure).

**Fix:** Change line 593 to `watch_text=pos.watch_text` so the counter is carried through the code-default branch exactly as `exit_latch` is, and add a regression case: watch_text=1 -> a code-default session -> a hostile unconfirmed session must give watch_text==2 and `text_watch_alert(...) is True`.

### [MINOR] `src/jevbot/candidates.py:285`

`_atm_iv` re-derives the per-expiry ATM sigma locally (a two-strike interpolation of the chain's own IVs at the parity forward) instead of using the single contracted source, `data/surface.atm_term` (DESIGN 3.7), whose `atm_iv_bp` is 'the fitted smile at k = 0 when a fit exists, else the two-strike interpolation'. candidates.py therefore always uses what the rest of the system treats as the FALLBACK value. `EM_T` feeds the `credit_short_min_em` (0.8 EM) guard — i.e. which strike the short leg lands on, real money — and `Candidate.short_distance_em`, so those numbers can disagree with WP02's expected moves and with the SHORT_DIST bucket shown to Jev. Section 16 makes any pure logic two same-wave packages need WP00's, and a change request is supposed to go back to WP00's owner; the implementer's report lists `contract_requests: []`.

**Evidence:** DESIGN.md:3301 `EM_T = sigma_atm(expiry) * sqrt(tau_expiry) * spot`; DESIGN.md:1584-1586 defines `atm_term` with `atm_iv_bp` = fitted-at-k=0 and `atm_iv_2s_bp` = 'always the two-strike value (QC, 5.3)'. config/default.toml ships `data.atm_iv_tolerance = 0.10` precisely to flag when the fitted and two-strike ATM IVs diverge, so the design expects them to differ. candidates.py:285-307 computes only the two-strike form; DESIGN.md:4879-4880 ('no wave-1 package ever imports another wave-1 package' / shared pure logic is WP00's).

**Fix:** Raise a contract request to WP00 for a shared `atm_sigma(chain, expiry)` (or have WP01's `surface.enrich` add an `atm_iv` column to the enriched chain that candidates.py reads), and delete the local copy. Until then, state the divergence and its bound in the module docstring next to `_atm_iv` rather than only in the hand-off report.

### [MINOR] `src/jevbot/rules.py:142`

The code-side bucket-code literals — `_CODE_DEFAULT_CLOSE_CODES = {"breached","at_strike"}` (142), `_ADVERSE_MOVES = {"adverse","strongly_adverse"}` (140), `_TREND_UP`/`_TREND_DOWN`/`_TREND_RANGE` (133-135), `_IV_RICH`/`_IV_CHEAP_RANK` (136-137) — are bare strings with no import-time check against `vocab`, unlike every other vocabulary the module uses (`_READ_IDS` vs `vocab.RULES_READABLE_IDS` at 111, `NO_MATCH_LABEL`, `REGIME_VETO_LABELS`, `VETO_IDS`, `is_reason`). A WP00 rename of `"breached"`/`"at_strike"` in `vocab.SHORT_DIST` would make the decider-down code default never close a breached short — a fail-OPEN regression — while every test still passes, because the tests hard-code the same literals rather than reading `vocab`.

**Evidence:** `grep -n 'vocab\.' src/jevbot/rules.py` shows no reference to `vocab.SHORT_DIST`, `vocab.MOVE_SINCE_ENTRY`, `vocab.TREND_DIR`, `vocab.IV_RV` or `vocab.PCTL5`. tests/unit/test_rules.py:965-977 parametrises the code default over the literal strings `('far','about_one_move','close','at_strike','breached')`; tests/property/test_prop_text_monotone.py:60 defines `CONFIRMING_MOVES` as its own literal set and filters `vocab.MOVE_SINCE_ENTRY` by it, so a vocab rename would silently shrink the confirmed-move set without failing.

**Fix:** Add an import-time assertion in rules.py of the same shape as the `_READ_IDS` check at line 111, e.g. `for code in _CODE_DEFAULT_CLOSE_CODES | _ADVERSE_MOVES: ...` against `vocab.SHORT_DIST` / `vocab.MOVE_SINCE_ENTRY`, and likewise for the trend / IV code sets against `vocab.TREND_DIR`, `vocab.IV_RV`, `vocab.PCTL5`; and have the tests parametrise from `vocab.*` instead of repeating the literals.

### [MINOR] `src/jevbot/candidates.py:780`

`_decision_keys` hardcodes the per-session snapshot preference `dec > eod > first slot`, ignoring `cfg.cadence.recorded_mode` (`"dec_exec"` = decide on `dec`, `"eod_eod"` = decide on `eod`). On a recorded archive that carries all three slots, `scan()` therefore measures feasibility on the `dec` snapshot even when the run will decide on `eod`. Those numbers are not advisory: section 8 says `data scan-candidates` records the facts that gate `register_trial(purpose="final")` and the paper boot, so the gate can pass or fail on a snapshot the run never decides on.

**Evidence:** DESIGN.md:3318-3323 ('`data scan-candidates` stores them in `manifests/scan_candidates.json` keyed by `candidate_config_hash` ... `register_trial` refuses `purpose = "final"`, and the paper boot refuses to start ...'); config/default.toml `[cadence] recorded_mode = "dec_exec"  # replay of recorded days: "dec_exec" (decide dec, fill exec, mark eod) | "eod_eod" (decide eod, fill next eod)`. candidates.py:772-781 takes no `cfg` and never reads `recorded_mode`. tests/unit/test_candidates.py:882 (`test_scan_uses_one_key_per_session`) pins the hardcoded `dec` preference with a dec+eod pair, so the eod_eod case is untested.

**Fix:** Pass `cfg` into `_decision_keys` and pick `Slot.EOD` first when `cfg.cadence.recorded_mode == "eod_eod"`, otherwise `Slot.DEC`; add a test for each mode over the same dec+eod fixture.

### [MINOR] `src/jevbot/candidates.py:658`

The per-leg liquidity re-check in `_build` is unreachable dead code: every leg is drawn from `_rows()` (candidates.py:427-440), which already applies `structmath.leg_liquidity_rejects` with the same `sold` flag and the same `cfg.liquidity`, against the same snapshot quotes. So no `Candidate` can ever carry a `liq:bid` / `liq:crossed` / `liq:spread` / `liq:oi` reject, although DESIGN 2.3 and section 8 describe a Candidate with non-empty `rejects` as 'a priced structure that fails liquidity / economics', and the 15.1 `candidates` row's 'liquidity rejects via `structmath`' item is only ever exercised through `eligible()`, never through `build()`'s reject list.

**Evidence:** candidates.py:437 (`if structmath.leg_liquidity_rejects(quote, sold=sold, cfg=cfg): continue`) vs candidates.py:659-660 (the same call over the same legs/quotes). tests/unit/test_candidates.py:684 asserts exactly this: `assert candidate.rejects == ()  # the traded legs are always drawn from eligible rows`. Coverage confirms lines around 659-660 are executed but the `rejects.extend(...)` never yields a code; `vocab.LIQUIDITY_REJECTS` members are unreachable on the `Candidate.rejects` path.

**Fix:** Either drop the redundant loop and say in the docstring that the liquidity filter is enforced at selection time so `Candidate.rejects` carries only economics / ex-div codes, or keep it as an assertion (`raise InvariantError` if it ever fires) so a future change to `_rows` cannot silently emit an untested reject path. Also record with WP00 that `liq:*` is unreachable in `CandidateReject`/`Candidate`, since `vocab.CANDIDATE_REJECTS` implies otherwise.

### [MINOR] `src/jevbot/candidates.py:820`

`scan()` lazily imports `jevbot.fills.BandFillModel` (WP06) as the default fill model. Section 16 states that no wave-1 package ever imports another wave-1 package, and the documented 'lazy cross-wave convenience' exemption is explicitly scoped to CLI files ('the only places a CLI file reaches outside its own package'); `candidates.py` is not a CLI file, and the listed edge is the opposite direction (`data scan-candidates` -> `jevbot.candidates.scan`). The default path is also entirely untested (candidates.py:820-822 uncovered), so a signature change in WP06's `BandFillModel.__init__` would break `data scan-candidates` with nothing in WP04 catching it.

**Evidence:** DESIGN.md:4879-4880 and 4887-4890; DESIGN.md:3267 prints `scan(cfg, provider, calendar, start, end)` with no fill-model parameter. Coverage report for this suite lists `820-822` as missing. I confirmed the path does currently work (/tmp/wp04probe/p3.py: 'lazy BandFillModel path OK; rows 7'), so this is an ownership/coverage issue, not a break today.

**Fix:** Make `fill_model` a required parameter of `scan()` and have the caller (`cli/data_cmds.py`, WP01, which already imports `jevbot.candidates.scan` lazily) construct the `BandFillModel` — that keeps the cross-package edge inside a CLI file where section 16 licenses it. If the printed signature must be preserved verbatim, raise it as a contract request to WP00 rather than resolving it inside WP04.

## WP06:1

### [MAJOR] `src/jevbot/fills.py:148`

`net_of` (and therefore `BandFillModel.price`) ignores `OrderLeg.ratio`, while `p_bp` (line 340), `check`'s contract count (line 305), `fees_micro` (line 434) and `liquidation` (line 388) all honour it. A ratio > 1 leg that the module explicitly accepts is counted ONCE in the signed net but twice everywhere else, so `Book.apply`'s `cash[b] -= net[b] * 100 * qty` (10.8) books half the cash while the mark and the fee count the full size.

**Evidence:** DESIGN 10.3: `net[band] = sum(buy prices) - sum(sell prices)`; for a 1x-2x-1x structure the per-unit net must weight the leg by its ratio, which is exactly what `liquidation` does. Probe p7 on a single BUY leg quoted 100 x 110: ratio=1 -> net.worst=110, fees_micro=40_300, liquidation=(-100,-105,False); ratio=2 -> net.worst=110 (unchanged!), fees_micro=80_600, liquidation=(-200,-210,False). `_legs` (fills.py:246) only rejects `ratio < 1`, although its own message says "v1 ratios are 1", and `tests/unit/test_fills.py:265 test_a_ratio_leg_counts_towards_the_leg_class` deliberately prices a ratio=2 leg and asserts only that the leg class moved from 7500 to 6600 bp - enshrining the half-supported path instead of catching the wrong net. `LegFill` (2.4) carries no ratio field either, so a ledger replay cannot recover it.

**Fix:** 2.4 says `Leg.ratio` / `OrderLeg.ratio` is "always 1 in v1" and section 8 builds no ratio spreads: make `_legs` reject `ratio != 1` outright (fills.py:246) and drop the `sum(leg.ratio ...)` in `price` (line 340) in favour of `len(checked)` per 10.3's `n_legs`; rewrite tests/unit/test_fills.py:265 to assert the refusal. If ratios are ever wanted, `net_of` must multiply each band price by the ratio and `LegFill` must carry it.

### [MAJOR] `src/jevbot/ledger.py:513`

`set_meta` raises `InvariantError` for `last_verified_seq` on the FIRST write, and the high-water mark is exposed only as a `SqliteLedger`-specific property (`last_verified_seq`, line 435) that the `Ledger` Protocol (3.4) does not declare and WP00's `MemoryLedger` neither maintains nor exposes. DESIGN 9.6 R5 ("ledger.verify(): full at startup, incremental since the last verified seq each cycle") therefore has no portable implementation: consumers typed against `Ledger` cannot read it at all, and swapping in the MemoryLedger double breaks at runtime. The filed contract request only asks for a comment in protocols.py, which does not fix either side.

**Evidence:** Probe p4 output: SqliteLedger -> `last_verified_seq -> 3`, `get_meta('last_verified_seq') -> 3`, `set_meta('last_verified_seq','3') -> InvariantError meta key 'last_verified_seq' is maintained by verify(), not by set_meta()`. MemoryLedger -> `last_verified_seq -> AttributeError: 'MemoryLedger' object has no attribute 'last_verified_seq'`, `get_meta('last_verified_seq') -> None` after a full verify(), `set_meta(...,'3') -> ok`, `set_meta(...,'5') -> InvariantError ... write-once`. 13.4 lists `last_verified_seq` among the `meta` rows; 3.4 says every implementation must be interchangeable with the double (INV-19/INV-24), and tests/unit/test_ledger.py:104 asserts exactly that interchangeability for the chain.

**Fix:** Turn the contract request into a behaviour change rather than a comment: put the high-water mark behind the Protocol - either have `verify()` return the seq it reached, or document `get_meta(ledger.LAST_VERIFIED_SEQ)` as the single read that BOTH implementations maintain inside `verify()` - and ask WP00 to mirror it in `MemoryLedger.verify()`. Alternatively drop the `set_meta` refusal and special-case that one key as last-write-wins in both implementations, so 13.4's row can advance without violating 3.4's write-once rule for every other key.

### [MAJOR] `tests/unit/test_ledger.py:237`

The section-16 acceptance item "chain verify + tamper detection" is only half tested: the prev-hash linkage branch of `verify()` (`src/jevbot/ledger.py:413-414`) is never executed by any test, so the one attack the hash CHAIN exists to stop - re-chaining a forged entry by recomputing its own hash from the stored `prev_hash` - has no regression test.

**Evidence:** `--cov-branch` reports `src/jevbot/ledger.py ... Missing 213-214, 414, 535`; line 414 is `raise LedgerCorrupt(f"ledger seq {seq}: prev_hash does not match the hash of seq {seq - 1}")`. All seven parametrised cases in `test_verify_detects_every_tamper` (line 249) are caught earlier, by the seq-gap check (line 412) or the hash recompute (line 421): rewriting only `payload` leaves that row's own `prev_hash` intact. Probe p3 (recompute seq 2's payload AND its hash with `canon.ledger_entry_hash(stored prev_hash, 2, 'mark', ...)` through a raw connection with the triggers dropped) shows the behaviour is correct - `LedgerCorrupt: ledger seq 3: prev_hash does not match the hash of seq 2` - but nothing in the suite would notice if that check were deleted.

**Fix:** Add a parametrised tamper case to `test_verify_detects_every_tamper` that re-chains seq 2 (write a new canonical payload plus the hash recomputed from seq 2's stored `prev_hash`) and asserts `LedgerCorrupt` matching "seq 3", plus a case that tampers with the `prev_hash` column directly. Both should also assert that `entries()` still streams (the chain check is only in `verify`).

### [MINOR] `src/jevbot/ledger.py:404`

`verify()` is specified in 3.4 as "recompute the chain; raises LedgerCorrupt", but this implementation writes: `_advance_verified` (line 424) does `BEGIN IMMEDIATE` + an UPSERT into `meta`. `__init__` also writes on every open (`executescript(SCHEMA_SQL)` and `PRAGMA user_version=1`). A store that cannot be written - held by another writer, or read-only - therefore fails a read-only operation with a raw `sqlite3.OperationalError`, which maps to generic exit 1 instead of 9 (2.9 `LedgerCorrupt`).

**Evidence:** Probe p6: with a `per_session` writer holding an open `BEGIN IMMEDIATE`, a second `SqliteLedger` on the same path opens fine but `verify()` raises `sqlite3.OperationalError: database is locked` - not `LedgerCorrupt`, and not a `jevbot.errors` type at all. On a committed store the same call moves `meta` from `[]` to `[('last_verified_seq','3')]`, confirming verify() mutates. WP11's baseline 6 reads "the reference RUN STORE's states via `Ledger.get_states`" (section 16) and the report commands read the views, so read-only opens of a finished run store are a real path.

**Fix:** Make the marker write best-effort and non-fatal (wrap `_advance_verified` so a write failure never fails a successful verification), and wrap `sqlite3.Error` escaping `append` / `commit` / `verify` / `__init__` in `LedgerCorrupt` so the documented exception type and exit code 9 hold.

### [MINOR] `src/jevbot/fills.py:399`

10.5's fallback bound for an unusable SHORT leg is `max(intrinsic, last ask)`. With `last is None` the code substitutes the unusable row's own `max(bid, ask, 0)` (0 when the contract is missing). For the commonest unusable-short case - `ask <= 0` with a positive bid - that substitute is the BID, which is below the true ask, so the mark can understate what closing costs in exactly the ANOMALY case 10.5 wants bounded conservatively.

**Evidence:** 10.5: "Equity uses the conservative mark, so the loss triggers are biased early, not late." fills.py:400 `seen = 0 if q is None else max(q.bid, q.ask, 0)` then `bound = max(intrinsic_cents(..., up=True), seen)`. tests/unit/test_fills.py:569-571 only pins the crossed case (bid 700 > ask 600, where `seen` happens to exceed the ask); no test covers a short leg with a positive bid and `ask == 0`, where the bound lands on the bid. The `FillModel.liquidation` Protocol only passes structure-level `(liq_value, mid_value)`, so the per-leg last ask 10.5 names is genuinely unavailable - this is a contract gap, not a local coding choice.

**Fix:** Raise it as a contract request against 3.4 rather than silently substituting: give `liquidation` the per-leg last marks the same way `price` already takes `last_marks` (or carry the last ask on `Position`), so `max(intrinsic, last ask)` can be evaluated as 10.5 states. Until then, document the substitution in the docstring as potentially optimistic and add the `ask == 0, bid > 0` short-leg case to `test_the_fallback_bound_prices_unusable_legs_when_no_previous_mark_exists`.

### [MINOR] `src/jevbot/fills.py:306`

A BUY leg quoted with a positive bid and no ask is reported as `crossed_or_locked` only, never `no_quote`, because the `elif` at line 308 suppresses the second code. 10.4 defines `no_quote` as "(a) any BUY leg without an ask (`ask <= 0`)" with no exception for a positive bid, so the fill-reject funnel attributes a genuinely absent ask to the wrong reason.

**Evidence:** Probe p7: `check([buy_to_open], 1, quote(bid=100, ask=0)) -> ('crossed_or_locked',)`. Both 10.4 rules match this quote literally (`ask <= bid` with `bid > 0` AND a BUY leg with `ask <= 0`), and `check` returns a tuple of codes precisely so several can be reported. No test covers `bid > 0, ask == 0` for a BUY leg - `test_each_rejection_code_of_10_4` uses `(0, 0)` for no_quote and `(110, 100)` / `(100, 100)` for crossed.

**Fix:** Evaluate the two rules independently (drop the `elif`, keeping the existing `FILL_REJECTS`-order sort) so such a row yields `(no_quote, crossed_or_locked)`, and add the `(100, 0)` BUY case to `test_each_rejection_code_of_10_4`. Rejection stays band-independent either way.

### [MINOR] `src/jevbot/ledger.py:183`

`store.parent.mkdir(parents=True, exist_ok=True, mode=0o700)` applies 0700 only to the LEAF directory; `pathlib` creates the intermediate parents with the default mode. The deviation note claims the store "creates a missing run directory (mode 0700, matching D1's $JEVBOT_DATA)", which is not what happens when more than one level is missing.

**Evidence:** Probe p5 on `<tmp>/runs/r1/run.sqlite` with none of the parents present: `[('<tmp>', '0o755'), ('<tmp>/runs', '0o755'), ('<tmp>/runs/r1', '0o700')]`. DESIGN 13.1 / D1: the data directory tree is "outside git, created mode 700".

**Fix:** Walk the missing ancestors and create each with `mode=0o700` (or `os.chmod` them after `mkdir`), or narrow the claim in the module docstring / deviation note to "the leaf run directory only" and leave the tree's permissions to whoever creates `$JEVBOT_DATA`.

## WP07a:1

### [BLOCKER] `src/jevbot/eval/metrics.py:204`

The equity series used for every headline P&L number starts at the FIRST session's SESSION_END equity, so the first session's P&L is silently dropped from total_pnl, total_return, CAGR, max_drawdown, worst_day, CVaR and the returns series. 2.4 / 10.8 define `day_start_equity` = initial cash on the first session, and RUN_START carries `initial_cash`, so the day-1 return is fully defined. `load.daily_frame` (load.py:443) does not expose `initial_cash` and load.py has no RUN_START loader, so no downstream caller can repair it.

**Evidence:** 12.2 "From the daily equity series per band: total return, CAGR, ... max drawdown"; 2.4 PortfolioState.day_start_equity "headline-band equity of the PREVIOUS session's SESSION_END entry (initial cash on the first session)". Probe /tmp/wp07a/probe2.py on a 20-session fixture run: initial_cash 10,000,000; first SESSION_END equity 9,999,488 (day 1 opens a position and pays a fee). `equity_metrics(...).total_pnl` reports 1072 cents; the run's true P&L (end_equity - initial_cash) is 560 cents — a 91% overstatement. `daily_returns` yields 19 returns for 20 sessions. `rejected_mid_only` (metrics.py:243) is decided on these same wrong `total_pnl` values, so a configuration whose losses are concentrated on day 1 can be reported as profitable.

**Fix:** Have `load.daily_frame` carry the RUN_START `initial_cash` (a new loader or an extra column) and have `_equity_series` / `equity_metrics` prepend it as the t0 point of each band's series, so the series is [initial_cash, SESSION_END_1 ... SESSION_END_n] and n_sessions returns are produced for n sessions. Add a fixture-run test asserting `total_pnl == end_equity - RunFixture.initial_cash`.

### [BLOCKER] `src/jevbot/eval/load.py:431`

`reference_history()` returns ONE ROW PER FORECAST ENTRY, not one per event. Every evaluated event is written twice — once for the `with_text` and once for the `without_text` request (0.1 item 5, V9, 10.1 step 6a: "append FORECAST entries for BOTH request kinds that exist this session") — and both rows carry the identical market pair (question_id, horizon, session, p_implied, y, resolved_on). The training history therefore double-counts every event.

**Evidence:** 12.3: `history` columns are `question_id, horizon, session, p_implied, y, resolved_on` — there is deliberately no `with_text` column, because a training pair is a property of the EVENT, not of a decider's forecast. Probe /tmp/wp07a/probe1.py on a 20-session fixture run: `reference_history` returns 726 rows for 363 distinct event_keys; `groupby([question_id, session, p_implied]).size().max() == 2`. Consequences: (a) `reference_min_events = 250` (12.1: "per primary question ... counted over resolved training pairs") is reached at 125 real events, defeating the exact guard 12.1 spells out — "below `reference_min_events` a reference is *unavailable* (NaN)" and "a reference trained on ... a few dozen events would be beaten by any constant forecaster"; (b) sessions with no news archive produce only the `entry` forecast (2414: "no `entry_text` request exists at all"), so events get weight 1 or 2 depending on news coverage — the expanding base rate and the PAV fit become news-coverage-weighted rather than plain frequencies. The reference run is a MockJev backtest and V12 resolves `news.enabled = auto` to ON for MockJev, so the real reference store will contain both sets.

**Fix:** Deduplicate to one training pair per `event_key` before projecting the six columns, e.g. `usable = usable.drop_duplicates(subset="event_key")` (or select a single forecast set) in `reference_history`. Add a test that builds a reference store holding both `with_text` values and asserts `len(history) == n_events` and that each (question_id, session) appears exactly once per underlying — `tests/unit/test_eval_load.py:159` cannot catch this today because `make_reference_history_store` writes only `with_text=False` rows (make_run_fixture.py:1194).

### [MAJOR] `src/jevbot/eval/metrics.py:429`

`risk_normalised_pnl` and `ExposureMetrics.max_loss_days_at_risk` sum `open_max_loss` over MARK ROWS, not over SESSIONS. 12.2 states the denominator literally as "sum over sessions of open max_loss". A paper run writes two MARK entries per session (11.3: `run_cycle(..., Phase.DECIDE)` at close-25 marks, and `run_cycle(..., Phase.CLOSE_OUT)` at close+10 marks again); a `--recorded-mode dec_exec` backtest likewise. The denominator then doubles and the headline risk-normalised P&L halves.

**Evidence:** 12.2: "**Risk-normalised P&L: `pnl / sum over sessions of open max_loss`** (\"per unit of max-loss-days at risk\") for every run and baseline". Probe /tmp/wp07a/probe2.py: on the 20-session fixture (1 MARK/session) `max_loss_days_at_risk = 2,560,000` and `risk_normalised_pnl = 0.0013671875`; with the same marks frame duplicated to simulate 2 MARK entries per session the values become 5,120,000 and 0.00068359375. Since 12.2 says this metric is computed "for every run and baseline", a paper/recorded run and a mirror backtest baseline would be compared on denominators that differ by a factor of 2.

**Fix:** Aggregate `open_max_loss` per session first (e.g. `marks.groupby("session")["open_max_loss"].max()` or the designated end-of-session mark) and sum that, in both `exposure_metrics` (line 419) and `risk_normalised_pnl`. Add a regression test with two MARK rows per session asserting the result equals the one-mark-per-session value.

### [MAJOR] `src/jevbot/eval/load.py:279`

`forecasts_frame` silently drops the FORECAST payload's snapshot `key` (its `slot`) and `p_implied_spread_ppm`, and keeps only `kind`/`horizon_sessions`/`resolve_on` of the spec. Because `eval/load.py` is declared "the ONLY door between a run.sqlite and the rest of eval/", the pre-registered sensitivity analysis that needs the slot cannot be computed downstream at all. The implementer's own deviation note claims the opposite: "It extracts every field those views name plus the ones they omit (question_hash, the snapshot key, p_implied_spread_ppm, the full spec)".

**Evidence:** 12.1 `sensitivity` (a pre-registered, committed list): "reference from the exec snapshot and the dec/exec average (indicative-feed noise)" — this requires the per-forecast `key.slot` (dec | exec | eod). 2.7 `Forecast.p_implied_spread_ppm` is "the call-spread cross-check when an expiry lands on the resolve session". Probe output: `sorted(forecasts_frame(...).columns)` = [as_of, decision_id, event_key, fidelity, forecast_id, horizon, implied_method, implied_quality, iv_history, missing, missing_reason, namespace, p, p_abstain, p_implied, p_implied_ppm, p_ppm, prereg, price_measure, question_hash, question_id, resolve_on, run_id, session, spec_kind, tier, underlying, with_text] — no `slot`, no `p_implied_spread_ppm`, no `ref`/`lo`/`hi`/`iv_var_ppm`. `grep -rn 'p_implied_spread' src/jevbot/eval/` returns nothing. The fixture does write both fields (make_run_fixture.py:702, 709), so the data is present and merely discarded.

**Fix:** Add `slot` (from `payload["key"]["slot"]`), `p_implied_spread_ppm` / `p_implied_spread`, and the remaining spec fields (`ref`, `lo`, `hi`, `iv_var_ppm`) to `forecasts_frame`, and add `slot` to `CALIBRATION_COLUMNS`. Add a test asserting the frame carries the slot and the spread cross-check for the fixture run.

### [MAJOR] `src/jevbot/eval/registry.py:295`

The `purpose = "final"` / paper-boot scan-facts gate reads ONLY `unsizeable_rate_by_tier[str(500000)]` and never looks at `exceeds_risk_budget_rate`, although `ScanRow` parses it (registry.py:235). Section 8 and the config comment define the gated quantity as `exceeds_risk_budget` OR `size_zero`. A (underlying, kind) whose candidate generator returns `CandidateReject("exceeds_risk_budget")` for (almost) every session never reaches sizing at all, so its `size_zero`/unsizeable rate at tier 500000 can legitimately be ~0 — and the gate passes a structure that can never actually trade.

**Evidence:** config/default.toml documentation in DESIGN.md:1734-1735: "`max_unsizeable_rate = 0.25` — `data scan-candidates`: an enabled (underlying, kind) whose candidates are **exceeds_risk_budget / size_zero** at the LOWEST non-zero tier more often than this in the latest scanned year blocks purpose \"final\" and the paper boot (section 8)". Section 8 (line 3319): "`scan()` reports, per (underlying, kind, year), the `exceeds_risk_budget` rate **and** the `size_zero` rate at each tier". `scan_facts_problems` (registry.py:284-302) never references `row.exceeds_risk_budget_rate`, and `tests/unit/test_eval_registry.py:116` `scan_row()` always sets it to a harmless 0.02, so no test exercises it.

**Fix:** Flag a pair when `max(row.exceeds_risk_budget_rate, row.unsizeable_rate_by_tier[str(tier_ppm)]) > max_unsizeable_rate` (or when the writer's `unsizeable_rate_by_tier` is documented in the contract request to already fold the budget rejects in — then delete the unused field and say so). Add a test with `exceeds_risk_budget_rate = 0.9` and `unsizeable = 0.0` asserting `final` is refused.

### [MAJOR] `src/jevbot/eval/metrics.py:457`

`cost_metrics.fees_cents` is computed as `round(sum(fills.fees_micro) / 10_000)`, i.e. from the per-fill ACCRUAL with banker's rounding over the whole run. 10.7 says the fee actually charged to the three cash balances is per session `fee_cents = cdiv(accrued, 10_000)` (ceiling), written as a FEE ledger entry. The reported fee therefore never reconciles with the equity series, and is systematically low by up to one cent per session. `load.py` exposes no loader for `LedgerKind.FEE` at all, so no caller can obtain the true figure.

**Evidence:** 10.7: "at session end `fee_cents = cdiv(accrued, 10_000)` is charged to all three cash balances in one FEE entry and the accumulator resets (Alpaca charges at end of day, rounded up to the cent)". 12.2 "Costs: fees". Probe /tmp/wp07a/probe2.py on the 20-session fixture: `cost_metrics(...).fees_cents = 403` while the FEE entries actually charged to cash sum to 240 over 20 entries; `[n for n in dir(load) if 'fee' in n.lower()]` is `[]` (only `fills_frame`, no FEE frame). The existing test (tests/unit/test_eval_metrics.py:222) checks only a hand-built two-fill frame and never compares against the FEE entries.

**Fix:** Add a `fees_frame(store)` loader over `LedgerKind.FEE` (`fee_cents`, `accrued_micro_before`) and have `cost_metrics` take it and report `sum(fee_cents)`; if the accrual is still wanted, report it as a separate `fees_accrued_micro`. Add a test asserting the reported fee equals the sum of the fixture's FEE entries.

### [MAJOR] `src/jevbot/eval/tiers.py:70`

`decided_live()` adds a `0.0 <= delay` lower bound that 12.1 does not have. In paper, `as_of` is the round-trip-compensated BROKER clock (INV-12) while `ledgered_wall` is the local wall clock; INV-12 explicitly tolerates up to 5 s of skew before opening orders are blocked. A local clock a couple of seconds behind the broker clock makes every promptly-appended entry score `decided_live = False`, silently demoting an entire forward paper run from Tier A to Tier B.

**Evidence:** 12.1 line 4214, the whole definition: "`decided_live` = the entry was appended by the paper runner with `ledgered_wall - as_of <= evidence.tier_a_max_log_delay_s` (600 s)" — one-sided. INV-12 (line 104): "Paper trading time comes from the broker clock (round-trip compensated). Skew > 5 s blocks **opening** orders". Probe /tmp/wp07a/probe3.py: `decided_live(as_of=19:35:00Z, ledgered_wall=19:34:58Z, max_log_delay_s=600)` returns False. This is not in the implementer's declared deviation list, and `tests/unit/test_eval_tiers.py:64` locks the behaviour in.

**Fix:** Either drop the lower bound (keep only `delay <= max_log_delay_s`) or allow a tolerance at least as large as the permitted clock skew (e.g. `-5.0 <= delay`), and state the choice as a deviation if the strict form is kept. Update the test at test_eval_tiers.py:64 accordingly.

### [MAJOR] `src/jevbot/eval/registry.py:513`

`register_trial` never validates `meta.purpose` against the closed set of 2.8. `RunMeta.purpose` is a plain `str` (types.py:1188), so a typo (`"Tune"`, `"fina1"`) silently skips the holdout guard, the Step 0 gate AND the scan-facts gate and registers the trial. All three refusals fail OPEN on a typo; the same typo also drops the run out of `SELECTION_PURPOSES`, so it never counts toward `N`.

**Evidence:** 2.8: `purpose: str  # "validate" | "tune" | "diagnostic" | "final" | "reference" | "paper" | "shadow"`. 12.1: "the trial registry refuses `purpose = \"tune\"` for any run whose window includes a session >= model_release_date"; "it also refuses `purpose = \"tune\"` / `\"final\"` for a Jev decider whose ... probe records [are missing]; and `purpose = \"final\"` without sizeable scan-candidates facts". Probe /tmp/wp07a/probe4.py: `register_trial(run_meta(purpose="Tune", decider="live_jev", end=2027-01-04))` succeeds with no `HoldoutViolation` and no Step 0 refusal; `purpose="fina1"` likewise skips the scan-facts gate.

**Fix:** Add `PURPOSES: Final[frozenset[str]] = frozenset({"validate","tune","diagnostic","final","reference","paper","shadow"})` and raise `ConfigError`/`EvalError` in `register_trial` when `meta.purpose not in PURPOSES`. Add a parametrised test over the typos.

### [MINOR] `src/jevbot/eval/metrics.py:631`

`DecisionProcessMetrics.n_entries_emitted` counts EVERY approved RISK_VERDICT, including the verdicts of CLOSE / KILL orders, and `risk_reject_rates` / `candidate_reject_rates` use the same all-verdicts denominator (`n_verdicts`, line 644). Only entry verdicts belong in an "entries emitted" count.

**Evidence:** 12.2 "Decision process: request counts, abstention funnel by first failing step ... risk-reject rate by check". Probe /tmp/wp07a/probe2.py: `n_entries_emitted = 17` on the fixture; adding an equal number of approved close-order verdicts raises it to 51. The fixture never writes a RISK_VERDICT for a close (`_close_due_positions`, make_run_fixture.py:892-974 writes ORDER_INTENT + FILL only), so `tests/unit/test_eval_metrics.py:366` cannot catch it, while 9.1 requires every close to pass `approve()` in a real run.

**Fix:** Restrict the verdict loop to verdicts whose `decision_id` belongs to an entry DECISION (the `_pair_index`/decisions frame already supplies the mapping), or count only verdicts joined to `kind == "entry"` decisions. Extend the fixture to write close verdicts and assert the count.

### [MINOR] `src/jevbot/eval/load.py:797`

`trades_frame` returns `pd.DataFrame.from_records([])` — a frame with NO columns — whenever FILL entries exist but no position group has an OPEN fill (a store resumed after its opening session, or one whose only fills are closes). The documented column contract is only honoured on the `fills.empty` path (line 741).

**Evidence:** Probe /tmp/wp07a/probe3.py: feeding `trades_frame` a fills frame containing only `purpose == "close"` rows yields `columns == []`. Downstream, `metrics.trade_metrics` guards on `column not in trades.columns` and returns an all-NaN result, but `max_adverse_excursion` (metrics.py:344) does `trade["position_id"]` and `abstention_funnel`-style callers indexing by name would raise KeyError.

**Fix:** Hoist the documented column tuple into a module constant and return `_empty(TRADE_COLUMNS)` whenever `records` is empty, exactly as the `fills.empty` branch does. Add the case to `test_empty_frames_have_their_columns`.

### [MINOR] `src/jevbot/eval/prereg.py:592`

`forecast_is_prereg` compares two RFC 3339 texts lexicographically, and its helper claims "RFC 3339 UTC text compares lexicographically in the same order as the instants it spells" (line 596). That is false for `canon.render_as_of`, which emits sub-second digits ONLY when non-zero (2.7), so `"...T12:00:00Z"` sorts AFTER `"...T12:00:00.250000Z"` ('Z' = 0x5A > '.' = 0x2E).

**Evidence:** Probed directly: `canon.render_as_of(12:00:00)` = '2026-09-17T12:00:00Z', `canon.render_as_of(12:00:00.250000)` = '2026-09-17T12:00:00.250000Z'; `a < b` is False although a precedes b. `pre.forecast_is_prereg(registered_at=a, first_tier_a_at=b, status="registered")` returns False, i.e. a forecast made 250 ms AFTER the registration is not tagged `prereg` (12.1: "forecasts are tagged `prereg = true` only if the first Tier A forecast postdates the registration"). It fails closed (under-tags), but only within the same whole second. `tests/unit/test_eval_prereg.py:460` uses two whole-second timestamps and so never hits it.

**Fix:** Parse both values to `datetime` (`datetime.fromisoformat(text.replace("Z", "+00:00"))`, as `load._parse_as_of` already does) and compare instants; delete the false docstring claim. Add a sub-second case to the test.

### [MINOR] `src/jevbot/eval/tiers.py:177`

`assert_poolable` fails OPEN on a missing key: a caller that forgets to pass the `tier` (or `namespace`, or `price_measure`, or `with_text`) column silently passes the INV-22 guard. The docstring even asserts the opposite of what the code does ("a caller that cannot know the price measure of its rows must not be able to *silently* pass the guard").

**Evidence:** INV-22: "tiers and namespaces are never pooled", enforced by `eval/tiers.py`, `eval/report.py`. 12.1: "`eval/report.py` raises `TierViolation` when asked to pool tiers, namespaces, price measures or the with-text / without-text forecast sets in one number". Probe /tmp/wp07a/probe3.py: `assert_poolable({"namespace": ["a","a"]}, what="a pooled Brier over two tiers")` passes without complaint because `tier` was never supplied. `tests/unit/test_eval_tiers.py:161` locks this fail-open behaviour in as intended.

**Fix:** Require every key in `keys` to be present and raise `TierViolation` (or a distinct error) naming the missing columns, with an explicit `waive=(...)` argument for the two exemptions 12.1 names (the reference history and `eval/agreement.py`). Since `load.calibration_frame` guarantees all four columns (CALIBRATION_COLUMNS), no legitimate report caller loses anything.

### [MINOR] `src/jevbot/eval/prereg.py:508`

`register()` trusts `power.chosen_interval` as written in `prereg/power.v1.json` and never re-derives it from that file's own `size` table, and never checks that the power file is committed to git (the prereg TOML is checked, at line 478). A hand-edited or buggy power file whose `chosen_interval` contradicts its `holds` flags is accepted, which defeats the refusal 12.1 words as "`interval` above MUST equal the first candidate that holds its size".

**Evidence:** 12.3: "`chosen_interval` = the first candidate that holds its size under **all** nulls at **both** looks"; `SizeEntry.holds` (prereg.py:404) already stores the per-(interval, null, look) verdict, and `PowerFile.size` carries every entry, so the value is fully re-derivable. 12.1 `power` also requires the file to be "committed next to this file". No test in tests/unit/test_eval_prereg.py constructs a power file whose `chosen_interval` disagrees with its `size` rows.

**Fix:** Recompute `first(candidate for candidate in spec.interval_candidates if all(entry.holds for entry in power.size if entry.interval == candidate))` and refuse when it differs from `power.chosen_interval`; also re-check each `holds` against `rejection_rate <= alpha + 2*sqrt(alpha*(1-alpha)/reps)` (the `size_rule`), and run `git_path_committed` on the power file. Add the two corresponding refusal tests.

## WP07b:1

### [BLOCKER] `src/jevbot/eval/bootstrap.py:188`

The percentile branch of `bootstrap_ci` (and therefore `lower_bound(interval="percentile")`, the interval the committed prereg registers) silently absorbs a non-finite value in the pre-registered `d_t` series: every bootstrap replicate that happens to draw the NaN session is thrown away (`usable = draws[np.isfinite(draws)]`) and the one-sided bound is taken over the surviving, selection-biased replicates. No error, no flag, no count. The very same input is refused by the other two methods (line 200-201 studentised, line 254-255 null_calibrated), so the three pre-registered interval methods disagree about whether the data are usable. `paired_bootstrap_ci` (line 302) and `cluster_bootstrap_ci` (line 343) do the same. Non-finite d_t is a documented, reachable output of this unit's own `calibration.loss_differential` ("a session on which a question has no usable row yields NaN for that session", calibration.py:933), and `eligible_sessions` marks such a session eligible because eligibility is defined only on reference availability (12.1 `eligible_session`), so the NaN reaches `lower_bound` on the ordinary verdict path.

**Evidence:** DESIGN 12.1 `success` = "at a look: the one-sided (1 - alpha) lower bound of mean d_t is > 0 against BOTH references"; `missing` = "void outcomes are excluded and listed" (excluded, not silently averaged away inside the resampler). /tmp/wp07b/probe_misc.py builds 6 sessions x 2 primary questions x 3 underlyings with q2 all-void on session 4: `eligible_sessions` returns all 6 sessions, `loss_differential` returns `[-0.0023, 0.0177, -0.0023, nan, -0.0023, 0.0177]`. /tmp/wp07b/probe_nan_d.py: with one NaN in a 200-session d_t only 725 of 2000 replicates are finite, yet `lower_bound(..., interval="percentile")` returns a number, while `interval="studentised"` and `interval="null_calibrated"` both raise EvalError on the identical input. /tmp/wp07b/probe_final.py, 60 seeds, N(0, 0.01) d_t with the most negative session replaced by NaN: the bound from the NaN path differs from the correct "drop that session" bound by up to 4.9e-4 and the sign of the bound - i.e. the pre-registered go/no-go verdict - flips in 6 of 60 trials (10%).

**Fix:** Validate the series up front instead of filtering replicates. In `lower_bound`, `bootstrap_ci`, `paired_bootstrap_ci` and `cluster_bootstrap_ci`, after `np.asarray(...).ravel()` raise `EvalError` naming the non-finite positions (e.g. "d_t is non-finite at sessions [...]: exclude and list them (12.1) before taking the bound"), so the caller drops those sessions explicitly and the report lists them, exactly as 12.1 requires. Keep the replicate-level `isfinite` filter only for a genuinely degenerate user-supplied `stat`, and then record and expose the number of dropped replicates rather than discarding them silently.

### [MAJOR] `tests/unit/test_eval_bootstrap.py:270`

The size-check acceptance test required by 15.1 / 12.3 is weakened to the point where it certifies an over-sized method as "nominal". It asserts `block_rate <= tolerance + 0.02` where `tolerance = alpha + 3*sqrt(alpha(1-alpha)/n_sims)`, i.e. it accepts an empirical rejection rate up to 0.1114 against a nominal alpha of 0.05 - more than twice alpha, and well above the prereg's own `size_rule` budget. The method it validates (`percentile`, which is what `prereg/prereg.v1.toml:14` actually registers) genuinely does not hold its size on the pre-registered d_t structure, so the test's docstring claim ("nominal size for the chosen interval") is not demonstrated by anything it asserts.

**Evidence:** prereg `size_rule` = "empirical rejection rate under every null forecaster <= alpha + 2*sqrt(alpha*(1-alpha)/eval.null_sim_reps), at each look" -> with `eval.null_sim_reps = 2000` the budget is 0.0597. /tmp/wp07b/probe_size.py, 1000 simulations of the test's own `_overlapping_correlated_d` (n=320, horizon 5, 3 underlyings, rho=0.85, true mean 0): empirical size of `lower_bound(alpha=0.05, block=12, interval="percentile")` = 0.0990 +/- 0.0094 (MC se) - roughly 2x nominal and a clear failure of the prereg rule. /tmp/wp07b/probe_size2.py reproduces the test's own run at its seed: block_rate 0.0760, assert bound 0.1114, so the test passes on a 250-sim draw of a truly 9.9%-sized method. 12.3 anticipates exactly this ("A percentile bound from about 12 effective blocks of a heavy-tailed, cross-correlated, overlapping d_t is known to under-cover ... so this is checked, not assumed"). Corroboration from the sibling suite: in test_eval_calibration.py:841 the zero-skill constant forecaster's one-sided 99% bound against `base_rate_expanding` is +5.8e-05 - a false rejection at alpha = 0.01 - which the test excuses in a comment ("may narrowly win, the sampling noise of the estimate being a second-order Brier cost") instead of flagging; measured with /tmp/wp07b/probe_reg.py.

**Fix:** Assert against the prereg's own `size_rule` formula rather than an ad-hoc `+ 0.02` slack: with `n_sims` simulations require `rate <= alpha + 2*sqrt(alpha*(1-alpha)/n_sims)` and raise `n_sims` until that budget is tight enough to be meaningful (>= 1000 for alpha=0.05). Because `percentile` cannot pass that budget on this d_t structure, the test must do what the size check exists to do - exercise the candidates in the order of `interval_candidates` and assert that the FIRST one holding its size is the one the test then calls the chosen interval (studentised measured 0.0733 over 150 sims, MC se 0.021, so it needs its own properly sized run) - and keep the i.i.d. `block = 1` arm as the over-sized comparator it already is.

### [MAJOR] `src/jevbot/eval/calibration.py:582`

`_dates` mis-detects `pandas.NaT` as a real date: `NaTType` subclasses `datetime.datetime`, so `isinstance(value, date)` is True and `isinstance(value, pd.Timestamp)` is False, and NaT is appended straight into the date list. The subsequent `sorted(set(values))` then dies with a raw `TypeError: Cannot compare NaT with datetime.date object` - not an `EvalError`, so the caller gets no usable diagnostic. This is hit by `session_grid` (line 599) and therefore by `base_rate_expanding`, `recalibrate_walkforward` and `build_references` on their default `sessions=None` path, and by `unique_sessions` / `eligible_sessions` (lines 858, 884) on a NaT `session`. The module's own sibling helper `_dates_or_none` (line 670) already tolerates NaT, so the two date readers of one module disagree.

**Evidence:** 12.3 requires the report to state "N resolved / open / void / missing", so still-open forecasts are part of the calibration frame; the unit's own contract request to WP07a spells the column as "`resolved_on` (date or NaT when still open)", and test_eval_calibration.py:534 (`test_timestamp_typed_session_columns_are_accepted`) establishes that pandas datetime64 columns are a supported input - but that test has no null rows, and `test_void_and_open_outcomes_are_never_training_pairs` (line 529) sets `resolved_on = None` on an OBJECT column, which `_dates` skips, so neither test reaches the NaT branch. /tmp/wp07b/probe_nat.py: a 10-row frame with `session`/`resolved_on` run through `pd.to_datetime` and the last 2 rows still open gives `session_grid RAISED: TypeError Cannot compare NaT with datetime.date object`, and identically for `base_rate_expanding`, `recalibrate_walkforward` and `build_references`.

**Fix:** In `_dates`, test for missingness before the date branch, exactly as `_dates_or_none` does: `if value is None or (value is not None and pd.isna(value)): continue` placed ahead of the `isinstance(value, date)` check (`pd.isna` handles NaT, None and float NaN in one test). Add a regression test that feeds `base_rate_expanding` / `recalibrate_walkforward` / `eligible_sessions` a datetime64 frame containing open (NaT) rows.

### [MINOR] `src/jevbot/eval/dsr.py:249`

The guard flag for the "undefined / zero `Var(SR_n)`" branch is `dsr_var_sr_undefined`, but 12.6 names a single flag, `dsr_trials<2`, for both guard conditions. A report or consumer keyed on the spec's flag string (12.6 / 12.8 print the flags) will never see the variance-undefined case, even though the statistic is deflated by SR0 = 0 exactly as in the N < 2 case.

**Evidence:** DESIGN 12.6 Guards: "`N < 2` or an undefined / zero `Var(SR_n)` => `SR0 = 0`, DSR = PSR(0), flagged `dsr_trials<2` (never `Z(0) = -inf`)". dsr.py:246-249 splits this into two mutually exclusive branches and emits a flag name that appears nowhere in the spec; test_eval_dsr.py:253 then pins the non-spec name.

**Fix:** Emit `dsr_trials<2` in both branches (optionally adding `dsr_var_sr_undefined` alongside it as extra detail, never instead of it) and update test_eval_dsr.py:253 to assert the spec's flag is present.

### [MINOR] `src/jevbot/eval/agreement.py:173`

A DECISION whose `rules` block carries no `action` is silently turned into the empty string (`str(rules.get("action", ""))`), so two stores that both lack the field count as agreeing, inflating the 12.9(a) gate-decision agreement rate towards 1.0 on malformed data. The same function refuses a missing `rules` block (line 149) and a missing `underlying` (line 152) with an EvalError, so the handling of the payload contract is inconsistent - and `action` is the field statistic (a) is actually about.

**Evidence:** DESIGN 12.9(a): "gate-decision agreement rate - share of (underlying, session) pairs whose `EntryDecision.action` and `kind` are equal"; 2.6 makes `action` a non-optional field of `EntryDecision` ("enter" | "no_trade") and 2.11 fixes the DECISION payload as `rules: EntryDecision ... builtins`. test_eval_agreement.py covers the missing-`rules` and missing-`underlying` refusals (lines 451, 459) but nothing asserts a missing `action` is refused.

**Fix:** Treat a missing or non-string `action` like a missing `underlying`: `raise EvalError(f"{path}: entry DECISION at seq {row['seq']} names no action")`, and add the matching refusal test next to `test_a_decision_without_an_underlying_is_refused`.

## WP08:1

### [MAJOR] `src/jevbot/paper/broker.py:586`

With `orders.repost_same_id = true` the re-POST's `except` clause catches only `BrokerAmbiguous`. A `BrokerRejected` from the second POST - which is exactly the duplicate-client-order-id 422 that V4 / probe P-ALP-2 is about - escapes `submit()` unchanged, even though that rejection PROVES the first POST created the order. `reconcile.submit_approved` then ledgers `ORDER_STATUS{REJECTED}` (a TERMINAL status) for an order that is live and resting at the broker.

**Evidence:** DESIGN 11.5 step 4 / V4: 'Only when orders.repost_same_id = true ... is ONE re-POST of the same id allowed'; the broker's own `_REJECT_TAGS` already carries `('duplicate', 'duplicate_client_order_id')` but never acts on it. Probe /tmp/wp08probe/p4.py: FakeTradingClient with `fail_next('timeout_after_accept')` (order IS stored, our call raises) + lookups that fail after the pre-lookup, `repost_same_id=True` -> `submit()` raises `BrokerRejected(status=422, reject_code=42210000, tag='duplicate_client_order_id')` while `client.orders[cid]['status'] == 'accepted'` and `broker.posts == 2`. Blast radius, read from WP05: `reconcile.submit_approved` (src/jevbot/reconcile.py:170) maps BrokerRejected -> ORDER_STATUS{REJECTED}; `types.TERMINAL_STATUSES` contains REJECTED, so `portfolio.Book.open_orders` (src/jevbot/portfolio.py:263) drops the order and `reconcile.ingest` never polls it again - the one fill path goes blind. Reconcile R1 (src/jevbot/reconcile.py:422) does NOT cancel it either: `book.has_intent(...)` is true, so it is skipped as 'ours'. For a CLOSE/KILL intent the ladder re-approves at attempt+1 and submits a SECOND closing order -> the position can be closed twice (INV-09/INV-21 territory); for an OPEN the eventual fill trips RECONCILE_MISMATCH and flattens the book. No test exercises a rejected re-POST: test_paper_broker.py only covers the re-POST succeeding and a second `connection_reset`.

**Fix:** In the `repost_same_id` branch catch `BrokerError` (both classes), not just `BrokerAmbiguous`: on any failure of the re-POST run `self._lookup_after_ambiguity(cid)` once more and return the adopted state if found; if still not found, raise `BrokerAmbiguous` (never `BrokerRejected`) because the first POST's outcome was never resolved. Add a drill to tests/unit/test_paper_broker.py: first POST accepted-then-timed-out + deaf lookups + repost_same_id=True -> the live order is adopted, or at worst BrokerAmbiguous, and never BrokerRejected.

### [MAJOR] `src/jevbot/paper/lock.py:101`

`self.path.parent.mkdir(mode=_STATE_DIR_MODE, parents=True, exist_ok=True)` creates the MISSING PARENTS with the default permissions, not with `mode` (documented Python behaviour: 'they are created with the default permissions without taking mode into account'). On a fresh install boot step B1 therefore creates `$JEVBOT_DATA` itself as 0755 - world-readable - and boot step B2 then refuses to start.

**Evidence:** DESIGN 13.1: '$JEVBOT_DATA, outside git, created mode 700'; 11.2 orders B1 (lock.acquire) strictly before B2 (config + secrets, 'data dir mode 700'). Probe /tmp/wp08probe/p7.py on a fresh dir: `acquire(base)` -> `$JEVBOT_DATA` mode 0o755 (state/ is 0o700, the lock file 0o600), then `config.ensure_data_dir(base)` (src/jevbot/config.py:1271) raises ConfigError 'the data directory ... has mode 755: it must be 700' -> exit 2. So every first `jevbot paper run` on a new data dir fails and leaves a world-readable data directory behind that the operator must chmod by hand. Probe p6.py also shows a pre-existing `state/` at 0755 is left at 0755 (exist_ok=True never chmods). The existing test `test_acquire_creates_the_state_directory_private` hides this because `tmp_path` already exists, so only the `state/` level is ever asserted.

**Fix:** Create every missing level explicitly at 0o700 instead of relying on `parents=True` (walk `self.path.parent.parents` bottom-up creating each missing dir with `os.mkdir(d, 0o700)` then `os.chmod(d, 0o700)`), and chmod `state/` to 0o700 even when it already exists. Add a test that `acquire()` on a NON-existent data dir leaves both `$JEVBOT_DATA` and `$JEVBOT_DATA/state` with `st_mode & 0o077 == 0`, and one that `config.ensure_data_dir` accepts the directory acquire() just created.

### [MINOR] `src/jevbot/paper/broker.py:306`

The equity-flatten branch of `build_order_request` never validates the share count, while the option branch does (`if order.qty <= 0: raise InvariantError`). A zero or negative `intent.equity_qty` is serialised straight onto the wire.

**Evidence:** Probe /tmp/wp08probe/p1.py: equity intents with `equity_qty` 0 and -5 build without error and `to_request_fields()` yields `qty=0.0` and `qty=-5.0`. 2.4 says the three equity_* fields carry 'whole shares, <= abs(the broker's share position)'; 9.5 K4 uses this order as the kill switch's assigned-stock flatten, so a malformed payload turns into an Alpaca 422 and a flatten that silently does not happen (the kill sequence then sits in NOT_FLAT).

**Fix:** Mirror the option guard: `if int(intent.equity_qty) <= 0: raise InvariantError(...)` before building the MarketOrderRequest, and parametrise the existing `test_an_impossible_payload_is_a_bug_not_an_order` table with equity_qty 0 and -5.

### [MINOR] `src/jevbot/paper/broker.py:503`

`open_orders()` builds `GetOrdersRequest(status=OPEN, nested=True)` with no `limit`, so the Alpaca server default applies and the result is silently capped at 50 orders. The docstring promises 'Every working order the broker holds - ours and foreign'.

**Evidence:** Probe: `GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True).to_request_fields()` == {'status': ..., 'nested': True} - no limit. alpaca-py's own docstring for the field (site-packages/alpaca/trading/requests.py:203) states 'Defaults to 50 and max is 500'. Consumer: reconcile R1 (src/jevbot/reconcile.py:422) uses this list to find and cancel FOREIGN orders, so beyond 50 working orders a foreign order is silently never cancelled. (The kill switch's flat check at killswitch.py:230/245/257 is unaffected: truncation can only shrink a non-empty list, so 'empty' still means 'really empty'.)

**Fix:** Pass `limit=500` on the request and page with `after`/`until` (or assert the returned length is below the cap and raise `BrokerError` otherwise) so the sweep can never silently go partial.

### [MINOR] `src/jevbot/paper/broker.py:538`

`activities()` stops after `_ACTIVITY_MAX_PAGES` (50) pages and RETURNS the truncated list instead of raising, so a page-51 assignment row is silently invisible to reconcile R2.

**Evidence:** Probe /tmp/wp08probe/p3.py: 5100 OPASN rows at the broker -> `broker.activities(...)` returns 5000 rows after exactly 50 `get` calls, with no error. R2 (src/jevbot/reconcile.py:~437) trips KillTrigger.ASSIGNMENT off these rows; a dropped OPASN is a missed assignment (INV-11 / 9.5 K3 depend on it). The page loop also relies on the API's default ordering, which is `desc`, so it is the OLDEST rows that get dropped.

**Fix:** Raise `BrokerError` (fail closed) when the loop exhausts `_ACTIVITY_MAX_PAGES` with a full final page, and pass an explicit `direction=asc` so paging is deterministic. Add a test that 50 full pages followed by more data raises rather than truncating.

### [MINOR] `src/jevbot/paper/broker.py:597`

`_post` maps the vendor row to an `OrderState` OUTSIDE `_call`, so a `BrokerError` raised by `to_order_state` after a SUCCESSFUL POST escapes `submit()` as a bare `BrokerError` - a class the 3.4 contract ('raises BrokerRejected / BrokerAmbiguous') and `reconcile.submit_approved` do not handle.

**Evidence:** Probe /tmp/wp08probe/p3.py section E: a client whose POST succeeds but returns a row with a non-numeric `qty` makes `broker.submit()` raise plain `BrokerError` (isinstance BrokerRejected=False, BrokerAmbiguous=False) while the order exists at the broker. `reconcile.submit_approved` (src/jevbot/reconcile.py:168-177) catches only those two classes, so the exception propagates out of the order worker and no ORDER_STATUS entry is written after `SUBMITTING` - the 9.6 submit protocol's trail is left incomplete for a live order. (Low reachability with the real pinned `Order` model, whose fields are pydantic-validated.)

**Fix:** Wrap the mapping: `try: return to_order_state(raw, fallback_client_order_id=cid) except BrokerError as exc: raise BrokerAmbiguous(...) from exc` - the POST landed, so the outcome is unknown, not an error; the caller then ledgers UNKNOWN and the next `ingest`/reconcile R1 adopts the order by client id.

### [MINOR] `src/jevbot/paper/clock.py:211`

`sync()` measures the skew against local UTC at the MIDPOINT of the round trip (correct, D25) but anchors `now()` at `t1`, the boottime at the END of the round trip. `now()` therefore reads systematically rtt/2 BEHIND true broker time - i.e. the bot always thinks it is slightly earlier than it is, the unsafe direction next to the close-5-min submission cut-off.

**Evidence:** 11.4: 't0 = boottime(); r = get_clock(); t1 = boottime()' with 'local_utc_mid = local UTC at (t0 + t1) / 2 (round-trip compensated ...)' and 'now() = broker_ts + (boottime_now - boottime_at_sync)'. `broker_ts` is true at roughly (t0+t1)/2, but line 211 sets `_boottime_at_sync = t1`, so at boottime t the code returns broker_ts + (t - t1) instead of broker_ts + (t - t_mid): a constant lag of rtt/2 (50-150 ms on a typical Alpaca round trip). The unit's own test `test_the_skew_is_measured_at_the_midpoint_of_the_round_trip` uses rtt=400 ms and would show a 200 ms lag if `now()` were asserted after it.

**Fix:** Set `self._boottime_at_sync = t0 + rtt_s / 2` (the same midpoint the skew already uses) so `now()` and `skew_ms` share one anchor, and add a test that with boottimes [100.0, 100.4] and a zero-skew reading, `now()` at boottime 100.4 equals `broker_ts + 200 ms`.

## WP08:2

### [MAJOR] `src/jevbot/paper/clock.py:212`

The CLOCK_BOOTTIME sleep detector of 11.4 only runs inside `now()`. `sync()` overwrites `_last_boottime = t1` without ever comparing it to the previous reading, so a host sleep that is first observed by a `sync()` call is erased: `synced` stays True and `needs_sync` goes False, and 11.4's consequence of a gap ("re-sync + reconcile before anything else") can never fire. 11.4 requires a boottime gap > 90 s *or any wake from sleep* to mark the clock unsynced, and 11.4 itself says sync runs "every 60 s, at every phase boundary and before every order" - so sync() winning the race is the normal case, not an exotic one.

**Evidence:** DESIGN 11.4: "A boottime gap > 90 s between heartbeats, or any wake from sleep, marks the clock unsynced: re-sync + reconcile before anything else." Probe /tmp/wp08probe/p1.py (real BrokerClock, scripted boottime): sync at boottime 100.0 -> VM sleeps 600 s -> next call is sync() at boottime 700.0 -> `synced=True, needs_sync=False` ("the wake is invisible"). Control with the identical 600 s gap observed by `now()` first -> `synced=False`. tests/unit/test_paper_clock.py only covers the now()-first order (test_a_boottime_gap_marks_the_clock_unsynced_but_still_answers, test_a_resync_after_a_gap_clears_the_flag), so the sync()-first order is untested.

**Fix:** Do the gap check in `sync()` as well: before assigning, compare `t0` with `_last_boottime` and, when the difference exceeds `_boottime_gap_s`, record it in a sticky observable the runner must consume (e.g. `self._slept = True`, cleared by an explicit `acknowledge_sleep()`), because setting `_unsynced` inside `sync()` would be cleared two lines later. Add a test that syncs, jumps boottime past the gap, syncs again and asserts the wake is still visible.

### [MAJOR] `src/jevbot/paper/clock.py:326` — RESOLVED

`_apply_cross_check` returns `checked`, which is built only from the broker's rows, so a session that XNYS knows and the Alpaca calendar does not is alerted and then silently DROPPED from the calendar. The unit's recorded deviation asserts the opposite ("alert - rather than drop - a session only one calendar knows, so a mandatory-exit day can never disappear (INV-11/INV-21)"), so the deviation register misstates the shipped behaviour on an INV-11 path: 11.3 drives every action time of the day from this calendar, and a day this object calls a non-session gets no cycle at all, hence no mandatory exit.

**Evidence:** tests/unit/test_paper_clock.py:415 asserts exactly the drop: `assert calendar.is_session(date(2026, 11, 27)) is False` for a session present only in the cross-check. Probe /tmp/wp08probe/p3.py reproduces it with a broker calendar missing 2026-11-26: the alert "the cross-check lists a session the broker does not" is raised and `is_session(2026-11-26)` returns False (no ValueError - the date is inside the loaded bounds, it is simply gone). DESIGN 3.1 fixes only "the earlier close wins and an alert is raised"; it does not authorise removing a session the exchange calendar has.

**Fix:** Pick one and make code and deviation agree. Safer: merge a cross-check-only session into `checked` (using the cross-check's own open/close) so the session set is the union and no mandatory-exit day can vanish, keeping the alert. Otherwise leave the drop but rewrite the deviation to say plainly that a session only XNYS knows is dropped, and name the INV-11 exposure it leaves.

### [MINOR] `src/jevbot/paper/clock.py:211`

`now()` is not round-trip compensated. `sync()` measures the skew against local UTC at the MIDPOINT of the call (correct), but then anchors the extrapolation at `_boottime_at_sync = t1`, the end of the round trip. Since `broker_ts` is the broker's time at roughly the midpoint, `now()` runs systematically rtt/2 behind real broker time, and the error is in the unsafe direction: the bot believes it is earlier than it is, so risk check 7 (no submissions after close - 5 min, 11.4) can let an order through late. It also makes `now()` step backwards by rtt/2 at every resync.

**Evidence:** Probe /tmp/wp08probe/p1.py: a 400 ms round trip with a perfectly accurate broker (`skew_ms == 0`, so the compensation demonstrably works for the skew) gives `now() = 19:45:00.200` when the true broker time at that boottime is `19:45:00.400` - `lag_ms = 200.0`, i.e. exactly rtt/2. No test in tests/unit/test_paper_clock.py exercises `now()` with a non-zero rtt: test_now_is_the_broker_stamp_plus_elapsed_boottime_never_the_host_clock uses boottimes=[100.0, 100.0].

**Fix:** Set `self._boottime_at_sync = (t0 + t1) / 2` (the instant `broker_ts` actually refers to) and add a case with a non-zero rtt asserting `now()` immediately after the sync equals `broker_ts + rtt/2`.

### [MINOR] `src/jevbot/paper/broker.py:503`

`open_orders()` builds `GetOrdersRequest(status=OPEN, nested=True)` with no `limit`, so the Alpaca API applies its default page size of 50 and the adapter silently truncates. The Broker protocol (3.4) promises "every working order the broker holds"; reconcile R1 uses this to spot foreign orders and 9.5 K2 polls it until empty, so past 50 working orders (ours plus anything placed by hand in the Alpaca console) a foreign order can be invisible for a whole session.

**Evidence:** alpaca-py 0.44.0 requests.py GetOrdersRequest docstring: "limit (Optional[int]): The maximum number of orders in response. Defaults to 50 and max is 500." Probe /tmp/wp08probe/p2.py prints the wire fields actually sent: `{'status': QueryOrderStatus.OPEN, 'nested': True}` - no limit. The fake never holds more than a handful of orders, so no test can catch it.

**Fix:** Pass `limit=500` (the API maximum) and page with `after`/`until` or `direction` when a full page comes back, or at minimum raise/alert when exactly `limit` rows are returned instead of returning a silently truncated tuple. Add a test with >50 open orders in FakeTradingClient.

### [MINOR] `src/jevbot/paper/broker.py:306`

The equity-flatten branch of `build_order_request` validates that `equity_side` and `equity_qty` are set and that no limit was approved, but never checks `equity_qty >= 1`, while the option branch does check `order.qty <= 0`. A zero- or negative-share flatten is therefore POSTed instead of being caught as the bug it is - the vendor model does not stop it, because pydantic's `_field_is_set` treats 0 as set.

**Evidence:** broker.py:316 refuses `order.qty <= 0` for option orders with "an option order is at least one contract", but the equity branch at broker.py:304-310 passes `qty=int(intent.equity_qty)` unchecked. Probe /tmp/wp08probe/p2.py: `MarketOrderRequest(symbol='SPY', qty=0, side=SELL, time_in_force=DAY, client_order_id='x').to_request_fields()` -> `{'symbol': 'SPY', 'qty': 0.0, ...}` - accepted and serialised. 9.5 K4 defines `equity_qty = abs(shares)`, so 0 can only be a caller bug.

**Fix:** Add `if int(intent.equity_qty) < 1: raise InvariantError(...)` next to the existing equity checks, and add it to the parametrised `test_an_impossible_payload_is_a_bug_not_an_order` table in tests/unit/test_paper_broker.py:226.

### [MINOR] `src/jevbot/paper/broker.py:420`

`AlpacaPaperBroker.__init__` defaults `now` to the HOST wall clock (`datetime.now(UTC)`), and that value becomes `AccountSnapshot.ts` in `account()`. INV-12 says paper trading time comes from the broker clock, and 0.1(10) forbids the wall clock in hashed material; nothing in this unit forces WP12 to inject `clock.now`, so the silent default is host time on a machine whose drift is the exact reason BrokerClock exists.

**Evidence:** broker.py:420 `self._now = now if now is not None else (lambda: datetime.now(UTC))`, consumed at broker.py:494 `account_snapshot(raw, now=self._now())`. AccountSnapshot.ts is a 2.4 field the risk/reconcile side reads. tests/unit/test_paper_broker.py always injects `now=lambda: NOW`, so the default path is never exercised.

**Fix:** Either make `now` a required keyword argument (the runner can pass a late-bound `lambda: clock.now()` after BrokerClock is built on `raw_clock`), or default it to the clock reading and document in the docstring that the host wall clock is a last-resort fallback only used before the first sync.

### [MINOR] `src/jevbot/paper/clock.py:342`

`is_early_close` compares each session's exchange-local close against `_modal_close()`, the MOST COMMON close of whatever window was fetched. When the loaded window contains more early closes than regular ones the comparison inverts and genuine half days report `is_early_close() == False`; with a one- or two-row window it is always False. The Calendar protocol (3.1) promises "early closes honoured (D4, G4)".

**Evidence:** Probe /tmp/wp08probe/p3.py on rows [2026-11-25 16:00, 2026-11-27 13:00, 2026-12-24 13:00]: `is_early_close` returns False for BOTH real 13:00 half days (the modal close is 13:00). The weakness is already enshrined as expected behaviour in two tests - tests/unit/test_paper_clock.py:370 (`is_early_close(session) is False` on a one-row 16:00 calendar whose cross-check forced it to 13:00) and :598 (`AlpacaCalendar(rows).is_early_close(2026-11-27) is False` for a genuine 13:00 row). No consumer exists in src/ yet, which is the only reason this is not worse.

**Fix:** Use the MAXIMUM exchange-local close of the window as the regular close instead of the modal one (a full day is always the longest session, so any shorter close is early); this still contains no time-of-day literal (INV-13). Better still, delegate to the cross-check calendar's `is_early_close` when one was supplied. Then correct the two tests that currently assert the wrong answer.

### [MINOR] `src/jevbot/paper/alpaca_client.py:291`

Step 7's `get_account()` runs outside any wall-clock deadline. INV-08 requires every broker HTTP call to carry a `(connect, read)` timeout AND a hard wall-clock deadline; `make_clients` installs only the former. A server that trickles bytes resets the requests read timeout on every chunk, so boot step B4 can hang indefinitely - before B9 writes the first heartbeat, and `jevbot-paper.service` is `Type=simple` with no startup watchdog (11.10), so neither the dead-man nor the Windows watchdog sees a wedged boot.

**Evidence:** alpaca_client.py:291 calls `assert_tradable_account(trading)` -> `trading.get_account()` directly; the deadline worker of INV-08 lives only in `AlpacaPaperBroker._run_once` (broker.py:446), which is constructed after `make_clients` returns. DESIGN 11.1 step 7 and INV-08 ("Every broker HTTP call has a (connect, read) timeout **and** a hard wall-clock deadline") both apply to this call.

**Fix:** Wrap the step-7 `get_account()` in the same abandoned-worker-thread deadline pattern (a small shared helper used by both `make_clients` and `AlpacaPaperBroker._run_once`), with a boot deadline derived from `orders.call_deadline_s`, and add a hang test that asserts `make_clients` fails closed rather than blocking.
