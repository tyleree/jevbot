"""`data/store.py`: the 13.1 layout, atomic IO, manifests and the selected-partition hash (DESIGN.md 13.1-13.2, 15.1)."""

import hashlib
import json
import os
import stat
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from jevbot.data import store
from jevbot.data.store import DataStore, FileEntry, Manifest
from jevbot.errors import DataError, DataUnavailable, ManifestMismatch
from jevbot.types import Fidelity, Slot, SnapshotKey

CREATED: datetime = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


class StubProvider:
    """The slice of `ChainProvider` that `selected_paths` uses (the real providers are WP01b's)."""

    def __init__(self, source: str = "mirror", underlyings: tuple[str, ...] = ("SPY",)) -> None:
        self._source = source
        self._underlyings = underlyings

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.EOD_QUOTES

    @property
    def source(self) -> str:
        return self._source

    def underlyings(self) -> tuple[str, ...]:
        return self._underlyings

    def keys(self, underlying: str, start: date, end: date) -> list[SnapshotKey]:
        return []

    def get_chain(self, underlying: str, key: SnapshotKey) -> None:
        return None

    def manifest_hash(self) -> str:
        return "stub"


@pytest.fixture
def data_store(tmp_path: Path) -> DataStore:
    return DataStore(tmp_path / "jevbot-data", create=True)


def frame() -> pd.DataFrame:
    return pd.DataFrame({"session": pd.to_datetime([date(2024, 5, 16), date(2024, 5, 17)]), "close": [1.5, 2.5]})


# ======================================================================================================================
# Layout (13.1)
# ======================================================================================================================


def test_layout_spells_every_entry_of_the_13_1_tree(data_store: DataStore) -> None:
    s, root = data_store, data_store.root
    assert s.mirror_repo_dir() == root / "raw" / "mirror" / "options-dataset-hist"
    assert s.mirror_options_file("SPY", 2012) == s.mirror_repo_dir() / "spy" / "options_2012.parquet"
    assert s.mirror_underlying_prices() == s.mirror_repo_dir() / "underlying_prices.parquet"
    assert s.cboe_csv("vix") == root / "raw" / "cboe" / "VIX_History.csv"
    assert s.cboe_meta("VIX") == root / "raw" / "cboe" / "VIX.meta.json"
    assert s.treasury_csv(2019) == root / "raw" / "treasury" / "bill_rates_2019.csv"
    assert s.fomc_html(date(2026, 9, 17)) == root / "raw" / "fomc" / "fomc_calendar_2026-09-17.html"
    assert s.enriched_partition("mirror", "spy", 2015) == root / "pq" / "enriched" / "mirror" / "SPY" / "year=2015.parquet"
    assert s.bars_path("alpaca", "QQQ") == root / "pq" / "bars" / "alpaca" / "QQQ.parquet"
    assert s.daily_path("mirror", "IWM") == root / "pq" / "daily" / "mirror" / "IWM.parquet"
    assert s.volidx_path("vix9d") == root / "pq" / "volidx" / "VIX9D.parquet"
    assert s.rates_path() == root / "pq" / "rates" / "tbill_13w.parquet"
    assert s.events_path() == root / "pq" / "events" / "events.csv"
    assert s.news_partition(2025, 3) == root / "pq" / "news" / "alpaca" / "year=2025" / "month=03.parquet"
    assert s.news_coverage_path() == root / "pq" / "news" / "alpaca" / "coverage.parquet"
    assert s.recorded_chain("SPY", date(2026, 2, 3), Slot.DEC) == root / "recorded" / "SPY" / "date=2026-02-03" / "chain_dec.parquet"
    assert (
        s.recorded_underlying("SPY", date(2026, 2, 3), Slot.EOD) == root / "recorded" / "SPY" / "date=2026-02-03" / "underlying_eod.parquet"
    )
    assert s.recorded_news(date(2026, 2, 3)) == root / "recorded" / "news" / "date=2026-02-03" / "news.jsonl"
    assert s.recorded_close("SPY", date(2026, 2, 3)) == root / "recorded" / "closes" / "SPY" / "date=2026-02-03.json"
    assert s.recorded_manifest(date(2026, 2, 3)) == root / "recorded" / "manifest" / "date=2026-02-03.json"
    assert s.manifest_path("mirror") == root / "manifests" / "mirror.json"
    assert s.scan_candidates_path() == root / "manifests" / "scan_candidates.json"
    assert s.cache_db() == root / "cache" / "decisions.sqlite"
    assert s.registry_db() == root / "registry.sqlite" and s.spend_db() == root / "state" / "spend.sqlite"
    assert s.run_dir("r1") == root / "runs" / "r1" and s.paper_dir("exp") == root / "paper" / "exp"
    assert s.lock_file() == root / "state" / "jevbot.lock" and s.kill_file() == root / "state" / "KILL"
    assert s.log_file(date(2026, 2, 3)) == root / "logs" / "jevbot-2026-02-03.jsonl"
    assert s.relative(s.rates_path()) == "pq/rates/tbill_13w.parquet"
    assert repr(s).startswith("DataStore(")


def test_dataset_dir_maps_the_provider_source_to_its_directory() -> None:
    assert store.dataset_dir("mirror") == "mirror"
    assert store.dataset_dir("alpaca_recorded") == "recorded"
    assert store.dataset_dir("alpaca_live") == "recorded"  # live snapshots are archived by the recorder
    assert store.dataset_dir("synthetic") == "synthetic"
    assert store.dataset_dir("proxy") == "proxy"  # the gap-fill rows of 5.4 live in the daily dataset
    with pytest.raises(ValueError, match="unknown data source"):
        store.dataset_dir("thetadata")


def test_the_selection_look_back_is_400_calendar_days() -> None:
    assert store.SELECTION_LOOKBACK_DAYS == 400
    assert store.selection_start(date(2015, 1, 5)) == date(2013, 12, 1)  # 400 days before, hand-counted: 2014 has 365


def test_the_schema_constants_match_13_2_and_the_gating_table_of_3_2() -> None:
    assert store.DAILY_COLUMNS[:5] == ("session", "slot", "px_c", "close_c", "close_knowable_at")
    assert store.DAILY_COLUMNS[-1] == "knowable_at" and len(store.DAILY_COLUMNS) == 16
    assert store.DAILY_KEY == ("session", "slot")  # composite key: one row per (session, slot)
    assert store.DAILY_COLUMN_KNOWABLE == {"close_c": "close_knowable_at"}
    # the ONLY column-level gates in the project (3.2): today's high / low / close / volume wait for the next open
    assert set(store.BARS_COLUMN_KNOWABLE) == {"high", "low", "close", "volume"}
    assert set(store.BARS_COLUMN_KNOWABLE.values()) == {"hlcv_knowable_at"}
    assert "volume" not in store.ENRICHED_COLUMNS  # 0.1 item 12: no same-day option volume column
    assert store.ENRICHED_COLUMNS[:2] == ("session", "slot") and "last_session" in store.ENRICHED_COLUMNS
    assert store.EVENTS_KEY == ("kind", "event_date", "underlying")


# ======================================================================================================================
# Atomic IO
# ======================================================================================================================


def test_ensure_dir_creates_every_level_mode_700(tmp_path: Path) -> None:
    target = store.ensure_dir(tmp_path / "a" / "b" / "c")
    assert target.is_dir()
    for path in (tmp_path / "a", tmp_path / "a" / "b", target):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700  # D1: the data tree is not world-readable
    assert store.ensure_dir(target) == target  # idempotent
    file_path = tmp_path / "a" / "file"
    file_path.write_text("x")
    with pytest.raises(DataError, match="not a directory"):
        store.ensure_dir(file_path)


def test_parquet_round_trip_is_atomic_and_leaves_no_temporary_file(data_store: DataStore) -> None:
    path = data_store.volidx_path("VIX")
    data_store.write_parquet(path, frame())
    back = data_store.read_parquet(path)
    pd.testing.assert_frame_equal(back, frame())
    assert [p.name for p in path.parent.iterdir()] == ["VIX.parquet"]  # the tmp file is gone
    # an overwrite replaces the bytes completely
    data_store.write_parquet(path, frame().head(1))
    assert len(data_store.read_parquet(path)) == 1
    assert list(data_store.read_parquet(path, columns=["close"]).columns) == ["close"]


def test_csv_and_json_round_trip(data_store: DataStore) -> None:
    data_store.write_csv(data_store.events_path(), pd.DataFrame({"kind": ["fomc_decision"], "event_date": ["2024-06-12"]}))
    assert data_store.events_path().read_text(encoding="utf-8") == "kind,event_date\nfomc_decision,2024-06-12\n"
    assert list(data_store.read_csv(data_store.events_path())["kind"]) == ["fomc_decision"]
    path = data_store.path("manifests", "x.json")
    data_store.write_json(path, {"b": 1, "a": [2, 3]})
    assert path.read_text(encoding="utf-8") == '{\n  "a": [\n    2,\n    3\n  ],\n  "b": 1\n}\n'
    assert data_store.read_json(path) == {"a": [2, 3], "b": 1}


def test_missing_files_raise_dataunavailable_never_an_empty_frame(data_store: DataStore) -> None:
    for call in (data_store.read_parquet, data_store.read_csv, data_store.read_json):
        with pytest.raises(DataUnavailable):
            call(data_store.path("pq", "nothing.parquet"))


def test_write_once_refuses_to_overwrite(data_store: DataStore) -> None:
    """`recorded/closes/...` is written ONCE (5.4): the official close of a session can never be rewritten."""
    path = data_store.recorded_close("SPY", date(2026, 2, 3))
    data_store.write_json(path, {"session": "2026-02-03", "close_c": 45_000}, once=True)
    with pytest.raises(DataError, match="written once"):
        data_store.write_json(path, {"session": "2026-02-03", "close_c": 1}, once=True)
    assert data_store.read_json(path)["close_c"] == 45_000
    assert [p.name for p in path.parent.iterdir()] == ["date=2026-02-03.json"]


def test_a_failing_write_leaves_the_previous_bytes_and_no_temporary_file(data_store: DataStore) -> None:
    path = data_store.volidx_path("VIX")
    data_store.write_parquet(path, frame())
    before = path.read_bytes()

    def explode(handle: object) -> None:
        raise RuntimeError("vendor exploded mid-write")

    with pytest.raises(RuntimeError, match="vendor exploded"):
        store.atomic_write(path, explode)
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == ["VIX.parquet"]


# ======================================================================================================================
# Manifests (13.1)
# ======================================================================================================================


def test_manifest_hash_is_the_documented_formula() -> None:
    """`manifest_hash = sha256("\\n".join(sorted(f"{path}:{sha256}")))` - hand-spelt, not re-derived from the module."""
    entries = [FileEntry(path="b.parquet", bytes=2, sha256="bb"), FileEntry(path="a.parquet", bytes=1, sha256="aa")]
    expected = hashlib.sha256(b"a.parquet:aa\nb.parquet:bb").hexdigest()
    assert expected == "abdeb3dbf9503fbcd01cb65f23ed85147162c4549414894a27190feb8948555c"  # pinned literal
    assert store.manifest_hash(entries) == expected
    assert store.manifest_hash(reversed(entries)) == expected  # order-insensitive by construction
    assert store.manifest_hash([]) == hashlib.sha256(b"").hexdigest()
    # the SIZE is descriptive: only path and digest enter the hash
    assert store.manifest_hash([FileEntry(path="a.parquet", bytes=999, sha256="aa"), entries[0]]) == expected


def test_sha256_file_streams_the_real_bytes(data_store: DataStore) -> None:
    path = data_store.path("raw", "x.bin")
    data_store.write_bytes(path, b"hello world")
    assert store.sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_manifest_round_trip_carries_the_facts(data_store: DataStore) -> None:
    a, b = data_store.enriched_partition("mirror", "SPY", 2014), data_store.enriched_partition("mirror", "SPY", 2015)
    for path in (a, b):
        data_store.write_parquet(path, frame())
    facts = {"underlying_unadjusted_verified": True, "basis_bp_median": 1.5, "years": [2014, 2015]}
    manifest = data_store.build_manifest("enriched", [b, a], created_at=CREATED, facts=facts)
    assert [e.path for e in manifest.files] == ["pq/enriched/mirror/SPY/year=2014.parquet", "pq/enriched/mirror/SPY/year=2015.parquet"]
    assert manifest.created_at == "2026-01-02T03:04:05Z" and manifest.dataset == "enriched"
    assert manifest.files[0].sha256 == store.sha256_file(a) and manifest.files[0].bytes == a.stat().st_size
    data_store.write_manifest(manifest)
    assert data_store.read_manifest("enriched") == manifest
    assert data_store.verify_manifest("enriched") == manifest
    assert data_store.manifest_facts("enriched") == facts
    assert data_store.manifest_facts("never-written") == {}  # a dataset with no manifest has no facts, not an error


def test_manifest_detects_a_flipped_byte_a_resize_a_deletion_and_an_edited_hash(data_store: DataStore) -> None:
    path = data_store.volidx_path("VIX")
    data_store.write_parquet(path, frame())
    data_store.write_manifest(data_store.build_manifest("volidx", [path], created_at=CREATED))
    assert data_store.verify_manifest("volidx").files[0].path == "pq/volidx/VIX.parquet"

    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0x01  # one flipped bit, same length
    path.write_bytes(bytes(raw))
    with pytest.raises(ManifestMismatch, match="sha256 differs"):
        data_store.verify_manifest("volidx")

    path.write_bytes(bytes(raw[:-1]))
    with pytest.raises(ManifestMismatch, match="bytes, manifest says"):
        data_store.verify_manifest("volidx")

    path.unlink()
    with pytest.raises(ManifestMismatch, match="missing"):
        data_store.verify_manifest("volidx")

    # tampering with the manifest itself: the recomputed hash of the listed files no longer matches
    data_store.write_parquet(path, frame())
    good = data_store.build_manifest("volidx", [path], created_at=CREATED)
    payload = json.loads(json.dumps({**{k: v for k, v in _as_dict(good).items()}, "manifest_hash": "0" * 64}))
    data_store.write_json(data_store.manifest_path("volidx"), payload)
    with pytest.raises(ManifestMismatch, match="manifest_hash does not match"):
        data_store.verify_manifest("volidx")


def test_a_manifest_for_a_missing_file_or_a_broken_manifest_is_refused(data_store: DataStore) -> None:
    with pytest.raises(DataUnavailable, match="no such file"):
        data_store.build_manifest("volidx", [data_store.volidx_path("VIX")], created_at=CREATED)
    with pytest.raises(DataUnavailable, match="no manifest"):
        data_store.read_manifest("volidx")
    data_store.write_json(data_store.manifest_path("volidx"), {"dataset": "volidx"})
    with pytest.raises(DataError, match="not a valid manifest"):
        data_store.read_manifest("volidx")


# ======================================================================================================================
# selected_manifest_hash (13.1): what a run may read, and only that
# ======================================================================================================================


def _build_world(data_store: DataStore) -> None:
    for year in (2014, 2015, 2016):
        data_store.write_parquet(data_store.enriched_partition("mirror", "SPY", year), frame())
    data_store.write_parquet(data_store.enriched_partition("synthetic", "SPY", 2015), frame())
    data_store.write_parquet(data_store.daily_path("mirror", "SPY"), frame())
    data_store.write_parquet(data_store.bars_path("alpaca", "SPY"), frame())
    data_store.write_parquet(data_store.volidx_path("VIX"), frame())
    data_store.write_parquet(data_store.rates_path(), frame())
    data_store.write_csv(data_store.events_path(), frame())
    data_store.write_parquet(data_store.news_coverage_path(), frame())
    for year, month in ((2015, 1), (2015, 2), (2016, 6)):
        data_store.write_parquet(data_store.news_partition(year, month), frame())


def test_selected_paths_cover_the_window_and_nothing_else(data_store: DataStore) -> None:
    _build_world(data_store)
    provider = StubProvider()
    selected = [data_store.relative(p) for p in data_store.selected_paths(provider, ["SPY"], date(2015, 1, 1), date(2015, 12, 31))]
    assert selected == [
        "pq/bars/alpaca/SPY.parquet",
        "pq/daily/mirror/SPY.parquet",
        "pq/enriched/mirror/SPY/year=2015.parquet",
        "pq/events/events.csv",
        "pq/news/alpaca/coverage.parquet",
        "pq/news/alpaca/year=2015/month=01.parquet",
        "pq/news/alpaca/year=2015/month=02.parquet",
        "pq/rates/tbill_13w.parquet",
        "pq/volidx/VIX.parquet",
    ]
    # another provider selects its own enriched dataset
    other = [
        data_store.relative(p) for p in data_store.selected_paths(StubProvider("synthetic"), ["SPY"], date(2015, 1, 1), date(2015, 6, 1))
    ]
    assert "pq/enriched/synthetic/SPY/year=2015.parquet" in other and not [p for p in other if "enriched/mirror" in p]
    # a subset of tables selects a subset of files
    chains_only = data_store.selected_paths(provider, ["SPY"], date(2015, 1, 1), date(2015, 12, 31), tables=["enriched"])
    assert [data_store.relative(p) for p in chains_only] == ["pq/enriched/mirror/SPY/year=2015.parquet"]
    with pytest.raises(ValueError, match="unknown selected tables"):
        data_store.selected_paths(provider, ["SPY"], date(2015, 1, 1), date(2015, 12, 31), tables=["chains"])
    with pytest.raises(ValueError, match="selection window is empty"):
        data_store.selected_paths(provider, ["SPY"], date(2015, 2, 1), date(2015, 1, 1))


def test_selected_manifest_hash_is_stable_under_added_later_partitions(data_store: DataStore) -> None:
    """Adding 2017 data must not invalidate a 2015 backtest (13.1) - but touching a byte the run READS must."""
    _build_world(data_store)
    provider = StubProvider()
    window = (["SPY"], date(2015, 1, 1), date(2015, 12, 31))
    before = data_store.selected_manifest_hash(provider, *window)

    data_store.write_parquet(data_store.enriched_partition("mirror", "SPY", 2017), frame())
    data_store.write_parquet(data_store.news_partition(2017, 3), frame())
    data_store.write_parquet(data_store.enriched_partition("mirror", "QQQ", 2015), frame())  # another underlying
    assert data_store.selected_manifest_hash(provider, *window) == before

    data_store.write_parquet(data_store.enriched_partition("mirror", "SPY", 2015), frame().head(1))
    after = data_store.selected_manifest_hash(provider, *window)
    assert after != before

    # the same files under a different provider are a different data hash
    assert data_store.selected_manifest_hash(StubProvider("alpaca_recorded"), *window) != after


def test_selected_manifest_hash_is_the_manifest_formula_over_the_selection(data_store: DataStore) -> None:
    _build_world(data_store)
    provider = StubProvider()
    window = (["SPY"], date(2015, 1, 1), date(2015, 3, 31))
    paths = data_store.selected_paths(provider, *window)
    lines = [f"@provider/mirror:{hashlib.sha256(b'mirror').hexdigest()}"]
    lines += [f"{data_store.relative(p)}:{hashlib.sha256(p.read_bytes()).hexdigest()}" for p in paths]
    assert data_store.selected_manifest_hash(provider, *window) == hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()
    assert hashlib.sha256(b"mirror").hexdigest() == "00154761637ca746c354a6d9cfbf1da1a92e79afa6bb127bb8a1c434e9c73170"


def test_selected_manifest_hash_of_an_empty_tree_still_identifies_the_provider(tmp_path: Path) -> None:
    empty = DataStore(tmp_path / "empty", create=True)
    synthetic = empty.selected_manifest_hash(StubProvider("synthetic"), ["SPY"], date(2015, 1, 1), date(2015, 1, 31))
    mirror = empty.selected_manifest_hash(StubProvider("mirror"), ["SPY"], date(2015, 1, 1), date(2015, 1, 31))
    assert synthetic != mirror and len(synthetic) == 64


def test_data_store_does_not_create_anything_until_asked(tmp_path: Path) -> None:
    lazy = DataStore(tmp_path / "lazy")
    assert not (tmp_path / "lazy").exists()
    assert lazy.rates_path() == tmp_path / "lazy" / "pq" / "rates" / "tbill_13w.parquet"
    lazy.write_parquet(lazy.rates_path(), frame())
    assert lazy.rates_path().is_file() and os.access(tmp_path / "lazy", os.R_OK)


def _as_dict(manifest: Manifest) -> dict[str, object]:
    import msgspec

    payload = msgspec.to_builtins(manifest)
    assert isinstance(payload, dict)
    return payload
