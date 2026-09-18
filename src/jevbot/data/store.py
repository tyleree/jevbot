"""`$JEVBOT_DATA` layout, atomic parquet / JSON IO and dataset manifests (DESIGN.md 13.1-13.2; WP01).

Three jobs, all single-sourced here so that no other module ever spells a data path or a schema:

* **Layout.** `DataStore` maps every entry of the 13.1 tree to a `Path`: the raw mirror clone and the raw Cboe / Treasury /
  FOMC downloads, the derived parquet datasets (`pq/enriched`, `pq/bars`, `pq/daily`, `pq/volidx`, `pq/rates`, `pq/events`,
  `pq/news`), the recorder archives, the manifests, the caches and databases, run directories, `state/` and `logs/`.
  Directories are created mode 700 (D1); `config.ensure_data_dir` owns the "outside the git repo" check.
* **Atomic IO.** Every write goes to a temporary file in the destination directory, is flushed and `fsync`-ed, renamed over
  the target and the directory `fsync`-ed: a reader never sees a half-written parquet file, and a crash leaves either the
  old bytes or the new ones. `write_*_once` refuses to overwrite (the recorder's close files, 5.4).
* **Manifests.** `manifests/<dataset>.json` = `{dataset, created_at, files:[{path, bytes, sha256}], manifest_hash, facts}`
  with `manifest_hash = sha256("\\n".join(sorted(f"{path}:{sha256}")))` (13.1). `verify_manifest` re-hashes every listed
  file and the manifest itself, so a flipped byte anywhere is a `ManifestMismatch`.
  `DataStore.selected_manifest_hash(provider, underlyings, start, end, tables)` is a run's `data_manifest_hash`: the same
  formula over the partitions **selected** by (provider, underlyings, window, tables) - so adding a 2027 partition cannot
  invalidate a 2015-2020 backtest, while touching any byte a run can read does.

The schema constants (13.2 `daily`, the `bars` / `volidx` / `rates` / `events` / `news` layouts and their `knowable_at`
column gating) live here too; `data/series.py` builds its `PitTable`s from them.

No clock, no network: `created_at` is passed in by the caller. This module and the rest of `data/` are the only places
(besides `paper/live_data.py` and `paper/recorder.py`) that may call `read_parquet` / `read_csv` (section 1).
"""

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final

import msgspec
import pandas as pd

from jevbot import canon
from jevbot.errors import DataError, DataUnavailable, ManifestMismatch
from jevbot.types import CHAIN_COLUMNS, Slot

if TYPE_CHECKING:
    from jevbot.protocols import ChainProvider

__all__ = [
    "BARS_COLUMNS",
    "BARS_COLUMN_KNOWABLE",
    "DAILY_COLUMNS",
    "DAILY_COLUMN_KNOWABLE",
    "DAILY_KEY",
    "DAILY_SOURCES",
    "DATASET_DIR_BY_SOURCE",
    "ENRICHED_COLUMNS",
    "EVENTS_COLUMNS",
    "EVENTS_KEY",
    "NEWS_COLUMNS",
    "NEWS_COVERAGE_COLUMNS",
    "RATES_COLUMNS",
    "SELECTED_TABLES",
    "SELECTION_LOOKBACK_DAYS",
    "VOLIDX_COLUMNS",
    "DataStore",
    "FileEntry",
    "Manifest",
    "atomic_write",
    "dataset_dir",
    "ensure_dir",
    "manifest_hash",
    "selection_start",
    "sha256_file",
]

# ======================================================================================================================
# Schemas (13.1 / 13.2). One source of truth for the datasets and their point-in-time gating.
# ======================================================================================================================

# 13.2: one row per (session, slot); `close_c` / `close_knowable_at` on `eod` rows only (5.4).
DAILY_COLUMNS: Final[tuple[str, ...]] = (
    "session",
    "slot",
    "px_c",
    "close_c",
    "close_knowable_at",
    "iv30_bp",
    "iv30_2s_bp",
    "iv90_bp",
    "skew25_bp",
    "atm_term_json",
    "rv20_bp",
    "spot_measure",
    "div_unmodelled",
    "basis_suspect",
    "source",
    "knowable_at",
)
DAILY_KEY: Final[tuple[str, ...]] = ("session", "slot")
DAILY_COLUMN_KNOWABLE: Final[Mapping[str, str]] = {"close_c": "close_knowable_at"}
DAILY_SOURCES: Final[tuple[str, ...]] = ("mirror", "recorded", "synthetic", "proxy")

# 13.1 raw daily bars: the ROW gate is `open_knowable_at` (open(D) + 60 s); high / low / close / volume are gated a second
# time by `hlcv_knowable_at` (the next session's open), so today's close can never be read at today's decision (5.1).
BARS_COLUMNS: Final[tuple[str, ...]] = (
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "knowable_at",
    "open_knowable_at",
    "hlcv_knowable_at",
)
BARS_COLUMN_KNOWABLE: Final[Mapping[str, str]] = {
    "high": "hlcv_knowable_at",
    "low": "hlcv_knowable_at",
    "close": "hlcv_knowable_at",
    "volume": "hlcv_knowable_at",
}

VOLIDX_COLUMNS: Final[tuple[str, ...]] = ("session", "close", "knowable_at")
RATES_COLUMNS: Final[tuple[str, ...]] = ("session", "rate_bp", "knowable_at")
EVENTS_COLUMNS: Final[tuple[str, ...]] = (
    "kind",
    "event_date",
    "underlying",
    "amount_cents",
    "scheduled",
    "cancelled",
    "knowable_at",
    "knowable_rule",
    "source_url",
    "fetched_at",
)
EVENTS_KEY: Final[tuple[str, ...]] = ("kind", "event_date", "underlying")
NEWS_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "created_at",
    "updated_at",
    "received_at",
    "knowable_at",
    "headline",
    "summary",
    "source",
    "symbols",
)
NEWS_COVERAGE_COLUMNS: Final[tuple[str, ...]] = ("underlying", "start", "end", "fetched_at")

# `pq/enriched/<source>/<UND>/year=<YYYY>.parquet`: session, slot + CHAIN_COLUMNS (13.1)
ENRICHED_COLUMNS: Final[tuple[str, ...]] = ("session", "slot", *CHAIN_COLUMNS)

# `ChainProvider.source` (2.2: "mirror" | "synthetic" | "alpaca_recorded" | "alpaca_live") -> the dataset directory name
# used by `pq/enriched/<dir>` and `pq/daily/<dir>`. Live snapshots are archived by the recorder, so both Alpaca sources
# read and write the `recorded` dataset.
DATASET_DIR_BY_SOURCE: Final[Mapping[str, str]] = {
    "mirror": "mirror",
    "synthetic": "synthetic",
    "alpaca_recorded": "recorded",
    "alpaca_live": "recorded",
}

# The tables a run's `data_manifest_hash` selects over, and the look-back that covers the 260-session feature windows (13.1).
SELECTED_TABLES: Final[tuple[str, ...]] = ("enriched", "bars", "daily", "volidx", "rates", "events", "news")
SELECTION_LOOKBACK_DAYS: Final = 400

_DIR_MODE: Final = 0o700
_PARQUET_COMPRESSION: Final = "zstd"
_READ_CHUNK: Final = 1 << 20


def selection_start(start: date) -> date:
    """`run.start - LOOKBACK` (13.1): the first calendar day a run's feature windows can reach back to."""
    return start - timedelta(days=SELECTION_LOOKBACK_DAYS)


# ======================================================================================================================
# Manifests
# ======================================================================================================================


class FileEntry(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True, order=True):
    """One file of a dataset: its path RELATIVE to the data root (posix), its size and its sha256."""

    path: str
    bytes: int
    sha256: str


class Manifest(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """`manifests/<dataset>.json` (13.1). `created_at` is descriptive text and is never hashed."""

    dataset: str
    created_at: str  # RFC 3339 UTC (canon.render_as_of)
    files: tuple[FileEntry, ...]
    manifest_hash: str
    facts: dict[str, Any] = msgspec.field(default_factory=dict)


def manifest_hash(files: Iterable[FileEntry]) -> str:
    """THE 13.1 formula: `sha256("\\n".join(sorted(f"{path}:{sha256}")))`. Order- and duplicate-insensitive by construction."""
    return canon.sha256_hex("\n".join(sorted(f"{e.path}:{e.sha256}" for e in files)))


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file (datasets are far larger than memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


# ======================================================================================================================
# The store
# ======================================================================================================================


class DataStore:
    """The `$JEVBOT_DATA` tree of 13.1: paths, atomic IO and manifests.

    `DataStore(root)` never touches the filesystem; `DataStore(root, create=True)` creates the root mode 700. Every writer
    creates the directories it needs. All paths returned are absolute and inside `root`.
    """

    def __init__(self, root: Path | str, *, create: bool = False) -> None:
        self._root = Path(root).expanduser()
        if create:
            ensure_dir(self._root)

    @property
    def root(self) -> Path:
        return self._root

    def __repr__(self) -> str:
        return f"DataStore({str(self._root)!r})"

    # --- layout (13.1) --------------------------------------------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        """`root / parts` - the only place a relative data path is resolved."""
        return self._root.joinpath(*parts)

    def relative(self, path: Path) -> str:
        """`path` as a posix string relative to the data root (what manifests store). ValueError when it is outside."""
        return path.resolve().relative_to(self._root.resolve()).as_posix()

    # raw downloads
    def raw_dir(self) -> Path:
        return self.path("raw")

    def mirror_repo_dir(self) -> Path:
        return self.path("raw", "mirror", "options-dataset-hist")

    def mirror_options_file(self, underlying: str, year: int) -> Path:
        return self.mirror_repo_dir() / underlying.lower() / f"options_{year:04d}.parquet"

    def mirror_underlying_prices(self) -> Path:
        return self.mirror_repo_dir() / "underlying_prices.parquet"

    def cboe_csv(self, symbol: str) -> Path:
        return self.path("raw", "cboe", f"{symbol.upper()}_History.csv")

    def cboe_meta(self, symbol: str) -> Path:
        return self.path("raw", "cboe", f"{symbol.upper()}.meta.json")

    def treasury_csv(self, year: int) -> Path:
        return self.path("raw", "treasury", f"bill_rates_{year:04d}.csv")

    def treasury_meta(self, year: int) -> Path:
        return self.path("raw", "treasury", f"bill_rates_{year:04d}.meta.json")

    def fomc_html(self, fetched: date) -> Path:
        return self.path("raw", "fomc", f"fomc_calendar_{fetched.isoformat()}.html")

    # derived parquet datasets
    def enriched_dir(self, source: str, underlying: str) -> Path:
        return self.path("pq", "enriched", dataset_dir(source), underlying.upper())

    def enriched_partition(self, source: str, underlying: str, year: int) -> Path:
        return self.enriched_dir(source, underlying) / f"year={year:04d}.parquet"

    def enriched_years(self, source: str, underlying: str) -> tuple[int, ...]:
        """The years that exist for (source, underlying), ascending."""
        directory = self.enriched_dir(source, underlying)
        if not directory.is_dir():
            return ()
        years: list[int] = []
        for child in directory.glob("year=*.parquet"):
            text = child.name[len("year=") : -len(".parquet")]
            if text.isdigit():
                years.append(int(text))
        return tuple(sorted(years))

    def bars_path(self, source: str, underlying: str) -> Path:
        return self.path("pq", "bars", source, f"{underlying.upper()}.parquet")

    def bars_sources(self) -> tuple[str, ...]:
        return self._subdirectories(self.path("pq", "bars"))

    def daily_path(self, source: str, underlying: str) -> Path:
        return self.path("pq", "daily", dataset_dir(source), f"{underlying.upper()}.parquet")

    def daily_sources(self) -> tuple[str, ...]:
        return self._subdirectories(self.path("pq", "daily"))

    def volidx_path(self, name: str) -> Path:
        return self.path("pq", "volidx", f"{name.upper()}.parquet")

    def volidx_names(self) -> tuple[str, ...]:
        directory = self.path("pq", "volidx")
        if not directory.is_dir():
            return ()
        return tuple(sorted(child.stem for child in directory.glob("*.parquet")))

    def rates_path(self) -> Path:
        return self.path("pq", "rates", "tbill_13w.parquet")

    def events_path(self) -> Path:
        return self.path("pq", "events", "events.csv")

    def news_dir(self) -> Path:
        return self.path("pq", "news", "alpaca")

    def news_partition(self, year: int, month: int) -> Path:
        if not 1 <= month <= 12:
            raise ValueError(f"month must be 1..12, got {month}")
        return self.news_dir() / f"year={year:04d}" / f"month={month:02d}.parquet"

    def news_coverage_path(self) -> Path:
        return self.news_dir() / "coverage.parquet"

    # recorder archives
    def recorded_dir(self, underlying: str, session: date) -> Path:
        return self.path("recorded", underlying.upper(), f"date={session.isoformat()}")

    def recorded_chain(self, underlying: str, session: date, slot: Slot) -> Path:
        return self.recorded_dir(underlying, session) / f"chain_{slot.value}.parquet"

    def recorded_underlying(self, underlying: str, session: date, slot: Slot) -> Path:
        return self.recorded_dir(underlying, session) / f"underlying_{slot.value}.parquet"

    def recorded_news(self, session: date) -> Path:
        return self.path("recorded", "news", f"date={session.isoformat()}", "news.jsonl")

    def recorded_clock(self, session: date) -> Path:
        return self.path("recorded", "clock", f"date={session.isoformat()}", "clock.jsonl")

    def recorded_close(self, underlying: str, session: date) -> Path:
        return self.path("recorded", "closes", underlying.upper(), f"date={session.isoformat()}.json")

    def recorded_manifest(self, session: date) -> Path:
        return self.path("recorded", "manifest", f"date={session.isoformat()}.json")

    # manifests, databases, runs, state, logs
    def manifests_dir(self) -> Path:
        return self.path("manifests")

    def manifest_path(self, dataset: str) -> Path:
        return self.manifests_dir() / f"{dataset}.json"

    def scan_candidates_path(self) -> Path:
        return self.manifests_dir() / "scan_candidates.json"

    def cache_db(self) -> Path:
        return self.path("cache", "decisions.sqlite")

    def registry_db(self) -> Path:
        return self.path("registry.sqlite")

    def spend_db(self) -> Path:
        return self.path("state", "spend.sqlite")

    def run_dir(self, run_id: str) -> Path:
        return self.path("runs", run_id)

    def paper_dir(self, experiment: str) -> Path:
        return self.path("paper", experiment)

    def state_dir(self) -> Path:
        return self.path("state")

    def lock_file(self) -> Path:
        return self.state_dir() / "jevbot.lock"

    def heartbeat_file(self) -> Path:
        return self.state_dir() / "heartbeat.json"

    def kill_file(self) -> Path:
        return self.state_dir() / "KILL"

    def rearm_file(self) -> Path:
        return self.state_dir() / "REARM"

    def alert_file(self) -> Path:
        return self.state_dir() / "ALERT"

    def logs_dir(self) -> Path:
        return self.path("logs")

    def log_file(self, day: date) -> Path:
        return self.logs_dir() / f"jevbot-{day.isoformat()}.jsonl"

    def probes_step0_dir(self) -> Path:
        return self.path("probes", "step0")

    def probes_alpaca_dir(self) -> Path:
        return self.path("probes", "alpaca")

    def _subdirectories(self, directory: Path) -> tuple[str, ...]:
        if not directory.is_dir():
            return ()
        return tuple(sorted(child.name for child in directory.iterdir() if child.is_dir()))

    # --- atomic IO ------------------------------------------------------------------------------------------------------

    def write_bytes(self, path: Path, payload: bytes, *, once: bool = False) -> Path:
        """Atomically write `payload` to `path` (tmp + fsync + rename + directory fsync). `once`: refuse to overwrite."""
        return atomic_write(path, lambda handle: handle.write(payload), once=once)

    def write_text(self, path: Path, text: str, *, once: bool = False) -> Path:
        return self.write_bytes(path, text.encode("utf-8"), once=once)

    def write_json(self, path: Path, obj: object, *, once: bool = False) -> Path:
        """Pretty, key-sorted JSON with a trailing newline: a manifest diff in git review is readable."""
        return self.write_text(path, json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n", once=once)

    def read_json(self, path: Path) -> Any:
        if not path.is_file():
            raise DataUnavailable(f"no such file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_parquet(self, path: Path, df: pd.DataFrame, *, once: bool = False) -> Path:
        """Atomically write a DataFrame as parquet (pyarrow, zstd, no index)."""

        def dump(handle: IO[bytes]) -> None:
            df.to_parquet(handle, engine="pyarrow", index=False, compression=_PARQUET_COMPRESSION)

        return atomic_write(path, dump, once=once)

    def read_parquet(self, path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Read a parquet partition. `DataUnavailable` when the file is absent (never an empty frame by accident)."""
        if not path.is_file():
            raise DataUnavailable(f"no such parquet file: {path}")
        frame: pd.DataFrame = pd.read_parquet(path, engine="pyarrow", columns=list(columns) if columns is not None else None)
        return frame

    def write_csv(self, path: Path, df: pd.DataFrame, *, once: bool = False) -> Path:
        return self.write_text(path, df.to_csv(index=False, lineterminator="\n"), once=once)

    def read_csv(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        if not path.is_file():
            raise DataUnavailable(f"no such csv file: {path}")
        frame: pd.DataFrame = pd.read_csv(path, **kwargs)
        return frame

    # --- manifests ------------------------------------------------------------------------------------------------------

    def build_manifest(
        self, dataset: str, paths: Iterable[Path], *, created_at: datetime, facts: Mapping[str, Any] | None = None
    ) -> Manifest:
        """Hash every file of `paths` (missing files are a `DataUnavailable`) and assemble the manifest."""
        entries: list[FileEntry] = []
        for path in paths:
            if not path.is_file():
                raise DataUnavailable(f"dataset {dataset}: no such file: {path}")
            entries.append(FileEntry(path=self.relative(path), bytes=path.stat().st_size, sha256=sha256_file(path)))
        entries.sort()
        return Manifest(
            dataset=dataset,
            created_at=canon.render_as_of(created_at),
            files=tuple(entries),
            manifest_hash=manifest_hash(entries),
            facts=dict(facts or {}),
        )

    def write_manifest(self, manifest: Manifest) -> Path:
        path = self.manifest_path(manifest.dataset)
        self.write_json(path, msgspec.to_builtins(manifest))
        return path

    def read_manifest(self, dataset: str) -> Manifest:
        path = self.manifest_path(dataset)
        if not path.is_file():
            raise DataUnavailable(f"no manifest for dataset {dataset}: {path}")
        try:
            return msgspec.convert(self.read_json(path), type=Manifest)
        except (msgspec.ValidationError, json.JSONDecodeError) as exc:
            raise DataError(f"manifest {path} is not a valid manifest: {exc}") from exc

    def verify_manifest(self, dataset: str) -> Manifest:
        """Re-hash every listed file and the manifest itself. A missing file, a flipped byte or an edited manifest is a
        `ManifestMismatch` naming the offending paths."""
        manifest = self.read_manifest(dataset)
        problems: list[str] = []
        if manifest.manifest_hash != manifest_hash(manifest.files):
            problems.append("manifest_hash does not match the listed files")
        for entry in manifest.files:
            path = self.path(entry.path)
            if not path.is_file():
                problems.append(f"{entry.path}: missing")
                continue
            size = path.stat().st_size
            if size != entry.bytes:
                problems.append(f"{entry.path}: {size} bytes, manifest says {entry.bytes}")
                continue
            if sha256_file(path) != entry.sha256:
                problems.append(f"{entry.path}: sha256 differs")
        if problems:
            shown = "; ".join(problems[:5])
            more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
            raise ManifestMismatch(f"dataset {dataset} does not match its manifest: {shown}{more}")
        return manifest

    def manifest_facts(self, dataset: str) -> dict[str, Any]:
        """The `facts` block of a dataset's manifest, or `{}` when the manifest does not exist (13.1: the G13 verdict and
        `underlying_unadjusted_verified` live here)."""
        try:
            return dict(self.read_manifest(dataset).facts)
        except DataUnavailable:
            return {}

    # --- the selection a run reads (13.1) -------------------------------------------------------------------------------

    def selected_paths(
        self,
        provider: "ChainProvider",
        underlyings: Sequence[str],
        start: date,
        end: date,
        tables: Sequence[str] = SELECTED_TABLES,
    ) -> tuple[Path, ...]:
        """The partitions a run over `(provider, underlyings, [start, end], tables)` may open, sorted and existing only.

        `start` is already `run.start - LOOKBACK` (the caller applies `selection_start`). Year / month partitions outside
        the window are excluded - that is what keeps an old backtest's hash stable when later data arrives.
        """
        if end < start:
            raise ValueError(f"selection window is empty: {start.isoformat()}..{end.isoformat()}")
        unknown = [t for t in tables if t not in SELECTED_TABLES]
        if unknown:
            raise ValueError(f"unknown selected tables {unknown}: expected a subset of {list(SELECTED_TABLES)}")
        source = provider.source
        names = [u.upper() for u in underlyings]
        wanted = set(tables)
        out: set[Path] = set()
        if "enriched" in wanted:
            for underlying in names:
                for year in range(start.year, end.year + 1):
                    out.add(self.enriched_partition(source, underlying, year))
        if "bars" in wanted:
            for bars_source in self.bars_sources():
                out.update(self.bars_path(bars_source, underlying) for underlying in names)
        if "daily" in wanted:
            out.update(self.daily_path(source, underlying) for underlying in names)
        if "volidx" in wanted:
            out.update(self.volidx_path(name) for name in self.volidx_names())
        if "rates" in wanted:
            out.add(self.rates_path())
        if "events" in wanted:
            out.add(self.events_path())
        if "news" in wanted:
            out.add(self.news_coverage_path())
            out.update(self.news_partition(year, month) for year, month in _months(start, end))
        return tuple(sorted(path for path in out if path.is_file()))

    def selected_manifest_hash(
        self,
        provider: "ChainProvider",
        underlyings: Sequence[str],
        start: date,
        end: date,
        tables: Sequence[str] = SELECTED_TABLES,
    ) -> str:
        """A run's `data_manifest_hash` (13.1): the 13.1 manifest formula over `selected_paths`, plus one marker line for
        the provider so that two providers over the same files are never confused.

        The marker is the provider's `source` alone, never `provider.manifest_hash()`: a whole-dataset provider hash would
        change when 2027 data is added and would defeat the purpose of selecting. A file-less provider's own parameters
        (the synthetic seed, 5.10) reach the run through `config_hash`, which sits in the same RUN_START payload.
        """
        entries = [FileEntry(path=f"@provider/{provider.source}", bytes=0, sha256=canon.sha256_hex(provider.source))]
        for path in self.selected_paths(provider, underlyings, start, end, tables):
            entries.append(FileEntry(path=self.relative(path), bytes=path.stat().st_size, sha256=sha256_file(path)))
        return manifest_hash(entries)


# ======================================================================================================================
# Helpers
# ======================================================================================================================


def dataset_dir(source: str) -> str:
    """`ChainProvider.source` -> the `pq/enriched` / `pq/daily` directory name (`alpaca_recorded` -> `recorded`)."""
    mapped = DATASET_DIR_BY_SOURCE.get(source)
    if mapped is not None:
        return mapped
    if source in DAILY_SOURCES:
        return source
    raise ValueError(f"unknown data source {source!r}: expected one of {sorted(set(DATASET_DIR_BY_SOURCE) | set(DAILY_SOURCES))}")


def ensure_dir(path: Path) -> Path:
    """Create `path` and every missing parent with mode 700 (D1); an existing directory is left as it is."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=_DIR_MODE, exist_ok=True)
    if not path.is_dir():
        raise DataError(f"{path} exists and is not a directory")
    return path


def atomic_write(path: Path, dump: Callable[[IO[bytes]], object], *, once: bool = False) -> Path:
    """Write through a temporary file in the destination directory: `dump(handle)`, flush, fsync, rename, fsync the dir.

    A reader therefore sees either the previous bytes or the complete new ones - never a truncated parquet footer.
    `once = True` refuses to replace an existing file (`DataError`): the recorder's close archives are written once (5.4).
    """
    if once and path.exists():
        raise DataError(f"refusing to overwrite {path} (written once, 5.4)")
    directory = ensure_dir(path.parent)
    tmp = directory / f".{path.name}.tmp"
    try:
        with tmp.open("wb") as handle:
            dump(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if once and path.exists():
            raise DataError(f"refusing to overwrite {path} (written once, 5.4)")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def _months(start: date, end: date) -> tuple[tuple[int, int], ...]:
    """Every (year, month) touched by [start, end], ascending."""
    out: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return tuple(out)
