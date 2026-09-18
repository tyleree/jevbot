"""News sanitiser, entity masker and code-side recency split (DESIGN.md 5.8; D11, INV-16). Hostile input by assumption.

`prepare_news` runs the 5.8 pipeline over the items a `MarketView` served and returns the two lists of the state's news
block plus the counters that go into the provenance sidecar:

    1 pre-filter      too many symbols / not yet knowable => the ITEM is dropped; `updated_at > as_of` => the SUMMARY is
    2 sanitize        unescape, strip tags with html.parser, NFKC, delete control / format / private-use code points,
                      delete URLs, e-mail addresses and @handles, replace `` ` { } [ ] < > | \\ `` with a space,
                      straighten quotes, collapse whitespace, truncate at a word boundary; > 40% non-letter => dropped
    3 is_suspicious   imperative prompt-injection forms, our own option labels / state keys, `noul`, `jev` => DROPPED, counted
    4 mask            dictionary (longest first) + generic people / ticker patterns, dates, numbers, residual proper nouns
    5 leak check      the masked text must pass `canon.ensure_state_safe`'s string rules, else the ITEM is dropped
    6 caps            de-duplicate on the masked headline (keep the older), newest first, `news.max_items` across BOTH
                      lists, then drop oldest until the block fits `news.max_total_chars`
    7 recency split   `knowable_at > cutoff` => `since_previous_session`, else `earlier` - decided HERE, never by Jev

A failing item is dropped, never repaired, and counted. An exception anywhere in the pipeline propagates: the cycle turns
it into `news_pipeline_error`, blocks entries for that underlying and manages existing positions in code only.

Masking is lossy by design and a mitigation, not a cure (B7.1, G9). The structural defence is the request split plus
7.4 / 7.7: text can only veto, re-rank or - together with code-side market-data confirmation - close (INV-16).

`config.MASK_RULES_VERSION` is the ONE version constant of these rules; it is bumped there whenever a rule below changes
(`MaskTerms.version` = `sha256(MASK_RULES_VERSION + mask_terms file bytes)[:12]` is the only `mask_version`).
"""

import html
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from html.parser import HTMLParser
from typing import Any, Final

import msgspec

from jevbot import vocab
from jevbot.canon import dumps_ordered, ensure_state_safe
from jevbot.config import MASK_GROUPS, NewsConfig
from jevbot.errors import StateError
from jevbot.types import MaskTerms, NewsItem

__all__ = [
    "EARLIER_KEY",
    "RECENT_KEY",
    "NewsStats",
    "age_text",
    "drop_oldest",
    "is_suspicious",
    "letter_share",
    "mask",
    "prepare_news",
    "sanitize",
    "source_type",
]

RECENT_KEY: Final = "since_previous_session"
EARLIER_KEY: Final = "earlier"

MASK_MONTH: Final = "[month]"
MASK_DAY: Final = "[day]"
MASK_YEAR: Final = "[year]"
MASK_PERIOD: Final = "[period]"
MASK_NUMBER: Final = "[number]"
MASK_NAME: Final = "[name]"
MASK_LEVEL: Final = "a notable level"

_MIN_LETTER_SHARE: Final = 0.60  # a headline with > 40% non-letter characters is dropped (5.8 step 2)
_HOURS_AS_HOURS: Final = 48  # below 48 hours an age is "<n>h", above it "<n>d"


class NewsStats(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """5.8 counters. `dropped` counts every input item that does not reach the state (`hostile_dropped` is a subset of
    it); `ids` are the ids of the items that did, newest first, for `Provenance.news_ids`."""

    kept: int
    kept_recent: int
    dropped: int
    hostile_dropped: int
    ids: tuple[str, ...]


# ======================================================================================================================
# 2. sanitize
# ======================================================================================================================


class _TagStripper(HTMLParser):
    """Tags removed with `html.parser` - never with a regex (5.8 step 2)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


_WHITESPACE_CONTROLS: Final = str.maketrans({"\t": " ", "\n": " ", "\r": " ", "\f": " ", "\v": " "})
_QUOTES: Final = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "«": '"',
        "»": '"',
    }
)
_DELETED_CATEGORIES: Final[frozenset[str]] = frozenset({"Cc", "Cf", "Co", "Cs", "Cn"})
_BRACKETS: Final = str.maketrans(dict.fromkeys("`{}[]<>|\\", " "))
_URL_RE: Final = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE: Final = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_HANDLE_RE: Final = re.compile(r"(?<![\w.])@\w+")
_SPACES_RE: Final = re.compile(r"\s+")


def _strip_tags(text: str) -> str:
    parser = _TagStripper()
    try:
        parser.feed(text)
        parser.close()
    except Exception:  # a malformed document is unusable input, never a repaired one (5.8)
        return ""
    return parser.text()


def sanitize(text: str, *, max_chars: int) -> str:
    """5.8 step 2. Returns the cleaned text, or `""` when nothing usable is left (the caller drops the item)."""
    if not isinstance(text, str):
        raise TypeError(f"textmask.sanitize takes a str, got {type(text).__name__}")
    if max_chars < 1:
        raise ValueError(f"textmask.sanitize: max_chars must be >= 1, got {max_chars}")
    out = _strip_tags(html.unescape(text))
    out = unicodedata.normalize("NFKC", out)
    out = out.translate(_WHITESPACE_CONTROLS)  # line breaks become spaces BEFORE the C* categories are deleted
    out = "".join(char for char in out if unicodedata.category(char) not in _DELETED_CATEGORIES)
    out = _URL_RE.sub(" ", out)
    out = _EMAIL_RE.sub(" ", out)
    out = _HANDLE_RE.sub(" ", out)
    out = out.translate(_BRACKETS)  # text can no longer imitate state paths, JSON or markup
    out = out.translate(_QUOTES)
    out = _SPACES_RE.sub(" ", out).strip()
    return _truncate(out, max_chars)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind(" ")
    if boundary > 0:
        cut = cut[:boundary]
    return cut.rstrip(" ,;:-")


def letter_share(text: str) -> float:
    """Share of alphabetic characters among the non-whitespace characters (1.0 for an empty text)."""
    body = [char for char in text if not char.isspace()]
    if not body:
        return 1.0
    return sum(1 for char in body if char.isalpha()) / len(body)


# ======================================================================================================================
# 3. is_suspicious (case-insensitive, IMPERATIVE forms only: ordinary macro headlines survive)
# ======================================================================================================================

_SUSPICIOUS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|previous|above|prior|rules?)\b", re.IGNORECASE),
    re.compile(r"\b(system prompt|developer message|you are an? (ai|assistant|language model|model)|as an ai)\b", re.IGNORECASE),
    re.compile(r"\b(answer|respond|reply|output)\s+(only\s+|exactly\s+|with\s+)?[\"']?(yes|no|true|false)\b", re.IGNORECASE),
    re.compile(r"\bset\s+(the\s+)?(probability|score|answer)\b", re.IGNORECASE),
    re.compile(r"(^|\s)(assistant|user|system)\s*:", re.IGNORECASE),
    re.compile(r"\b(jailbreak|base64)\b", re.IGNORECASE),
)
# any underscore-containing identifier that equals a question option label or a state key, plus `noul` and `jev`
_OUR_TOKENS_RE: Final = re.compile(
    r"\b(?:" + "|".join(re.escape(token) for token in sorted(vocab.UNDERSCORE_IDENTIFIERS | {"noul", "jev"})) + r")\b",
    re.IGNORECASE,
)


def is_suspicious(text: str) -> bool:
    """5.8 step 3: True when the text tries to instruct the model or names our own vocabulary."""
    return any(pattern.search(text) is not None for pattern in _SUSPICIOUS) or _OUR_TOKENS_RE.search(text) is not None


# ======================================================================================================================
# 4. mask
# ======================================================================================================================

_WORD_LEFT: Final = r"(?<![A-Za-z0-9])"
_WORD_RIGHT: Final = r"(?![A-Za-z0-9])"

_MONTHS_FULL: Final[tuple[str, ...]] = (
    "january",
    "february",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTHS_ABBR: Final[tuple[str, ...]] = ("Jan", "Feb", "Apr", "Jun", "Jul", "Aug", "Sept", "Sep", "Oct", "Nov", "Dec")
_MONTHS_AMBIGUOUS: Final[tuple[str, ...]] = ("May", "March", "Mar")  # also ordinary English words: only next to a number
_DAYS_FULL: Final[tuple[str, ...]] = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DAYS_ABBR: Final[tuple[str, ...]] = ("Mon", "Tues", "Tue", "Weds", "Wed", "Thurs", "Thur", "Thu", "Fri")
_DAYS_AMBIGUOUS: Final[tuple[str, ...]] = ("Sat", "Sun")

_PEOPLE_TITLE_RE: Final = re.compile(r"\b(?:Chair|Chairman|President|Governor|Secretary|CEO) [A-Z][a-z]+(?: [A-Z][a-z]+)?")
_EXCHANGE_TICKER_RE: Final = re.compile(r"\((?:NYSE|NASDAQ|AMEX|ARCA):\s*[A-Z.]{1,6}\)")
_CASHTAG_RE: Final = re.compile(r"\$([A-Z]{1,6})\b")
_YEAR_RE: Final = re.compile(r"\b(?:19|20)\d{2}\b")
_PERIOD_RE: Final = re.compile(r"\bQ[1-4]\b|\bFY\s?\d{2,4}\b|\b[1-4]Q\b|\bH[12]\b")
_ORDINAL_RE: Final = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\b", re.IGNORECASE)
_SINCE_RE: Final = re.compile(rf"\bsince {re.escape(MASK_MONTH)}(?: {re.escape(MASK_YEAR)})?", re.IGNORECASE)
_RECORD_RE: Final = re.compile(r"\brecord (?:high|low)\b|\ball-time\b", re.IGNORECASE)
_repeat_cache: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {}


def _collapse_repeats(text: str, terms: MaskTerms) -> str:
    """Two adjacent copies of the SAME replacement phrase are one entity to a reader ("a large company a large company"
    from `Apple (NASDAQ: AAPL)`, "a notable level a notable level" from `record high since [month]`). Collapsing them
    keeps the sentence readable and keeps `mask` idempotent."""
    patterns = _repeat_cache.get(terms.version)
    if patterns is None:
        phrases = sorted({*terms.replacements.values(), MASK_LEVEL})
        patterns = tuple((re.compile(rf"{re.escape(phrase)}(?:\s+{re.escape(phrase)})+"), phrase) for phrase in phrases)
        _repeat_cache[terms.version] = patterns
    for pattern, phrase in patterns:
        text = pattern.sub(phrase, text)
    return text


_BASIS_POINTS_RE: Final = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:basis points?|bps|bp)\b", re.IGNORECASE)
_PERCENT_RE: Final = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent\b|pct\b)", re.IGNORECASE)
_NUMBER_RE: Final = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
_PROPER_RUN_RE: Final = re.compile(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+")
_SENTENCE_END: Final = ".!?"

_MAGNITUDES: Final[tuple[tuple[float, str], ...]] = (
    (1.0, "a fraction of a percent"),
    (3.0, "a few percent"),
    (7.0, "several percent"),
)
_MAGNITUDE_LARGE: Final = "a very large percentage"

_dictionary_cache: dict[str, tuple[re.Pattern[str], Mapping[str, str]]] = {}


def _dictionary(terms: MaskTerms) -> tuple[re.Pattern[str] | None, Mapping[str, str]]:
    """One case-insensitive, word-bounded alternation over every listed term, LONGEST FIRST, plus term -> replacement."""
    cached = _dictionary_cache.get(terms.version)
    if cached is not None:
        return cached[0], cached[1]
    replacements: dict[str, str] = {}
    for group in MASK_GROUPS:
        phrase = terms.replacements[group]
        for term in terms.groups.get(group, ()):
            replacements.setdefault(term.casefold(), phrase)
    if not replacements:
        return None, {}
    ordered = sorted(replacements, key=lambda term: (-len(term), term))
    pattern = re.compile(_WORD_LEFT + "(?:" + "|".join(re.escape(term) for term in ordered) + ")" + _WORD_RIGHT, re.IGNORECASE)
    _dictionary_cache[terms.version] = (pattern, replacements)
    return pattern, replacements


def _apply_dictionary(text: str, terms: MaskTerms) -> str:
    pattern, replacements = _dictionary(terms)
    if pattern is None:
        return text
    return pattern.sub(lambda match: replacements[match.group(0).casefold()], text)


def _mask_tokens(text: str, tokens: Sequence[str], replacement: str, *, ignore_case: bool, near_number: bool) -> str:
    """Replace whole-word `tokens`; `near_number` restricts the replacement to occurrences adjacent to a number."""
    if not tokens:
        return text
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(_WORD_LEFT + "(?:" + "|".join(re.escape(token) for token in tokens) + ")" + _WORD_RIGHT, flags)
    if not near_number:
        return pattern.sub(replacement, text)

    def replace(match: re.Match[str]) -> str:
        before = text[max(0, match.start() - 8) : match.start()]
        after = text[match.end() : match.end() + 8]
        if re.search(r"\d[\s,]*$", before) or re.match(r"^[\s,]*\d", after):
            return replacement
        return match.group(0)

    return pattern.sub(replace, text)


def _mask_symbols(text: str, symbols: Sequence[str], replacement: str) -> str:
    """Bare tickers (the item's own `symbols` plus the tokens that carried a `$` cashtag sigil, 5.8 step 4a)."""
    tickers = sorted({symbol for symbol in symbols if symbol and symbol.isalpha()}, key=lambda s: (-len(s), s))
    if not tickers:
        return text
    pattern = re.compile(_WORD_LEFT + "(?:" + "|".join(re.escape(symbol) for symbol in tickers) + ")" + _WORD_RIGHT)
    return pattern.sub(replacement, text)


def _strip_cashtags(text: str) -> tuple[str, list[str]]:
    """Drop the `$` sigil of every cashtag and return the bare tokens: the `$` must not survive as a leak pattern, and
    a cashtag of one of our own funds must still reach the DICTIONARY (`$QQQ` -> "the fund", not "a large company")."""
    tokens: list[str] = []

    def replace(match: re.Match[str]) -> str:
        tokens.append(match.group(1))
        return match.group(1)

    return _CASHTAG_RE.sub(replace, text), tokens


def _magnitude(match: re.Match[str]) -> str:
    try:
        value = abs(float(match.group(1).replace(",", "")))
    except ValueError:  # pragma: no cover - the regex only matches parsable numbers
        return MASK_NUMBER
    for limit, phrase in _MAGNITUDES:
        if value < limit:
            return phrase
    return _MAGNITUDE_LARGE


def _mask_proper_nouns(text: str) -> str:
    """5.8 step 4d: a run of >= 2 consecutive Capitalised words that does NOT start a sentence becomes `[name]`."""

    def replace(match: re.Match[str]) -> str:
        prefix = text[: match.start()].rstrip()
        if not prefix or prefix[-1] in _SENTENCE_END:
            return match.group(0)  # the run starts a sentence: ordinary capitalisation, not a residual entity
        return MASK_NAME

    return _PROPER_RUN_RE.sub(replace, text)


def mask(text: str, terms: MaskTerms, *, symbols: Sequence[str] = ()) -> str:
    """5.8 step 4: dictionary and generic entities, then dates, then numbers, then residual proper nouns.

    Idempotent: `mask(mask(x)) == mask(x)`; every replacement phrase is lower case and free of digits, so no pass can
    match its own output.
    """
    company = terms.replacements["companies"]
    out = _EXCHANGE_TICKER_RE.sub(company, text)  # `(NASDAQ: AAPL)` goes as one token, before the index dictionary sees it
    out, cashtags = _strip_cashtags(out)
    out = _apply_dictionary(out, terms)
    out = _PEOPLE_TITLE_RE.sub(terms.replacements["people"], out)
    out = _mask_symbols(out, [*symbols, *cashtags], company)
    # dates
    out = _mask_tokens(out, _MONTHS_FULL, MASK_MONTH, ignore_case=True, near_number=False)
    out = _mask_tokens(out, _MONTHS_ABBR, MASK_MONTH, ignore_case=False, near_number=False)
    out = _mask_tokens(out, _MONTHS_AMBIGUOUS, MASK_MONTH, ignore_case=False, near_number=True)
    out = _mask_tokens(out, _DAYS_FULL, MASK_DAY, ignore_case=True, near_number=False)
    out = _mask_tokens(out, _DAYS_ABBR, MASK_DAY, ignore_case=False, near_number=False)
    out = _mask_tokens(out, _DAYS_AMBIGUOUS, MASK_DAY, ignore_case=False, near_number=True)
    out = _YEAR_RE.sub(MASK_YEAR, out)
    out = _PERIOD_RE.sub(MASK_PERIOD, out)
    out = _ORDINAL_RE.sub(MASK_DAY, out)
    out = _SINCE_RE.sub(MASK_LEVEL, out)
    out = _RECORD_RE.sub(MASK_LEVEL, out)
    # numbers (Jev is weak at them): basis points, then percentages, then everything else
    out = _BASIS_POINTS_RE.sub(f"{MASK_NUMBER} basis points", out)
    out = _PERCENT_RE.sub(_magnitude, out)
    out = _NUMBER_RE.sub(MASK_NUMBER, out)
    out = _mask_proper_nouns(out)
    out = _collapse_repeats(out, terms)  # "record high since [month]" is ONE notable level, "Apple (NASDAQ: AAPL)" ONE company
    return _SPACES_RE.sub(" ", out).strip()


# ======================================================================================================================
# Item rendering: age and source type
# ======================================================================================================================


def age_text(age: timedelta) -> str:
    """5.6: `"under 1h"`, `"<n>h"` below 48 hours, else `"<n>d"` (whole units, always rounded DOWN)."""
    hours = max(0.0, age.total_seconds()) / 3600.0
    if hours < 1.0:
        return "under 1h"
    if hours < _HOURS_AS_HOURS:
        return f"{int(hours)}h"
    return f"{int(hours // 24)}d"


_PRESS_RELEASE_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "globenewswire",
        "businesswire",
        "business wire",
        "prnewswire",
        "pr newswire",
        "accesswire",
        "newsfile",
        "einpresswire",
        "issuerdirect",
        "press release",
    }
)
_NEWSWIRE_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "benzinga",
        "reuters",
        "bloomberg",
        "dow jones",
        "dowjones",
        "associated press",
        "ap",
        "marketwatch",
        "cnbc",
        "barrons",
        "wsj",
        "the wall street journal",
        "financial times",
        "ft",
    }
)


def source_type(source: str) -> str:
    """The closed enum `vocab.NEWS_SOURCE_TYPES` (`newswire | press_release | other`) from the vendor's source name."""
    name = (source or "").strip().casefold()
    if name in _PRESS_RELEASE_SOURCES:
        return "press_release"
    if name in _NEWSWIRE_SOURCES:
        return "newswire"
    return "other"


# ======================================================================================================================
# The pipeline
# ======================================================================================================================


class _Prepared(msgspec.Struct, frozen=True, kw_only=True):
    knowable_at: datetime
    news_id: str
    headline_key: str
    item: dict[str, Any]


def _prepare_item(
    item: NewsItem, as_of: datetime, cfg: NewsConfig, terms: MaskTerms, underlyings: Sequence[str]
) -> tuple[_Prepared | None, bool]:
    """One item through steps 1-5. Returns `(prepared or None, hostile)`."""
    if len(item.symbols) > cfg.max_symbols_per_item or item.knowable_at > as_of:
        return None, False
    headline = sanitize(item.headline, max_chars=cfg.max_headline_chars)
    if not headline or letter_share(headline) < _MIN_LETTER_SHARE:
        return None, False
    raw_summary = None if item.summary is None or item.updated_at > as_of else item.summary
    summary = sanitize(raw_summary, max_chars=cfg.max_summary_chars) if raw_summary else ""
    if is_suspicious(headline) or (summary and is_suspicious(summary)):
        return None, True
    if cfg.mask:
        headline = mask(headline, terms, symbols=item.symbols)
        summary = mask(summary, terms, symbols=item.symbols) if summary else ""
    if not headline:
        return None, False
    rendered: dict[str, Any] = {
        "age": age_text(as_of - item.knowable_at),
        "source_type": source_type(item.source),
        "headline": headline,
        "summary": summary or None,
    }
    try:  # step 5: a residual leak drops the ITEM, never the run
        ensure_state_safe(rendered, masked=cfg.mask, underlyings=underlyings)
    except StateError:
        return None, False
    return _Prepared(knowable_at=item.knowable_at, news_id=item.id, headline_key=headline.casefold(), item=rendered), False


def _block_chars(lists: Mapping[str, list[dict[str, Any]]]) -> int:
    return len(dumps_ordered(dict(lists)))


def drop_oldest(lists: Mapping[str, list[dict[str, Any]]]) -> bool:
    """Remove the oldest item of the block (the last `earlier` item, else the last recent one). False when empty.

    `state.py` calls it to trim a whole state to `state.max_chars` after the block already fits `news.max_total_chars`.
    """
    for key in (EARLIER_KEY, RECENT_KEY):
        items = lists.get(key)
        if items:
            items.pop()
            return True
    return False


def prepare_news(
    items: Sequence[NewsItem],
    as_of: datetime,
    cutoff: datetime,
    cfg: NewsConfig,
    terms: MaskTerms,
    underlyings: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], NewsStats]:
    """The 5.8 pipeline. `cutoff` is the previous session's decision time: items newer than it are `since_previous_session`.

    The two lists are disjoint and each is newest first. The pending-event questions read the recent list ONLY (6.3 /
    6.5), so a 60-hour-old "decision due tomorrow" can never be read as still pending.
    """
    if as_of.tzinfo is None or cutoff.tzinfo is None:
        raise ValueError("textmask.prepare_news: as_of and cutoff must be tz-aware UTC datetimes")
    prepared: list[_Prepared] = []
    hostile = 0
    for item in items:
        entry, was_hostile = _prepare_item(item, as_of, cfg, terms, underlyings)
        hostile += int(was_hostile)
        if entry is not None:
            prepared.append(entry)
    prepared.sort(key=lambda entry: (entry.knowable_at, entry.news_id), reverse=True)  # newest first

    unique: list[_Prepared] = []
    position: dict[str, int] = {}
    for entry in prepared:  # de-duplicate on the masked headline, keeping the OLDER item
        at = position.get(entry.headline_key)
        if at is None:
            position[entry.headline_key] = len(unique)
            unique.append(entry)
        else:
            seen = unique[at]
            if (entry.knowable_at, entry.news_id) < (seen.knowable_at, seen.news_id):
                unique[at] = entry
    kept = unique[: max(0, cfg.max_items)]

    def split(entries: Sequence[_Prepared]) -> dict[str, list[dict[str, Any]]]:
        return {
            RECENT_KEY: [entry.item for entry in entries if entry.knowable_at > cutoff],
            EARLIER_KEY: [entry.item for entry in entries if entry.knowable_at <= cutoff],
        }

    while kept and _block_chars(split(kept)) > cfg.max_total_chars:
        kept = kept[:-1]  # drop oldest until the block fits `news.max_total_chars`
    lists = split(kept)
    stats = NewsStats(
        kept=len(kept),
        kept_recent=len(lists[RECENT_KEY]),
        dropped=len(items) - len(kept),
        hostile_dropped=hostile,
        ids=tuple(entry.news_id for entry in kept),
    )
    return lists, stats
