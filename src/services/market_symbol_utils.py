# -*- coding: utf-8 -*-
"""Shared market-symbol helpers for suffix-only offshore markets.

Keep this module dependency-light so it can be used by data providers, market
context, trading calendars, stock-index loading, and API input normalization
without introducing import cycles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SuffixMarketSpec:
    """A suffix-only Yahoo Finance market rule."""

    market: str
    suffixes: tuple[str, ...]
    digit_lengths: tuple[int, ...]
    # Optional regex for alphanumeric bases (e.g. Madrid ``SAN.MC``). When set
    # it replaces the numeric ``digit_lengths`` rule.
    base_pattern: Optional[str] = None

    def matches_base(self, base: str) -> bool:
        if self.base_pattern is not None:
            return re.fullmatch(self.base_pattern, base) is not None
        return base.isdigit() and len(base) in self.digit_lengths


_SUFFIX_MARKET_SPECS: tuple[SuffixMarketSpec, ...] = (
    SuffixMarketSpec("jp", ("T",), (4, 5)),
    SuffixMarketSpec("kr", ("KS", "KQ"), (6,)),
    # Taiwan support mirrors the same suffix-only pattern; keep it here so the
    # shared helpers stay complete for all yfinance-only offshore markets.
    SuffixMarketSpec("tw", ("TW", "TWO"), (4, 5, 6)),
    # Spain (Bolsa de Madrid / BME continuous market): letter tickers such as
    # SAN.MC, BBVA.MC, ITX.MC, A3M.MC.
    SuffixMarketSpec("es", ("MC",), (), base_pattern=r"[A-Z][A-Z0-9]{0,5}"),
    # Germany (XETRA): letter/digit tickers such as SAP.DE, VOW3.DE, 1COV.DE and
    # ETFs like NUKL.DE; at least one letter so numeric codes never match.
    SuffixMarketSpec("de", ("DE",), (), base_pattern=r"(?=[A-Z0-9]*[A-Z])[A-Z0-9]{1,6}"),
)

_MARKET_TO_SPEC = {spec.market: spec for spec in _SUFFIX_MARKET_SPECS}
_SUFFIX_TO_SPEC = {
    suffix: spec
    for spec in _SUFFIX_MARKET_SPECS
    for suffix in spec.suffixes
}


def split_suffix_symbol(stock_code: str) -> tuple[str, str] | None:
    """Return ``(base, suffix)`` for dotted symbols, upper-cased and stripped."""

    code = (stock_code or "").strip().upper()
    if "." not in code:
        return None
    base, suffix = code.rsplit(".", 1)
    if not base or not suffix:
        return None
    return base, suffix


# Yahoo Finance crypto pairs: BTC-USD, ETH-EUR, SOL-USD. Plain "BTC" is NOT crypto on
# Yahoo (it is a US-listed Bitcoin trust), so the quote currency is required.
_CRYPTO_PAIR_RE = re.compile(r"[A-Z0-9]{2,10}-(?:USD|EUR|GBP|USDT|USDC)")


def is_crypto_symbol(stock_code: str) -> bool:
    return _CRYPTO_PAIR_RE.fullmatch((stock_code or "").strip().upper()) is not None


def get_suffix_market(stock_code: str) -> Optional[str]:
    """Return jp/kr/tw/es/de/crypto for supported Yahoo-only symbols, else None."""

    if is_crypto_symbol(stock_code):
        return "crypto"
    parts = split_suffix_symbol(stock_code)
    if parts is None:
        return None
    base, suffix = parts
    spec = _SUFFIX_TO_SPEC.get(suffix)
    if spec is None:
        return None
    if not spec.matches_base(base):
        return None
    return spec.market


def is_suffix_market_symbol(stock_code: str, market: Optional[str] = None) -> bool:
    """Return whether a stock code is a supported suffix-only Yahoo symbol."""

    detected = get_suffix_market(stock_code)
    if market is None:
        return detected is not None
    return detected == (market or "").strip().lower()


def is_jp_suffix_symbol(stock_code: str) -> bool:
    return is_suffix_market_symbol(stock_code, "jp")


def is_kr_suffix_symbol(stock_code: str) -> bool:
    return is_suffix_market_symbol(stock_code, "kr")


def is_tw_suffix_symbol(stock_code: str) -> bool:
    return is_suffix_market_symbol(stock_code, "tw")


def is_es_suffix_symbol(stock_code: str) -> bool:
    return is_suffix_market_symbol(stock_code, "es")


def is_de_suffix_symbol(stock_code: str) -> bool:
    return is_suffix_market_symbol(stock_code, "de")


# European letter-ticker markets served by Yahoo Finance only (no Chinese sources).
EUROPEAN_SUFFIX_MARKETS = frozenset({"es", "de"})


def normalize_suffix_market_symbol(stock_code: str) -> Optional[str]:
    """Normalize supported suffix-only symbols to upper-case Yahoo form."""

    parts = split_suffix_symbol(stock_code)
    if parts is None:
        return None
    base, suffix = parts
    if get_suffix_market(f"{base}.{suffix}") is None:
        return None
    return f"{base}.{suffix}"


def normalize_crypto_symbol(stock_code: str) -> Optional[str]:
    """Upper-case Yahoo crypto pair (btc-usd -> BTC-USD), or None."""
    code = (stock_code or "").strip().upper()
    return code if is_crypto_symbol(code) else None


def suffix_base_lookup_allowed(canonical_code: str) -> bool:
    """Return True when a suffix-market code may be resolved from its bare base.

    JP/KR intentionally allow stock-index-backed bare-code lookup to support the
    existing MVP behavior. TW remains strict suffix-only for now because its
    follow-up index work is not part of this issue.
    """

    return get_suffix_market(canonical_code) in {"jp", "kr"}


def market_suffixes(market: str) -> tuple[str, ...]:
    spec = _MARKET_TO_SPEC.get((market or "").strip().lower())
    return spec.suffixes if spec else ()
