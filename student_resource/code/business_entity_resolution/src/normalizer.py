"""
Text normalization module for Business Entity Resolution.
Provides robust, high-throughput string, name, address, and token normalizers.
Accurately preserves Indic/Devanagari scripts while standardizing European/Latin accents with an ultra-fast isascii() bypass.
"""

import re
import unicodedata
from typing import List, Set, Optional, Tuple
from .config import LEGAL_SUFFIXES, ADDRESS_ABBREVIATIONS

# Compiled regex patterns for speed
RE_PUNCT = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_AND = re.compile(r"\s*&\s*")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_LATIN_DIACRITICS = re.compile(r"[\u0300-\u036f]")
RE_DOMAINS = re.compile(r"\.(com|in|org|net|co|io|fr|gov|edu)\b", re.IGNORECASE)

# Create sorted legal suffix list (longest first to match compound phrases like 'pvt ltd' before 'ltd')
SORTED_LEGAL_SUFFIXES = sorted(LEGAL_SUFFIXES, key=lambda x: len(x), reverse=True)
LEGAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in SORTED_LEGAL_SUFFIXES) + r")\b",
    re.IGNORECASE
)

# Common stopwords in business names
COMMON_NAME_STOPWORDS = {
    "the", "of", "and", "in", "for", "on", "at", "to", "a", "an",
    "by", "with", "from", "la", "le", "les", "du", "de", "des", "et",
    "का", "की", "के", "में", "से", "और"
}


def strip_accents_latin(text: str) -> str:
    """Normalize unicode characters and remove Latin diacritics with ultra-fast isascii bypass."""
    if not text or text.isascii():
        return text
    nfd = unicodedata.normalize('NFD', text)
    cleaned = RE_LATIN_DIACRITICS.sub('', nfd)
    return unicodedata.normalize('NFC', cleaned)


def clean_basic_text(text: Optional[str]) -> str:
    """Basic low-level cleaning: strip Latin accents, lowercase, unify '&' to 'and'."""
    if text is None or not isinstance(text, str) or text.lower() == 'null' or text.lower() == 'nan':
        return ""
    s = strip_accents_latin(text).lower()
    s = RE_DOMAINS.sub("", s)
    s = RE_AND.sub(" and ", s)
    s = RE_PUNCT.sub(" ", s)
    return RE_WHITESPACE.sub(" ", s).strip()


def normalize_business_name(name: Optional[str]) -> str:
    """
    Standardize a business name:
    1. Clean Latin accents, lowercase, strip web domains, replace '&' with 'and', remove punctuation.
    2. Remove legal suffixes (Pvt Ltd, LLC, Inc, SARL, प्राइवेट लिमिटेड, etc.).
    3. Collapse extra whitespace.
    """
    cleaned = clean_basic_text(name)
    if not cleaned:
        return ""
    stripped = LEGAL_PATTERN.sub(" ", cleaned)
    stripped = RE_WHITESPACE.sub(" ", stripped).strip()
    return stripped if stripped else cleaned


def get_spaceless_name(name: Optional[str]) -> str:
    """Return alphanumeric spaceless string for catching @handles and concatenated names."""
    norm = normalize_business_name(name)
    return re.sub(r"[^a-zA-Z0-9\u0900-\u0D7F]", "", norm)


def normalize_address(address: Optional[str]) -> str:
    """
    Standardize a business address:
    1. Clean Latin accents, lowercase, unify whitespace and punctuation.
    2. Expand common abbreviations (st -> street, rd -> road, ave -> avenue, etc.).
    """
    cleaned = clean_basic_text(address)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    expanded = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def extract_address_numbers(address: Optional[str]) -> List[str]:
    """Extract all numeric tokens from address with leading zeros stripped."""
    if not address or not isinstance(address, str):
        return []
    raw_nums = RE_NUMBERS.findall(address)
    normalized_nums = []
    for n in raw_nums:
        stripped = n.lstrip('0')
        normalized_nums.append(stripped if stripped else '0')
    return normalized_nums


def extract_address_keys(address: Optional[str]) -> List[str]:
    """
    Extract informative address anchoring keys (e.g. house number + street/locality token).
    """
    if not address:
        return []
    norm_addr = normalize_address(address)
    tokens = norm_addr.split()
    if not tokens:
        return []

    keys = []
    for i, t in enumerate(tokens):
        if any(c.isdigit() for c in t):
            clean_num = t.lstrip('0')
            if clean_num:
                for j in range(i + 1, min(i + 3, len(tokens))):
                    if not any(c.isdigit() for c in tokens[j]) and len(tokens[j]) >= 3:
                        keys.append(f"{clean_num}_{tokens[j]}")
                if len(clean_num) >= 3:
                    keys.append(clean_num)
    return keys


def extract_core_tokens(text: Optional[str], min_len: int = 3) -> List[str]:
    """Extract informative non-stopword tokens from cleaned text."""
    if not text:
        return []
    cleaned = normalize_business_name(text)
    tokens = [
        t for t in cleaned.split()
        if len(t) >= min_len and t not in COMMON_NAME_STOPWORDS
    ]
    return tokens


def extract_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from normalized text."""
    if not text:
        return set()
    s = f" {text} "
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}
