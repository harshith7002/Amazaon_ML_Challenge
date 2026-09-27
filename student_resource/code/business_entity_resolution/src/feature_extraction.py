"""
High-performance pairwise feature extraction engine for Business Entity Resolution.
Uses rapidfuzz C++ bindings for fast string distance computations.
30 features total — optimized for Macro F0.5 Rank-1 performance.
"""

from typing import Dict, List, Tuple
import numpy as np
from rapidfuzz import fuzz, distance
from .data_loader import EntityRecord

# Feature column names (30 total)
FEATURE_NAMES = [
    # Name similarity (11)
    "name_levenshtein_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_exact_match",
    "name_spaceless_exact",
    "name_spaceless_lev",
    "name_len_diff_ratio",
    # Address similarity (8)
    "address_levenshtein_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_exact_match",
    "address_key_jaccard",
    "address_key_overlap_count",
    "address_number_jaccard",
    "address_number_match_count",
    # Metadata (6)
    "country_exact_match",
    "is_source2",
    "is_source3",
    "is_missing_address_s1",
    "is_missing_address_sj",
    "composite_name_addr_sim",
    # New high-value features (5)
    "name_char_trigram_jaccard",   # catches OCR/transliteration similarity
    "address_number_exact_any",    # strong house number anchor
    "name_prefix4_match",          # first 4 chars same (strong signal for sorted names)
    "both_missing_address",        # both have no address — reduces false penalty
    "country_is_india",            # India has different name patterns (Devanagari etc.)
]


def _char_trigrams(text: str) -> set:
    """Extract character 3-grams from text."""
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}


def compute_pair_features(s1: EntityRecord, sj: EntityRecord) -> List[float]:
    """
    Compute 30 pairwise similarity features between Source 1 record and Target record.
    Optimized for Macro F0.5 — precision-weighted metric.
    """
    # ---- Name Similarities ----
    n1, n2 = s1.norm_name, sj.norm_name
    if n1 and n2:
        name_lev    = fuzz.ratio(n1, n2) / 100.0
        name_part   = fuzz.partial_ratio(n1, n2) / 100.0
        name_tsort  = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_tset   = fuzz.token_set_ratio(n1, n2) / 100.0
        name_jw     = distance.JaroWinkler.similarity(n1, n2)
        name_exact  = 1.0 if n1 == n2 else 0.0
        max_len     = max(len(n1), len(n2), 1)
        name_len_diff = abs(len(n1) - len(n2)) / max_len
        # Char trigram Jaccard (catches OCR errors, cross-script transliterations)
        tg1, tg2    = _char_trigrams(n1), _char_trigrams(n2)
        tg_union    = len(tg1 | tg2)
        name_trigram_j = len(tg1 & tg2) / tg_union if tg_union > 0 else 0.0
        # 4-char prefix match (strong signal when names sorted similarly)
        name_prefix4 = 1.0 if len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4] else 0.0
    else:
        name_lev = name_part = name_tsort = name_tset = name_jw = name_exact = 0.0
        name_len_diff = 1.0
        name_trigram_j = 0.0
        name_prefix4 = 0.0

    # Spaceless name (handles @handles, .com, concatenated names)
    sn1, sn2 = s1.spaceless_name, sj.spaceless_name
    if sn1 and sn2:
        name_sp_exact = 1.0 if sn1 == sn2 else 0.0
        name_sp_lev   = fuzz.ratio(sn1, sn2) / 100.0
    else:
        name_sp_exact = name_sp_lev = 0.0

    # Token overlap (Jaccard + count)
    t1, t2 = set(s1.name_tokens), set(sj.name_tokens)
    if t1 and t2:
        inter = t1 & t2
        name_jaccard     = len(inter) / len(t1 | t2)
        name_overlap_cnt = float(len(inter))
    else:
        name_jaccard     = 0.0
        name_overlap_cnt = 0.0

    # ---- Address Similarities ----
    a1, a2        = s1.norm_address, sj.norm_address
    missing_a1    = 1.0 if not s1.business_address else 0.0
    missing_a2    = 1.0 if not sj.business_address else 0.0
    both_missing  = 1.0 if (missing_a1 and missing_a2) else 0.0

    if a1 and a2:
        addr_lev   = fuzz.ratio(a1, a2) / 100.0
        addr_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_tset  = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_exact = 1.0 if a1 == a2 else 0.0
    else:
        addr_lev = addr_tsort = addr_tset = addr_exact = 0.0

    # Address anchor keys (house_number + street_token composite)
    ak1, ak2 = set(s1.address_keys), set(sj.address_keys)
    if ak1 and ak2:
        ak_inter       = ak1 & ak2
        ak_union       = ak1 | ak2
        ak_jaccard     = len(ak_inter) / len(ak_union)
        ak_overlap_cnt = float(len(ak_inter))
    else:
        ak_jaccard     = 0.0
        ak_overlap_cnt = 0.0

    # Numeric tokens (postal codes, building numbers)
    num1, num2 = set(s1.address_numbers), set(sj.address_numbers)
    if num1 and num2:
        num_inter       = num1 & num2
        num_union       = num1 | num2
        num_jaccard     = len(num_inter) / len(num_union)
        num_overlap_cnt = float(len(num_inter))
        num_exact_any   = 1.0 if len(num_inter) > 0 else 0.0
    else:
        num_jaccard     = 0.0
        num_overlap_cnt = 0.0
        num_exact_any   = 0.0

    # ---- Metadata ----
    c1, c2      = s1.country.lower(), sj.country.lower()
    cntry_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    is_s2       = 1.0 if sj.entity_id.startswith("S2-") else 0.0
    is_s3       = 1.0 if sj.entity_id.startswith("S3-") else 0.0
    is_india    = 1.0 if c1 in ("in", "india", "ind", "भारत") else 0.0
    composite   = (name_tsort + addr_tsort) / 2.0

    return [
        name_lev,
        name_part,
        name_tsort,
        name_tset,
        name_jw,
        name_jaccard,
        name_overlap_cnt,
        name_exact,
        name_sp_exact,
        name_sp_lev,
        name_len_diff,
        addr_lev,
        addr_tsort,
        addr_tset,
        addr_exact,
        ak_jaccard,
        ak_overlap_cnt,
        num_jaccard,
        num_overlap_cnt,
        cntry_match,
        is_s2,
        is_s3,
        missing_a1,
        missing_a2,
        composite,
        # New features
        name_trigram_j,
        num_exact_any,
        name_prefix4,
        both_missing,
        is_india,
    ]


def extract_features_for_candidate_pairs(
    candidate_map: Dict[str, List[str]],
    s1_records: Dict[str, EntityRecord],
    target_records: Dict[str, EntityRecord]
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Extract feature matrix for all (S1, Target) candidate pairs.
    Returns:
      - X: np.ndarray of shape (N_pairs, N_features)
      - pair_keys: list of (s1_id, target_id)
    """
    features_list = []
    pair_keys = []

    for s1_id, cand_ids in candidate_map.items():
        s1_rec = s1_records.get(s1_id)
        if not s1_rec:
            continue
        for tid in cand_ids:
            t_rec = target_records.get(tid)
            if not t_rec:
                continue
            feats = compute_pair_features(s1_rec, t_rec)
            features_list.append(feats)
            pair_keys.append((s1_id, tid))

    if not features_list:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []

    return np.array(features_list, dtype=np.float32), pair_keys
