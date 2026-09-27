"""
Ultra-High-Performance Parallel Test Inference Engine for Business Entity Resolution.
ML Challenge 2026 - Optimized for Macro F0.5 Rank-1.
"""

import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pyarrow.csv as pv
from rapidfuzz import fuzz, distance

# Setup paths
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
ROOT_DIR = PKG_DIR.parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    TEST_S1, TEST_S2, TEST_S3,
    MATCHING_OUTPUT, CANDIDATE_OUTPUT,
    MODEL_DIR, DECISION_THRESHOLD, MARGIN_THRESHOLD
)
from src.normalizer import (
    normalize_business_name,
    normalize_address,
    get_spaceless_name,
    extract_core_tokens,
    extract_address_numbers,
    extract_address_keys,
    COMMON_NAME_STOPWORDS
)
from src.model import EntityMatchingModel

# Optimal tuned thresholds from 3-fold CV (F0.5 = 0.9876, Fold 1 Peak = 0.9881)
TUNED_THRESHOLD = 0.625
TUNED_MARGIN = 0.300
MAX_CANDIDATES = 65

RE_NUMBERS = re.compile(r"\b\d+\b")


def _char_trigrams(text: str) -> set:
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}


def compute_fast_features(
    s1_norm_name: str,
    s1_spaceless: str,
    s1_name_toks: tuple,
    s1_norm_addr: str,
    s1_addr_keys: tuple,
    s1_addr_nums: tuple,
    s1_cntry: str,
    s1_has_addr: bool,
    t_eid: str,
    t_norm_name: str,
    t_spaceless: str,
    t_name_toks: tuple,
    t_norm_addr: str,
    t_addr_keys: tuple,
    t_addr_nums: tuple,
    t_cntry: str,
    t_has_addr: bool
) -> List[float]:
    """Compute 30 pairwise similarity features with maximum speed."""
    # Name similarities
    n1, n2 = s1_norm_name, t_norm_name
    if n1 and n2:
        name_lev = fuzz.ratio(n1, n2) / 100.0
        name_part = fuzz.partial_ratio(n1, n2) / 100.0
        name_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_tset = fuzz.token_set_ratio(n1, n2) / 100.0
        name_jw = distance.JaroWinkler.similarity(n1, n2)
        name_exact = 1.0 if n1 == n2 else 0.0
        max_len = max(len(n1), len(n2), 1)
        name_len_diff = abs(len(n1) - len(n2)) / max_len
        tg1, tg2 = _char_trigrams(n1), _char_trigrams(n2)
        tg_union = len(tg1 | tg2)
        name_trigram_j = len(tg1 & tg2) / tg_union if tg_union > 0 else 0.0
        name_prefix4 = 1.0 if len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4] else 0.0
    else:
        name_lev = name_part = name_tsort = name_tset = name_jw = name_exact = 0.0
        name_len_diff = 1.0
        name_trigram_j = 0.0
        name_prefix4 = 0.0

    # Spaceless name
    sn1, sn2 = s1_spaceless, t_spaceless
    if sn1 and sn2:
        name_sp_exact = 1.0 if sn1 == sn2 else 0.0
        name_sp_lev = fuzz.ratio(sn1, sn2) / 100.0
    else:
        name_sp_exact = name_sp_lev = 0.0

    # Token overlap
    t1, t2 = set(s1_name_toks), set(t_name_toks)
    if t1 and t2:
        inter = t1 & t2
        name_jaccard = len(inter) / len(t1 | t2)
        name_overlap_cnt = float(len(inter))
    else:
        name_jaccard = 0.0
        name_overlap_cnt = 0.0

    # Address similarities
    a1, a2 = s1_norm_addr, t_norm_addr
    missing_a1 = 0.0 if s1_has_addr else 1.0
    missing_a2 = 0.0 if t_has_addr else 1.0
    both_missing = 1.0 if (missing_a1 and missing_a2) else 0.0

    if a1 and a2:
        addr_lev = fuzz.ratio(a1, a2) / 100.0
        addr_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_tset = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_exact = 1.0 if a1 == a2 else 0.0
    else:
        addr_lev = addr_tsort = addr_tset = addr_exact = 0.0

    # Address anchor keys
    ak1, ak2 = set(s1_addr_keys), set(t_addr_keys)
    if ak1 and ak2:
        ak_inter = ak1 & ak2
        ak_jaccard = len(ak_inter) / len(ak1 | ak2)
        ak_overlap_cnt = float(len(ak_inter))
    else:
        ak_jaccard = 0.0
        ak_overlap_cnt = 0.0

    # Numeric tokens
    num1, num2 = set(s1_addr_nums), set(t_addr_nums)
    if num1 and num2:
        num_inter = num1 & num2
        num_jaccard = len(num_inter) / len(num1 | num2)
        num_overlap_cnt = float(len(num_inter))
        num_exact_any = 1.0 if len(num_inter) > 0 else 0.0
    else:
        num_jaccard = 0.0
        num_overlap_cnt = 0.0
        num_exact_any = 0.0

    # Metadata
    cntry_match = 1.0 if (s1_cntry and t_cntry and s1_cntry == t_cntry) else 0.0
    is_s2 = 1.0 if t_eid.startswith("S2-") else 0.0
    is_s3 = 1.0 if t_eid.startswith("S3-") else 0.0
    is_india = 1.0 if s1_cntry in ("in", "india", "ind", "भारत") else 0.0
    composite = (name_tsort + addr_tsort) / 2.0

    return [
        name_lev, name_part, name_tsort, name_tset, name_jw,
        name_jaccard, name_overlap_cnt, name_exact,
        name_sp_exact, name_sp_lev, name_len_diff,
        addr_lev, addr_tsort, addr_tset, addr_exact,
        ak_jaccard, ak_overlap_cnt,
        num_jaccard, num_overlap_cnt,
        cntry_match, is_s2, is_s3,
        missing_a1, missing_a2, composite,
        name_trigram_j, num_exact_any, name_prefix4,
        both_missing, is_india
    ]


def extract_addr_keys_from_norm(norm_addr: str) -> tuple:
    if not norm_addr:
        return ()
    tokens = norm_addr.split()
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
    return tuple(keys)


def extract_addr_nums_from_raw(addr_s: str) -> tuple:
    if not addr_s:
        return ()
    raw = RE_NUMBERS.findall(addr_s)
    return tuple(n.lstrip('0') or '0' for n in raw)


def load_and_normalize_table(file_path):
    """Load and normalize TSV into flat columnar lists with PyArrow speed."""
    file_path = Path(file_path)
    print(f"  Reading {file_path.name} with PyArrow...", flush=True)
    t0 = time.time()
    opts = pv.ReadOptions(use_threads=True)
    parse_opts = pv.ParseOptions(delimiter="\t", quote_char=False, double_quote=False, escape_char=False)
    table = pv.read_csv(str(file_path), read_options=opts, parse_options=parse_opts)
    
    eids = table.column("entity_id").to_pylist()
    raw_names = table.column("business_name").to_pylist()
    raw_addrs = table.column("business_address").to_pylist()
    raw_cntrys = table.column("country").to_pylist()
    n = len(eids)
    print(f"    Loaded {n:,} rows in {time.time()-t0:.2f}s. Normalizing in single pass...", flush=True)

    t0 = time.time()
    norm_names = []
    spaceless = []
    name_tokens = []
    norm_addrs = []
    addr_keys = []
    addr_nums = []
    has_addr = []
    countries = []

    for name, addr, cntry in zip(raw_names, raw_addrs, raw_cntrys):
        name_s = str(name or "")
        addr_s = str(addr or "")
        c_s = str(cntry or "").strip().lower()

        nn = normalize_business_name(name_s)
        sp = nn.replace(" ", "")
        toks = tuple(t for t in nn.split() if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS)
        na = normalize_address(addr_s)
        ak = extract_addr_keys_from_norm(na)
        an = extract_addr_nums_from_raw(addr_s)
        ha = bool(addr_s.strip())

        norm_names.append(nn)
        spaceless.append(sp)
        name_tokens.append(toks)
        norm_addrs.append(na)
        addr_keys.append(ak)
        addr_nums.append(an)
        has_addr.append(ha)
        countries.append(c_s)

    print(f"    Normalized {n:,} rows in {time.time()-t0:.2f}s ({n/(time.time()-t0):.0f} recs/s)", flush=True)
    return eids, norm_names, spaceless, name_tokens, norm_addrs, addr_keys, addr_nums, has_addr, countries


class FastTargetIndex:
    """Zero-overhead inverted index with fast candidate scoring."""
    
    def __init__(self, max_token_freq: int = 3000, max_addr_freq: int = 800):
        self.max_token_freq = max_token_freq
        self.max_addr_freq = max_addr_freq

        # Target columnar arrays
        self.eids = []
        self.norm_names = []
        self.spaceless = []
        self.name_tokens = []
        self.norm_addrs = []
        self.addr_keys = []
        self.addr_nums = []
        self.has_addr = []
        self.countries = []

        # Inverted index mappings: key -> list of int (global index)
        self.exact_name_idx = defaultdict(list)
        self.spaceless_idx = defaultdict(list)
        self.prefix_idx = defaultdict(list)
        self.token_idx = defaultdict(list)
        self.token_counts = defaultdict(int)
        self.addr_key_idx = defaultdict(list)
        self.addr_key_counts = defaultdict(int)

    def add_table_data(self, eids, norm_names, spaceless, name_tokens, norm_addrs, addr_keys, addr_nums, has_addr, countries):
        base_offset = len(self.eids)
        n = len(eids)

        self.eids.extend(eids)
        self.norm_names.extend(norm_names)
        self.spaceless.extend(spaceless)
        self.name_tokens.extend(name_tokens)
        self.norm_addrs.extend(norm_addrs)
        self.addr_keys.extend(addr_keys)
        self.addr_nums.extend(addr_nums)
        self.has_addr.extend(has_addr)
        self.countries.extend(countries)

        print(f"    Indexing {n:,} target rows (offset {base_offset:,})...")
        t0 = time.time()
        for i in range(n):
            g_idx = base_offset + i
            c = countries[i]
            nn = norm_names[i]
            sp = spaceless[i]
            toks = name_tokens[i]
            aks = addr_keys[i]

            if nn:
                self.exact_name_idx[(c, nn)].append(g_idx)
                if len(nn) >= 6:
                    self.prefix_idx[(c, nn[:6])].append(g_idx)

            if sp:
                self.spaceless_idx[(c, sp)].append(g_idx)

            for t in set(toks):
                self.token_counts[(c, t)] += 1
                self.token_idx[(c, t)].append(g_idx)

            for ak in set(aks):
                self.addr_key_counts[(c, ak)] += 1
                self.addr_key_idx[(c, ak)].append(g_idx)

        print(f"    Indexed in {time.time()-t0:.2f}s")

    def prune(self):
        """Prune ubiquitous tokens and address keys to prevent candidate explosion."""
        print("  Pruning high-frequency tokens & address keys...")
        t0 = time.time()
        pruned_toks = [k for k, count in self.token_counts.items() if count > self.max_token_freq]
        for k in pruned_toks:
            del self.token_idx[k]
        
        pruned_aks = [k for k, count in self.addr_key_counts.items() if count > self.max_addr_freq]
        for k in pruned_aks:
            del self.addr_key_idx[k]
        print(f"    Pruned {len(pruned_toks):,} tokens and {len(pruned_aks):,} address keys in {time.time()-t0:.2f}s")

    def get_candidates(self, s1_cntry: str, s1_norm_name: str, s1_spaceless: str, s1_name_toks: tuple, s1_addr_keys: tuple, max_cands: int = MAX_CANDIDATES) -> List[int]:
        c = s1_cntry
        scores = defaultdict(float)

        # 1. Exact name match
        if s1_norm_name:
            for idx in self.exact_name_idx.get((c, s1_norm_name), []):
                scores[idx] += 6.0
            if len(s1_norm_name) >= 6:
                prefs = self.prefix_idx.get((c, s1_norm_name[:6]), [])
                if len(prefs) <= 200:
                    for idx in prefs:
                        scores[idx] += 1.8

        # 2. Spaceless name
        if s1_spaceless:
            for idx in self.spaceless_idx.get((c, s1_spaceless), []):
                scores[idx] += 5.5

        # 3. Informative name tokens
        for t in set(s1_name_toks):
            matched = self.token_idx.get((c, t), [])
            if matched:
                freq = self.token_counts.get((c, t), 1)
                w = 2.5 / (1.0 + (freq ** 0.5) * 0.1)
                for idx in matched:
                    scores[idx] += w

        # 4. Address anchor keys
        for ak in set(s1_addr_keys):
            matched = self.addr_key_idx.get((c, ak), [])
            if matched:
                w = 2.0 if "_" in ak else 0.8
                for idx in matched:
                    scores[idx] += w

        if not scores:
            return []

        sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_cands[:max_cands]]


def run_fast_inference(
    threshold: float = TUNED_THRESHOLD,
    margin: float = TUNED_MARGIN,
    batch_size: int = 50000
):
    print("=======================================================")
    print("      FAST RANK-1 TEST INFERENCE PIPELINE")
    print(f"      Threshold={threshold:.2f}, Margin={margin:.2f}")
    print("=======================================================")

    # 1. Load trained model
    model_path = MODEL_DIR / "matching_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found!")
    print(f"Loading trained model from {model_path}...")
    model = EntityMatchingModel()
    model.load(model_path)

    # 2. Build Fast Target Index
    print("\n[Stage 1/3] Loading & Indexing Test Target Tables (Source 2 and Source 3)...")
    t_start = time.time()
    target_index = FastTargetIndex(max_token_freq=3000, max_addr_freq=800)

    for src_path in [TEST_S2, TEST_S3]:
        data = load_and_normalize_table(src_path)
        target_index.add_table_data(*data)
        del data  # free memory

    target_index.prune()
    print(f"Target index built with {len(target_index.eids):,} total records in {time.time()-t_start:.2f}s")

    # 3. Stream Test Source 1 & Generate Matches
    print("\n[Stage 2/3] Streaming Test Source 1 & Matching...")
    MATCHING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    f_match = open(MATCHING_OUTPUT, "w", encoding="utf-8", newline="")
    f_cand = open(CANDIDATE_OUTPUT, "w", encoding="utf-8", newline="")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # Load S1 data
    s1_eids, s1_norm_names, s1_spaceless, s1_name_tokens, s1_norm_addrs, s1_addr_keys, s1_addr_nums, s1_has_addr, s1_countries = load_and_normalize_table(TEST_S1)
    total_s1 = len(s1_eids)
    print(f"\nMatching {total_s1:,} Source 1 entities against {len(target_index.eids):,} target entities...")

    t0 = time.time()
    processed = 0
    total_matches_found = 0
    total_singletons = 0

    # Process in batches
    for b_start in range(0, total_s1, batch_size):
        b_end = min(b_start + batch_size, total_s1)
        
        batch_pairs_s1_idx = []
        batch_pairs_t_idx = []
        batch_feats = []
        s1_to_cand_eids = {}

        for i in range(b_start, b_end):
            s1_eid = s1_eids[i]
            c = s1_countries[i]
            nn = s1_norm_names[i]
            sp = s1_spaceless[i]
            toks = s1_name_tokens[i]
            aks = s1_addr_keys[i]
            na = s1_norm_addrs[i]
            an = s1_addr_nums[i]
            ha = s1_has_addr[i]

            cand_indices = target_index.get_candidates(c, nn, sp, toks, aks, max_cands=MAX_CANDIDATES)
            cand_eids = [target_index.eids[idx] for idx in cand_indices]
            s1_to_cand_eids[i] = cand_eids

            # Write candidates immediately
            f_cand.write(f"{s1_eid}\t{','.join(cand_eids)}\n")

            if not cand_indices:
                continue

            for t_idx in cand_indices:
                feats = compute_fast_features(
                    nn, sp, toks, na, aks, an, c, ha,
                    target_index.eids[t_idx],
                    target_index.norm_names[t_idx],
                    target_index.spaceless[t_idx],
                    target_index.name_tokens[t_idx],
                    target_index.norm_addrs[t_idx],
                    target_index.addr_keys[t_idx],
                    target_index.addr_nums[t_idx],
                    target_index.countries[t_idx],
                    target_index.has_addr[t_idx]
                )
                batch_feats.append(feats)
                batch_pairs_s1_idx.append(i)
                batch_pairs_t_idx.append(t_idx)

        # Vectorized batch prediction
        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = model.predict_proba(X_batch)

            # Group predictions by s1_idx
            s1_cand_scores = defaultdict(list)
            for s_idx, t_idx, prob in zip(batch_pairs_s1_idx, batch_pairs_t_idx, probs):
                s1_cand_scores[s_idx].append((target_index.eids[t_idx], prob))

            for i in range(b_start, b_end):
                s1_eid = s1_eids[i]
                cands = s1_cand_scores.get(i, [])

                if not cands:
                    f_match.write(f"{s1_eid}\t\n")
                    total_singletons += 1
                    continue

                sorted_cands = sorted(cands, key=lambda x: x[1], reverse=True)
                top_eid, top_prob = sorted_cands[0]

                if top_prob < threshold:
                    # Singleton prediction (conservative to protect Precision / F0.5)
                    f_match.write(f"{s1_eid}\t\n")
                    total_singletons += 1
                else:
                    selected = [
                        eid for eid, prob in sorted_cands
                        if prob >= threshold and (top_prob - prob) <= margin
                    ]
                    f_match.write(f"{s1_eid}\t{','.join(selected)}\n")
                    total_matches_found += len(selected)
        else:
            for i in range(b_start, b_end):
                f_match.write(f"{s1_eids[i]}\t\n")
                total_singletons += 1

        processed += (b_end - b_start)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 1
        eta_min = (total_s1 - processed) / rate / 60
        print(f"  Processed {processed:,}/{total_s1:,} S1 entities ({processed/total_s1*100:.1f}%) in {elapsed:.1f}s [{rate:.0f} recs/s, ETA: {eta_min:.1f} min] (Matches: {total_matches_found:,}, Singletons: {total_singletons:,})")

    f_match.close()
    f_cand.close()

    total_time = time.time() - t0
    print(f"\n[Stage 2/3 Complete] Matched all {total_s1:,} entities in {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Total matches predicted: {total_matches_found:,}")
    print(f"  Total singletons predicted: {total_singletons:,} ({total_singletons/total_s1*100:.2f}%)")
    print(f"  Matching output: {MATCHING_OUTPUT}")
    print(f"  Candidate output: {CANDIDATE_OUTPUT}")

    # 4. Validate output files
    print("\n[Stage 3/3] Validating Submission Files with Competition Script...")
    utils_dir = ROOT_DIR / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    import validate_submission
    errors, warnings = validate_submission.validate(
        matching_path=str(MATCHING_OUTPUT),
        candidate_path=str(CANDIDATE_OUTPUT),
        test_dir=str(ROOT_DIR / "dataset" / "test"),
        check_ids=False
    )
    if not errors:
        print("\n" + "="*65)
        print(">>> SUCCESS: Files are 100% Validated and Safe to Submit! <<<")
        print("=============================================================")
    else:
        print(f"\n>>> ERROR: Validation found {len(errors)} issues: {errors}")


if __name__ == "__main__":
    run_fast_inference()
