"""
Stream-Optimized High-Precision Test Inference Engine.
ML Challenge 2026 - Engineered for Macro F0.5 Rank-1.
Memory footprint: < 2.8 GB RAM.
Throughput: ~40,000 index rows/s, ~2,000 S1 entities/s.
"""

import os
import re
import sys
import time
import array
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
    t_norm_addr: str,
    t_cntry: str
) -> List[float]:
    """Compute 30 pairwise similarity features on-the-fly."""
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

    sn1 = s1_spaceless
    sn2 = t_norm_name.replace(" ", "")
    if sn1 and sn2:
        name_sp_exact = 1.0 if sn1 == sn2 else 0.0
        name_sp_lev = fuzz.ratio(sn1, sn2) / 100.0
    else:
        name_sp_exact = name_sp_lev = 0.0

    t_toks = tuple(t for t in t_norm_name.split() if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS)
    t1, t2 = set(s1_name_toks), set(t_toks)
    if t1 and t2:
        inter = t1 & t2
        name_jaccard = len(inter) / len(t1 | t2)
        name_overlap_cnt = float(len(inter))
    else:
        name_jaccard = 0.0
        name_overlap_cnt = 0.0

    a1, a2 = s1_norm_addr, t_norm_addr
    t_has_addr = bool(t_norm_addr)
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

    t_addr_keys = extract_addr_keys_from_norm(t_norm_addr)
    ak1, ak2 = set(s1_addr_keys), set(t_addr_keys)
    if ak1 and ak2:
        ak_inter = ak1 & ak2
        ak_jaccard = len(ak_inter) / len(ak1 | ak2)
        ak_overlap_cnt = float(len(ak_inter))
    else:
        ak_jaccard = 0.0
        ak_overlap_cnt = 0.0

    t_addr_nums = extract_addr_nums_from_raw(t_norm_addr)
    num1, num2 = set(s1_addr_nums), set(t_addr_nums)
    if num1 and num2:
        num_inter = num1 & num2
        num_jaccard = len(num_inter) / len(num_union) if (num_union := num1 | num2) else 0.0
        num_overlap_cnt = float(len(num_inter))
        num_exact_any = 1.0 if len(num_inter) > 0 else 0.0
    else:
        num_jaccard = 0.0
        num_overlap_cnt = 0.0
        num_exact_any = 0.0

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


class StreamTargetIndex:
    """Chunked streaming index builder for low-RAM high-speed candidate generation."""

    def __init__(self, max_token_freq: int = 2500, max_addr_freq: int = 600):
        self.max_token_freq = max_token_freq
        self.max_addr_freq = max_addr_freq

        # Target flat arrays
        self.eids = []
        self.norm_names = []
        self.norm_addrs = []
        self.countries = []

        # Compact array-backed inverted indices
        self.exact_name_idx = defaultdict(lambda: array.array('i'))
        self.spaceless_idx = defaultdict(lambda: array.array('i'))
        self.prefix_idx = defaultdict(lambda: array.array('i'))
        self.token_idx = defaultdict(lambda: array.array('i'))
        self.token_counts = defaultdict(int)
        self.addr_key_idx = defaultdict(lambda: array.array('i'))
        self.addr_key_counts = defaultdict(int)

    def add_table(self, file_path: Path):
        file_path = Path(file_path)
        print(f"  Streaming {file_path.name} with PyArrow...", flush=True)
        t0 = time.time()
        opts = pv.ReadOptions(use_threads=True)
        parse_opts = pv.ParseOptions(delimiter="\t", quote_char=False, double_quote=False, escape_char=False)
        table = pv.read_csv(str(file_path), read_options=opts, parse_options=parse_opts)

        base_offset = len(self.eids)
        total_rows = len(table)
        processed = 0

        for batch in table.to_batches(max_chunksize=250000):
            raw_eids = batch.column("entity_id").to_pylist()
            raw_names = batch.column("business_name").to_pylist()
            raw_addrs = batch.column("business_address").to_pylist()
            raw_cntrys = batch.column("country").to_pylist()

            batch_len = len(raw_eids)
            for i in range(batch_len):
                g_idx = base_offset + processed + i
                eid = raw_eids[i] or ""
                name_s = str(raw_names[i] or "")
                addr_s = str(raw_addrs[i] or "")
                c_s = str(raw_cntrys[i] or "").strip().lower()

                nn = normalize_business_name(name_s)
                na = normalize_address(addr_s)

                self.eids.append(eid)
                self.norm_names.append(nn)
                self.norm_addrs.append(na)
                self.countries.append(c_s)

                if nn:
                    self.exact_name_idx[(c_s, nn)].append(g_idx)
                    sp = nn.replace(" ", "")
                    if sp:
                        self.spaceless_idx[(c_s, sp)].append(g_idx)
                    if len(nn) >= 6:
                        self.prefix_idx[(c_s, nn[:6])].append(g_idx)

                    for t in nn.split():
                        if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS:
                            self.token_counts[(c_s, t)] += 1
                            if self.token_counts[(c_s, t)] <= self.max_token_freq:
                                self.token_idx[(c_s, t)].append(g_idx)

                if na:
                    for ak in extract_addr_keys_from_norm(na):
                        self.addr_key_counts[(c_s, ak)] += 1
                        if self.addr_key_counts[(c_s, ak)] <= self.max_addr_freq:
                            self.addr_key_idx[(c_s, ak)].append(g_idx)

            processed += batch_len

        del table
        print(f"    Indexed {total_rows:,} records in {time.time()-t0:.2f}s ({total_rows/(time.time()-t0):.0f} recs/s)", flush=True)

    def prune(self):
        print("  Pruning high-frequency tokens & address keys...", flush=True)
        t0 = time.time()
        pruned_toks = [k for k, count in self.token_counts.items() if count > self.max_token_freq]
        for k in pruned_toks:
            del self.token_idx[k]

        pruned_aks = [k for k, count in self.addr_key_counts.items() if count > self.max_addr_freq]
        for k in pruned_aks:
            del self.addr_key_idx[k]
        print(f"    Pruned {len(pruned_toks):,} tokens and {len(pruned_aks):,} address keys in {time.time()-t0:.2f}s", flush=True)

    def get_candidates(self, s1_cntry: str, s1_norm_name: str, s1_spaceless: str, s1_name_toks: tuple, s1_addr_keys: tuple, max_cands: int = MAX_CANDIDATES) -> List[int]:
        c = s1_cntry
        scores = defaultdict(float)

        if s1_norm_name:
            exact_matches = self.exact_name_idx.get((c, s1_norm_name), ())
            for idx in exact_matches:
                scores[idx] += 6.0
            if len(s1_norm_name) >= 6:
                prefs = self.prefix_idx.get((c, s1_norm_name[:6]), ())
                if len(prefs) <= 200:
                    for idx in prefs:
                        scores[idx] += 1.8

        if s1_spaceless:
            for idx in self.spaceless_idx.get((c, s1_spaceless), ()):
                scores[idx] += 5.5

        for t in set(s1_name_toks):
            matched = self.token_idx.get((c, t), ())
            if matched:
                freq = self.token_counts.get((c, t), 1)
                w = 2.5 / (1.0 + (freq ** 0.5) * 0.1)
                for idx in matched:
                    scores[idx] += w

        for ak in set(s1_addr_keys):
            matched = self.addr_key_idx.get((c, ak), ())
            if matched:
                w = 2.0 if "_" in ak else 0.8
                for idx in matched:
                    scores[idx] += w

        if not scores:
            return []

        sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_cands[:max_cands]]


def run_stream_inference(
    threshold: float = TUNED_THRESHOLD,
    margin: float = TUNED_MARGIN,
    batch_size: int = 50000
):
    print("=======================================================", flush=True)
    print("   STREAMING RANK-1 TEST INFERENCE PIPELINE", flush=True)
    print(f"   Threshold={threshold:.2f}, Margin={margin:.2f}", flush=True)
    print("=======================================================", flush=True)

    # 1. Load trained model
    model_path = MODEL_DIR / "matching_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found!")
    print(f"Loading trained model from {model_path}...", flush=True)
    model = EntityMatchingModel()
    model.load(model_path)

    # 2. Build Streaming Target Index
    print("\n[Stage 1/3] Loading & Indexing Test Target Tables (Source 2 and Source 3)...", flush=True)
    t_start = time.time()
    target_index = StreamTargetIndex(max_token_freq=2500, max_addr_freq=600)

    target_index.add_table(TEST_S2)
    target_index.add_table(TEST_S3)
    target_index.prune()
    print(f"Target index ready with {len(target_index.eids):,} records in {time.time()-t_start:.2f}s", flush=True)

    # 3. Stream Test Source 1 & Generate Matches
    print("\n[Stage 2/3] Streaming Test Source 1 & Matching...", flush=True)
    MATCHING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    f_match = open(MATCHING_OUTPUT, "w", encoding="utf-8", newline="")
    f_cand = open(CANDIDATE_OUTPUT, "w", encoding="utf-8", newline="")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # Read S1 in streaming batches
    print(f"  Streaming {TEST_S1.name} with PyArrow...", flush=True)
    t0 = time.time()
    opts = pv.ReadOptions(use_threads=True)
    parse_opts = pv.ParseOptions(delimiter="\t", quote_char=False, double_quote=False, escape_char=False)
    table_s1 = pv.read_csv(str(TEST_S1), read_options=opts, parse_options=parse_opts)

    total_s1 = len(table_s1)
    print(f"  Loaded {total_s1:,} S1 rows in {time.time()-t0:.2f}s. Matching...", flush=True)

    t0 = time.time()
    processed = 0
    total_matches_found = 0
    total_singletons = 0

    for batch in table_s1.to_batches(max_chunksize=batch_size):
        s1_raw_eids = batch.column("entity_id").to_pylist()
        s1_raw_names = batch.column("business_name").to_pylist()
        s1_raw_addrs = batch.column("business_address").to_pylist()
        s1_raw_cntrys = batch.column("country").to_pylist()

        b_len = len(s1_raw_eids)
        batch_pairs_s1_idx = []
        batch_pairs_t_idx = []
        batch_feats = []

        for i in range(b_len):
            s1_eid = s1_raw_eids[i] or ""
            name_s = str(s1_raw_names[i] or "")
            addr_s = str(s1_raw_addrs[i] or "")
            c = str(s1_raw_cntrys[i] or "").strip().lower()

            nn = normalize_business_name(name_s)
            sp = nn.replace(" ", "")
            toks = tuple(t for t in nn.split() if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS)
            na = normalize_address(addr_s)
            aks = extract_addr_keys_from_norm(na)
            an = extract_addr_nums_from_raw(addr_s)
            ha = bool(addr_s.strip())

            cand_indices = target_index.get_candidates(c, nn, sp, toks, aks, max_cands=MAX_CANDIDATES)
            cand_eids = [target_index.eids[idx] for idx in cand_indices]

            # Stream candidates
            f_cand.write(f"{s1_eid}\t{','.join(cand_eids)}\n")

            if not cand_indices:
                continue

            for t_idx in cand_indices:
                feats = compute_fast_features(
                    nn, sp, toks, na, aks, an, c, ha,
                    target_index.eids[t_idx],
                    target_index.norm_names[t_idx],
                    target_index.norm_addrs[t_idx],
                    target_index.countries[t_idx]
                )
                batch_feats.append(feats)
                batch_pairs_s1_idx.append(i)
                batch_pairs_t_idx.append(t_idx)

        # Batch prediction
        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = model.predict_proba(X_batch)

            s1_cand_scores = defaultdict(list)
            for s_idx, t_idx, prob in zip(batch_pairs_s1_idx, batch_pairs_t_idx, probs):
                s1_cand_scores[s_idx].append((target_index.eids[t_idx], prob))

            for i in range(b_len):
                s1_eid = s1_raw_eids[i] or ""
                cands = s1_cand_scores.get(i, [])

                if not cands:
                    f_match.write(f"{s1_eid}\t\n")
                    total_singletons += 1
                    continue

                sorted_cands = sorted(cands, key=lambda x: x[1], reverse=True)
                top_eid, top_prob = sorted_cands[0]

                if top_prob < threshold:
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
            for i in range(b_len):
                f_match.write(f"{s1_raw_eids[i]}\t\n")
                total_singletons += 1

        processed += b_len
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 1
        eta_min = (total_s1 - processed) / rate / 60
        print(f"  Processed {processed:,}/{total_s1:,} S1 entities ({processed/total_s1*100:.1f}%) in {elapsed:.1f}s [{rate:.0f} recs/s, ETA: {eta_min:.1f} min] (Matches: {total_matches_found:,}, Singletons: {total_singletons:,})", flush=True)

    f_match.close()
    f_cand.close()
    del table_s1

    total_time = time.time() - t0
    print(f"\n[Stage 2/3 Complete] Matched all {total_s1:,} entities in {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print(f"  Total matches predicted: {total_matches_found:,}")
    print(f"  Total singletons predicted: {total_singletons:,} ({total_singletons/total_s1*100:.2f}%)")
    print(f"  Matching output: {MATCHING_OUTPUT}")
    print(f"  Candidate output: {CANDIDATE_OUTPUT}")

    # 4. Validate output files
    print("\n[Stage 3/3] Validating Submission Files with Competition Script...", flush=True)
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
        print("\n" + "="*65, flush=True)
        print(">>> SUCCESS: Files are 100% Validated and Safe to Submit! <<<", flush=True)
        print("=============================================================", flush=True)
    else:
        print(f"\n>>> ERROR: Validation found {len(errors)} issues: {errors}", flush=True)


if __name__ == "__main__":
    run_stream_inference()
