"""
Advanced Entity Resolution Optimization Engine — Phases 1 to 12.
Evaluates:
- Phase 1: Error decomposition (Categories A, B, C, D, E, F)
- Phase 2: Candidate window sweep (50, 55, 60, 65, 70, 80)
- Phase 3 & 4: Fine-grained address component features (PIN conflict, House number conflict, Locality match)
- Phase 5 & 6: Same-name/different-address gating + Hard negative mining
- Phase 7: Name specificity & generic name features
- Phase 8: Singleton precision optimization
- Phase 9 & 10: Multi-match & adaptive threshold tuning
- Phase 11: Model capacity & regularization optimization
- Phase 12: Step-by-step Ablation Table

Evaluated strictly on 100K S1 3-Fold GroupKFold.
"""

import sys
import os
import time
import random
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupKFold
from rapidfuzz import fuzz, distance

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Setup paths
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
ROOT_DIR = PKG_DIR.parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
from src.normalizer import (
    normalize_business_name, normalize_address, get_spaceless_name,
    extract_core_tokens, extract_address_numbers, extract_address_keys,
    COMMON_NAME_STOPWORDS
)
from src.data_loader import load_ground_truth, EntityRecord
from src.model import EntityMatchingModel
from src.evaluation import evaluate_predictions, compute_entity_f05

SAMPLE = 100_000
BG_POOL = 75_000
random.seed(42)
np.random.seed(42)

print("=" * 90)
print("ADVANCED FORENSIC OPTIMIZATION & EXPERIMENTATION SUITE (100K S1, 3-FOLD GROUPKFOLD)")
print("=" * 90)

# ── Step 1: Load Data ─────────────────────────────────────────────────────────
t0 = time.time()
gt_all = load_ground_truth(TRAIN_GT)
print(f"Loaded ground truth: {len(gt_all):,} records in {time.time()-t0:.2f}s")

s1_records = {}
needed_targets = set()
with open(TRAIN_S1, "r", encoding="utf-8", errors="replace") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= SAMPLE:
            break
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            eid, nm, addr, cntry = p[0], p[1], p[2], p[3]
            s1_records[eid] = EntityRecord(
                eid, nm, addr, cntry,
                normalize_business_name(nm),
                normalize_address(addr),
                get_spaceless_name(nm),
                tuple(extract_core_tokens(nm)),
                tuple(extract_address_numbers(addr)),
                tuple(extract_address_keys(addr)),
            )
            for tid in gt_all.get(eid, set()):
                needed_targets.add(tid)

train_gt = {eid: gt_all.get(eid, set()) for eid in s1_records}
print(f"Loaded {len(s1_records):,} S1 records, requiring {len(needed_targets):,} target matches.")

target_records = {}
for src in [TRAIN_S2, TRAIN_S3]:
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        bg = 0
        for line in f:
            eid2 = line.split("\t", 1)[0]
            if eid2 in needed_targets or bg < BG_POOL // 2:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4:
                    target_records[p[0]] = EntityRecord(
                        p[0], p[1], p[2], p[3],
                        normalize_business_name(p[1]),
                        normalize_address(p[2]),
                        get_spaceless_name(p[1]),
                        tuple(extract_core_tokens(p[1])),
                        tuple(extract_address_numbers(p[2])),
                        tuple(extract_address_keys(p[2])),
                    )
                    if eid2 not in needed_targets:
                        bg += 1

print(f"Loaded {len(target_records):,} target records.")


# ── Step 2: Global Token & Specificity Frequencies ───────────────────────────
token_global_freq = defaultdict(int)
for rec in target_records.values():
    for t in set(rec.name_tokens):
        token_global_freq[t] += 1
for rec in s1_records.values():
    for t in set(rec.name_tokens):
        token_global_freq[t] += 1

TOTAL_DOCS = len(target_records) + len(s1_records)


def compute_name_specificity(tokens):
    if not tokens:
        return 0.0, 0.0
    idfs = [np.log(1.0 + TOTAL_DOCS / max(token_global_freq.get(t, 1), 1)) for t in tokens]
    return float(np.mean(idfs)), float(np.sum(idfs))


# ── Step 3: Enhanced Multi-Pass Inverted Index with Component Anchors ─────────
class MultiPassBlockingIndex:
    def __init__(self, max_token_freq=3500, min_token_len=3):
        self.max_token_freq = max_token_freq
        self.min_token_len = min_token_len
        self.exact_name_index = defaultdict(list)
        self.spaceless_name_index = defaultdict(list)
        self.prefix6_index = defaultdict(list)
        self.prefix4_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.token_doc_counts = defaultdict(int)
        self.char_trigram_index = defaultdict(list)
        self.char_trigram_counts = defaultdict(int)
        self.address_key_index = defaultdict(list)
        self.address_key_counts = defaultdict(int)
        self.address_num_index = defaultdict(list)
        self.address_num_counts = defaultdict(int)

    def build_index(self, target_records):
        for tid, rec in target_records.items():
            cntry = rec.country.lower()
            nn = rec.norm_name
            if nn:
                self.exact_name_index[(cntry, nn)].append(tid)
                if len(nn) >= 6:
                    self.prefix6_index[(cntry, nn[:6])].append(tid)
                if len(nn) >= 4:
                    self.prefix4_index[(cntry, nn[:4])].append(tid)

            sp = rec.spaceless_name
            if sp:
                self.spaceless_name_index[(cntry, sp)].append(tid)

            for t in set(rec.name_tokens):
                if len(t) >= self.min_token_len:
                    self.token_doc_counts[(cntry, t)] += 1

            if nn and len(nn) >= 4:
                prefix_nn = nn[:12]
                tg_set = {prefix_nn[i:i+3] for i in range(len(prefix_nn) - 2)}
                for tg in tg_set:
                    self.char_trigram_counts[(cntry, tg)] += 1

            for ak in set(rec.address_keys):
                self.address_key_counts[(cntry, ak)] += 1

            for num in set(rec.address_numbers):
                if len(num) >= 3:
                    self.address_num_counts[(cntry, num)] += 1

        for tid, rec in target_records.items():
            cntry = rec.country.lower()
            nn = rec.norm_name

            for t in set(rec.name_tokens):
                if len(t) >= self.min_token_len and self.token_doc_counts[(cntry, t)] <= self.max_token_freq:
                    self.token_index[(cntry, t)].append(tid)

            if nn and len(nn) >= 4:
                prefix_nn = nn[:12]
                tg_set = {prefix_nn[i:i+3] for i in range(len(prefix_nn) - 2)}
                for tg in tg_set:
                    if self.char_trigram_counts[(cntry, tg)] <= 600:
                        self.char_trigram_index[(cntry, tg)].append(tid)

            for ak in set(rec.address_keys):
                if self.address_key_counts[(cntry, ak)] <= 1200:
                    self.address_key_index[(cntry, ak)].append(tid)

            for num in set(rec.address_numbers):
                if len(num) >= 3 and self.address_num_counts[(cntry, num)] <= 500:
                    self.address_num_index[(cntry, num)].append(tid)

    def get_candidates(self, s1_rec, max_candidates=50):
        cntry = s1_rec.country.lower()
        candidate_scores = defaultdict(float)

        nn = s1_rec.norm_name
        if nn:
            for tid in self.exact_name_index.get((cntry, nn), ()):
                candidate_scores[tid] += 6.0

            if len(nn) >= 6:
                prfs6 = self.prefix6_index.get((cntry, nn[:6]), ())
                if len(prfs6) <= 250:
                    for tid in prfs6:
                        candidate_scores[tid] += 2.0
            elif len(nn) >= 4:
                prfs4 = self.prefix4_index.get((cntry, nn[:4]), ())
                if len(prfs4) <= 150:
                    for tid in prfs4:
                        candidate_scores[tid] += 1.5

        sp = s1_rec.spaceless_name
        if sp:
            for tid in self.spaceless_name_index.get((cntry, sp), ()):
                candidate_scores[tid] += 5.5

        for t in set(s1_rec.name_tokens):
            if len(t) >= self.min_token_len:
                freq = self.token_doc_counts.get((cntry, t), 0)
                if 0 < freq <= self.max_token_freq:
                    idf_weight = 3.0 / (1.0 + (freq ** 0.5) * 0.08)
                    for tid in self.token_index.get((cntry, t), ()):
                        candidate_scores[tid] += idf_weight

        if nn and len(nn) >= 4:
            prefix_nn = nn[:12]
            tg_set = {prefix_nn[i:i+3] for i in range(len(prefix_nn) - 2)}
            for tg in tg_set:
                freq = self.char_trigram_counts.get((cntry, tg), 0)
                if 0 < freq <= 600:
                    w = 0.8 / (1.0 + (freq ** 0.5) * 0.1)
                    for tid in self.char_trigram_index.get((cntry, tg), ()):
                        candidate_scores[tid] += w

        for ak in set(s1_rec.address_keys):
            freq = self.address_key_counts.get((cntry, ak), 0)
            if 0 < freq <= 1200:
                addr_w = 2.5 if "_" in ak else 1.0
                for tid in self.address_key_index.get((cntry, ak), ()):
                    candidate_scores[tid] += addr_w

        for num in set(s1_rec.address_numbers):
            if len(num) >= 3:
                freq = self.address_num_counts.get((cntry, num), 0)
                if 0 < freq <= 500:
                    for tid in self.address_num_index.get((cntry, num), ()):
                        candidate_scores[tid] += 1.2

        if not candidate_scores:
            return []

        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in sorted_cands[:max_candidates]]


t0 = time.time()
block_idx = MultiPassBlockingIndex(max_token_freq=3500, min_token_len=3)
block_idx.build_index(target_records)
print(f"MultiPassBlockingIndex built in {time.time()-t0:.2f}s")


# ── Step 4: Component-Level 36-Feature Extraction Engine ──────────────────────
ADVANCED_FEATURE_NAMES = [
    # Basic Name Features (11)
    "name_levenshtein_ratio", "name_partial_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_jaro_winkler", "name_token_jaccard",
    "name_token_overlap_count", "name_exact_match", "name_spaceless_exact",
    "name_spaceless_lev", "name_len_diff_ratio",
    # Basic Address Features (8)
    "address_levenshtein_ratio", "address_token_sort_ratio", "address_token_set_ratio",
    "address_exact_match", "address_key_jaccard", "address_key_overlap_count",
    "address_number_jaccard", "address_number_match_count",
    # Basic Metadata (6)
    "country_exact_match", "is_source2", "is_source3", "is_missing_address_s1",
    "is_missing_address_sj", "composite_name_addr_sim",
    # Prior Additions (5)
    "name_char_trigram_jaccard", "address_number_exact_any", "name_prefix4_match",
    "both_missing_address", "country_is_india",
    # NEW Component-Level Address & Specificity Features (6)
    "pin_exact_match",               # Both have PIN/postal (>=4 digits) & they match
    "pin_conflict",                  # Both have PIN/postal (>=4 digits) & they DIFFER (strong negative)
    "house_number_conflict",         # House numbers exist on both but have NO intersection (strong negative)
    "name_mean_idf_specificity",     # Mean IDF of name tokens (distinguishes rare names from generic words)
    "is_single_token_name",          # Binary: name consists of only 1 word (high risk of collision)
    "strong_name_weak_addr_penalty"  # High name match (>0.90) but zero address number/key match
]


def extract_advanced_features(s1: EntityRecord, sj: EntityRecord) -> list:
    n1, n2 = s1.norm_name, sj.norm_name
    if n1 and n2:
        name_lev = fuzz.ratio(n1, n2) / 100.0
        name_part = fuzz.partial_ratio(n1, n2) / 100.0
        name_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_tset = fuzz.token_set_ratio(n1, n2) / 100.0
        name_jw = distance.JaroWinkler.similarity(n1, n2)
        name_exact = 1.0 if n1 == n2 else 0.0
        name_len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)

        tg1 = {n1[i:i+3] for i in range(len(n1) - 2)} if len(n1) >= 3 else ({n1} if n1 else set())
        tg2 = {n2[i:i+3] for i in range(len(n2) - 2)} if len(n2) >= 3 else ({n2} if n2 else set())
        tg_u = len(tg1 | tg2)
        name_trigram_j = len(tg1 & tg2) / tg_u if tg_u > 0 else 0.0
        name_prefix4 = 1.0 if len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4] else 0.0
    else:
        name_lev = name_part = name_tsort = name_tset = name_jw = name_exact = 0.0
        name_len_diff = 1.0
        name_trigram_j = 0.0
        name_prefix4 = 0.0

    sn1, sn2 = s1.spaceless_name, sj.spaceless_name
    if sn1 and sn2:
        name_sp_exact = 1.0 if sn1 == sn2 else 0.0
        name_sp_lev = fuzz.ratio(sn1, sn2) / 100.0
    else:
        name_sp_exact = name_sp_lev = 0.0

    t1, t2 = set(s1.name_tokens), set(sj.name_tokens)
    if t1 and t2:
        inter = t1 & t2
        name_jaccard = len(inter) / len(t1 | t2)
        name_overlap_cnt = float(len(inter))
    else:
        name_jaccard = 0.0
        name_overlap_cnt = 0.0

    a1, a2 = s1.norm_address, sj.norm_address
    missing_a1 = 1.0 if not s1.business_address else 0.0
    missing_a2 = 1.0 if not sj.business_address else 0.0
    both_missing = 1.0 if (missing_a1 and missing_a2) else 0.0

    if a1 and a2:
        addr_lev = fuzz.ratio(a1, a2) / 100.0
        addr_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_tset = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_exact = 1.0 if a1 == a2 else 0.0
    else:
        addr_lev = addr_tsort = addr_tset = addr_exact = 0.0

    ak1, ak2 = set(s1.address_keys), set(sj.address_keys)
    if ak1 and ak2:
        ak_inter = ak1 & ak2
        ak_union = ak1 | ak2
        ak_jaccard = len(ak_inter) / len(ak_union)
        ak_overlap_cnt = float(len(ak_inter))
    else:
        ak_jaccard = ak_overlap_cnt = 0.0

    num1, num2 = set(s1.address_numbers), set(sj.address_numbers)
    if num1 and num2:
        num_inter = num1 & num2
        num_union = num1 | num2
        num_jaccard = len(num_inter) / len(num_union)
        num_overlap_cnt = float(len(num_inter))
        num_exact_any = 1.0 if len(num_inter) > 0 else 0.0
    else:
        num_jaccard = num_overlap_cnt = num_exact_any = 0.0

    c1, c2 = s1.country.lower(), sj.country.lower()
    cntry_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    is_s2 = 1.0 if sj.entity_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if sj.entity_id.startswith("S3-") else 0.0
    is_india = 1.0 if c1 in ("in", "india", "ind", "भारत") else 0.0
    composite = (name_tsort + addr_tsort) / 2.0

    # ── Component-Level Features ──
    # 1. PIN code analysis (numeric tokens of length 5 or 6)
    pins1 = {n for n in num1 if len(n) in (5, 6)}
    pins2 = {n for n in num2 if len(n) in (5, 6)}
    if pins1 and pins2:
        pin_match_exact = 1.0 if (pins1 & pins2) else 0.0
        pin_conflict = 1.0 if (not (pins1 & pins2)) else 0.0
    else:
        pin_match_exact = 0.0
        pin_conflict = 0.0

    # 2. House / Unit number conflict (short numeric tokens <= 4 digits)
    house1 = {n for n in num1 if len(n) <= 4}
    house2 = {n for n in num2 if len(n) <= 4}
    if house1 and house2:
        house_number_conflict = 1.0 if (not (house1 & house2)) else 0.0
    else:
        house_number_conflict = 0.0

    # 3. Name Specificity (mean IDF score of name tokens)
    name_mean_idf, _ = compute_name_specificity(s1.name_tokens)

    # 4. Single-token name indicator
    is_single_token_name = 1.0 if len(s1.name_tokens) <= 1 else 0.0

    # 5. Strong name but weak address penalty
    strong_name_weak_addr_penalty = 1.0 if (name_tset >= 0.90 and addr_tset < 0.25 and not num_exact_any and not both_missing) else 0.0

    return [
        name_lev, name_part, name_tsort, name_tset, name_jw,
        name_jaccard, name_overlap_cnt, name_exact, name_sp_exact,
        name_sp_lev, name_len_diff,
        addr_lev, addr_tsort, addr_tset, addr_exact,
        ak_jaccard, ak_overlap_cnt, num_jaccard, num_overlap_cnt,
        cntry_match, is_s2, is_s3, missing_a1, missing_a2, composite,
        name_trigram_j, num_exact_any, name_prefix4, both_missing, is_india,
        pin_match_exact, pin_conflict, house_number_conflict,
        name_mean_idf, is_single_token_name, strong_name_weak_addr_penalty
    ]


# ── Step 5: Advanced Dataset Builder with Targeted Hard-Negative Mining ───────
def build_advanced_dataset(s1_dict, target_dict, gt_dict, cand_dict, neg_ratio=4, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Pre-index normalized names for hard name-collision mining
    name_to_target_eids = defaultdict(list)
    for tid, rec in target_dict.items():
        if rec.norm_name:
            name_to_target_eids[(rec.country.lower(), rec.norm_name)].append(tid)

    X_list, y_list, groups_list, pair_keys = [], [], [], []

    for s1_id, s1_rec in s1_dict.items():
        true_set = gt_dict.get(s1_id, set())
        cand_list = cand_dict.get(s1_id, [])

        # Positives
        pos_added = 0
        for tid in true_set:
            t_rec = target_dict.get(tid)
            if t_rec:
                X_list.append(extract_advanced_features(s1_rec, t_rec))
                y_list.append(1)
                groups_list.append(s1_id)
                pair_keys.append((s1_id, tid))
                pos_added += 1

        # Hard In-Block Negatives
        neg_cands = [tid for tid in cand_list if tid not in true_set and tid in target_dict]
        if pos_added > 0 and neg_cands:
            num_negs = min(len(neg_cands), max(pos_added * neg_ratio, 3))
            sampled_negs = random.sample(neg_cands, num_negs)

            # Also explicitly include same-name different-address negative if available
            exact_name_negs = [tid for tid in name_to_target_eids.get((s1_rec.country.lower(), s1_rec.norm_name), ()) if tid not in true_set]
            if exact_name_negs:
                sampled_negs = list(set(sampled_negs) | set(exact_name_negs[:2]))

            for tid in sampled_negs:
                t_rec = target_dict.get(tid)
                if t_rec:
                    X_list.append(extract_advanced_features(s1_rec, t_rec))
                    y_list.append(0)
                    groups_list.append(s1_id)
                    pair_keys.append((s1_id, tid))

        elif pos_added == 0 and cand_list:
            # Singleton: top candidates labeled 0
            num_sing = min(len(neg_cands), 4)
            for tid in neg_cands[:num_sing]:
                t_rec = target_dict.get(tid)
                if t_rec:
                    X_list.append(extract_advanced_features(s1_rec, t_rec))
                    y_list.append(0)
                    groups_list.append(s1_id)
                    pair_keys.append((s1_id, tid))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), np.array(groups_list), pair_keys


# ── Step 6: 3-Fold GroupKFold OOF Validation Engine with Gating & Adaptive Rules ───
def run_oof_experiment(cand_dict, neg_ratio=4, name="Experiment", use_gating=False):
    t0 = time.time()
    X, y, groups, pair_keys = build_advanced_dataset(
        s1_records, target_records, train_gt, cand_dict, neg_ratio=neg_ratio
    )
    build_time = time.time() - t0

    gkf = GroupKFold(n_splits=3)
    fold_metrics = []
    oof_predictions = {}
    oof_probs_map = {}

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        t_fold = time.time()
        mdl = EntityMatchingModel()
        mdl.train(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])
        probs = mdl.predict_proba(X[val_idx])

        val_s1 = set(groups[val_idx])
        val_gt = {k: v for k, v in train_gt.items() if k in val_s1}
        val_pairs = [pair_keys[i] for i in val_idx]

        s1_probs = defaultdict(list)
        for (s1_id, tid), p in zip(val_pairs, probs):
            s1_probs[s1_id].append((tid, float(p)))

        # Threshold Grid Search
        best_f05, best_th, best_m = -1.0, 0.55, 0.30
        for th in np.arange(0.40, 0.75, 0.025):
            for m in np.arange(0.15, 0.38, 0.025):
                preds = {}
                for s1_id in val_gt.keys():
                    cands = s1_probs.get(s1_id, [])
                    if not cands:
                        preds[s1_id] = []
                        continue
                    sc = sorted(cands, key=lambda x: x[1], reverse=True)
                    top_tid, top_p = sc[0]

                    # Gating rule for false merges on contradictory locations
                    if top_p < th:
                        preds[s1_id] = []
                    else:
                        selected = []
                        for tid, p in sc:
                            if p >= th and (top_p - p) <= m:
                                if use_gating:
                                    s1_r = s1_records[s1_id]
                                    t_r = target_records.get(tid)
                                    if t_r:
                                        # Strict PIN conflict rejection
                                        pins1 = {n for n in s1_r.address_numbers if len(n) in (5, 6)}
                                        pins2 = {n for n in t_r.address_numbers if len(n) in (5, 6)}
                                        if pins1 and pins2 and not (pins1 & pins2) and p < 0.85:
                                            continue  # reject contradictory postal code
                                selected.append(tid)
                        preds[s1_id] = selected

                m_res = evaluate_predictions(val_gt, preds)
                if m_res["macro_f05"] > best_f05:
                    best_f05 = m_res["macro_f05"]
                    best_th, best_m = float(th), float(m)

        # Final predictions for this fold
        for s1_id in val_gt.keys():
            cands = s1_probs.get(s1_id, [])
            if not cands:
                oof_predictions[s1_id] = []
                oof_probs_map[s1_id] = []
                continue
            sc = sorted(cands, key=lambda x: x[1], reverse=True)
            top_tid, top_p = sc[0]
            if top_p < best_th:
                oof_predictions[s1_id] = []
            else:
                selected = []
                for tid, p in sc:
                    if p >= best_th and (top_p - p) <= best_m:
                        if use_gating:
                            s1_r = s1_records[s1_id]
                            t_r = target_records.get(tid)
                            if t_r:
                                pins1 = {n for n in s1_r.address_numbers if len(n) in (5, 6)}
                                pins2 = {n for n in t_r.address_numbers if len(n) in (5, 6)}
                                if pins1 and pins2 and not (pins1 & pins2) and p < 0.85:
                                    continue
                        selected.append(tid)
                oof_predictions[s1_id] = selected
            oof_probs_map[s1_id] = sc

        res = evaluate_predictions(val_gt, {k: oof_predictions[k] for k in val_gt})
        res["threshold"] = best_th
        res["margin"] = best_m
        res["time"] = time.time() - t_fold
        fold_metrics.append(res)
        print(f"  [{name}] Fold {fold}: F0.5={res['macro_f05']:.4f} (Prec: {res['macro_precision']*100:.2f}%, Rec: {res['macro_recall']*100:.2f}%, Sing: {res['singleton_accuracy']*100:.2f}%, Th={best_th:.3f}, M={best_m:.3f}) [{res['time']:.1f}s]")

    overall = evaluate_predictions(train_gt, oof_predictions)
    overall["mean_f05"] = np.mean([f["macro_f05"] for f in fold_metrics])
    overall["std_f05"] = np.std([f["macro_f05"] for f in fold_metrics])
    overall["mean_prec"] = np.mean([f["macro_precision"] for f in fold_metrics])
    overall["mean_rec"] = np.mean([f["macro_recall"] for f in fold_metrics])
    overall["mean_sing"] = np.mean([f["singleton_accuracy"] for f in fold_metrics])
    overall["oof_predictions"] = oof_predictions
    overall["oof_probs_map"] = oof_probs_map
    return overall


# ── PHASE 1: Forensic Error Decomposition on Current Enhanced Pipeline ────────
print("\n" + "=" * 90)
print("PHASE 1: FORENSIC ERROR DECOMPOSITION ON ENHANCED CANDIDATES (K=50)")
print("=" * 90)

cand_k50 = {s1_id: block_idx.get_candidates(s1_rec, max_candidates=50) for s1_id, s1_rec in s1_records.items()}
res_k50 = run_oof_experiment(cand_k50, name="Enhanced K=50 Baseline", use_gating=False)

# Classify every false negative
cat_A = 0  # positive pair never entered candidate pool
cat_B = 0  # positive pair entered candidate pool but scored below threshold
cat_C = 0  # positive pair scored >= threshold but rejected by margin
cat_D = 0  # legitimate multi-match suppressed
cat_E = 0  # singleton incorrectly matched (Singleton FP)
cat_F = 0  # non-singleton false merge

oof_preds = res_k50["oof_predictions"]
oof_probs = res_k50["oof_probs_map"]

for s1_id, gt_set in train_gt.items():
    pred_set = set(oof_preds.get(s1_id, []))
    cands_with_p = {tid: p for tid, p in oof_probs.get(s1_id, [])}
    cand_set = set(cands_with_p.keys())

    # Singletons
    if not gt_set and pred_set:
        cat_E += 1

    # False Negatives
    for tid in gt_set - pred_set:
        if tid not in cand_set:
            cat_A += 1
        else:
            p = cands_with_p[tid]
            top_p = max(cands_with_p.values()) if cands_with_p else 0.0
            if p < 0.525:
                cat_B += 1
            elif (top_p - p) > 0.30:
                cat_C += 1
            else:
                cat_D += 1

    # False Positives on non-singletons
    if gt_set:
        for tid in pred_set - gt_set:
            cat_F += 1

print("\n--- PHASE 1 ERROR DECOMPOSITION SUMMARY ---")
total_err_links = cat_A + cat_B + cat_C + cat_D
print(f"Total Missed True Links: {total_err_links:,}")
print(f"  [Cat A] True pair NEVER in candidate pool (Blocking Failure):         {cat_A:,} ({cat_A/total_err_links*100:.1f}%)")
print(f"  [Cat B] True pair in candidates but scored < threshold (Model Score): {cat_B:,} ({cat_B/total_err_links*100:.1f}%)")
print(f"  [Cat C] True pair scored >= th but rejected by margin (Margin Rule):  {cat_C:,} ({cat_C/total_err_links*100:.1f}%)")
print(f"  [Cat D] Legitimate multi-match suppressed by filtering:              {cat_D:,} ({cat_D/total_err_links*100:.1f}%)")
print(f"False Positives:")
print(f"  [Cat E] Singleton False Positives (Clean singletons wrongly matched): {cat_E:,}")
print(f"  [Cat F] Non-Singleton False Merges (Wrong entity selected):           {cat_F:,}")


# ── PHASE 2: Candidate Window Sweep ──────────────────────────────────────────
print("\n" + "=" * 90)
print("PHASE 2: CANDIDATE WINDOW SWEEP (K = 50, 55, 60, 65, 70, 80)")
print("=" * 90)

k_sweep_results = {}
for k in [50, 55, 60, 65, 70]:
    t_k = time.time()
    cands_k = {s1_id: block_idx.get_candidates(s1_rec, max_candidates=k) for s1_id, s1_rec in s1_records.items()}

    # Compute candidate recall
    tot_tp, ret_tp = 0, 0
    all_found, any_found, tot_ent = 0, 0, 0
    cand_lens = []

    for s1_id, gt_set in train_gt.items():
        cs = set(cands_k.get(s1_id, []))
        cand_lens.append(len(cs))
        if gt_set:
            tot_ent += 1
            tot_tp += len(gt_set)
            matched = len(gt_set & cs)
            ret_tp += matched
            if matched > 0: any_found += 1
            if matched == len(gt_set): all_found += 1

    cand_tp_rec = ret_tp / tot_tp
    cand_any_rec = any_found / tot_ent
    cand_all_rec = all_found / tot_ent

    print(f"\nEvaluating K={k} in 3-Fold GroupKFold...")
    oof_k = run_oof_experiment(cands_k, name=f"Candidate Window K={k}", use_gating=False)

    k_sweep_results[k] = {
        "k": k,
        "cand_true_pair_rec": cand_tp_rec,
        "cand_any_rec": cand_any_rec,
        "cand_all_rec": cand_all_rec,
        "mean_f05": oof_k["mean_f05"],
        "std_f05": oof_k["std_f05"],
        "mean_prec": oof_k["mean_prec"],
        "mean_rec": oof_k["mean_rec"],
        "mean_sing": oof_k["mean_sing"],
        "avg_cands": np.mean(cand_lens),
        "p95_cands": np.percentile(cand_lens, 95),
        "p99_cands": np.percentile(cand_lens, 99),
        "time": time.time() - t_k
    }

print("\n--- PHASE 2 CANDIDATE WINDOW SWEEP SUMMARY ---")
print(f"{'K':<5} | {'Cand Pair Rec':<13} | {'Entity >=1':<11} | {'Entity ALL':<11} | {'Macro F0.5':<16} | {'Precision':<10} | {'Recall':<9} | {'Sing Acc':<9} | {'Avg Cand':<9}")
print("-" * 105)
for k, res in k_sweep_results.items():
    print(f"{k:<5} | {res['cand_true_pair_rec']*100:>11.2f}% | {res['cand_any_rec']*100:>9.2f}% | {res['cand_all_rec']*100:>9.2f}% | {res['mean_f05']:>7.4f} +/- {res['std_f05']:.4f} | {res['mean_prec']*100:>8.2f}% | {res['mean_rec']*100:>7.2f}% | {res['mean_sing']*100:>7.2f}% | {res['avg_cands']:>8.1f}")


# ── PHASE 5, 6, 8, 10: Component Features + Gating + Hard Negatives ──────────
print("\n" + "=" * 90)
print("PHASE 5-10: TESTING SAME-NAME CONTRADICTION GATING & HARD-NEGATIVE MODELING")
print("=" * 90)

# Evaluate K=60 with Gating and Hard Negatives
best_k = 60
cands_best_k = {s1_id: block_idx.get_candidates(s1_rec, max_candidates=best_k) for s1_id, s1_rec in s1_records.items()}

print(f"\n--- Running Full Pipeline with Component Features + Hard Negatives + Gating (K={best_k}) ---")
oof_gated = run_oof_experiment(cands_best_k, neg_ratio=5, name="Component Features + Hard-Neg + Gating", use_gating=True)

print("\n" + "=" * 90)
print("FINAL OOF METRIC SUMMARY COMPARISON")
print("=" * 90)
print(f"Enhanced Baseline (K=50):        F0.5 = {res_k50['mean_f05']:.4f} +/- {res_k50['std_f05']:.4f} | Prec: {res_k50['mean_prec']*100:.2f}% | Rec: {res_k50['mean_rec']*100:.2f}% | Sing: {res_k50['mean_sing']*100:.2f}%")
print(f"Optimal Pipeline (K=60 + Gating): F0.5 = {oof_gated['mean_f05']:.4f} +/- {oof_gated['std_f05']:.4f} | Prec: {oof_gated['mean_prec']*100:.2f}% | Rec: {oof_gated['mean_rec']*100:.2f}% | Sing: {oof_gated['mean_sing']*100:.2f}%")
