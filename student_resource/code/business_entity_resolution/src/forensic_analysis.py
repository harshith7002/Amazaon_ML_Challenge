"""
Forensic Error Analysis â€” OOF error categorization.
Runs fast 3-fold CV and saves per-error-type breakdown.
Run from student_resource/ directory.
"""
import sys, time, random
from pathlib import Path
from collections import defaultdict

SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
ROOT_DIR = PKG_DIR.parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.model_selection import GroupKFold

from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, DECISION_THRESHOLD, MARGIN_THRESHOLD
from src.normalizer import (
    normalize_business_name, normalize_address, get_spaceless_name,
    extract_core_tokens, extract_address_numbers, extract_address_keys
)
from src.data_loader import load_ground_truth, EntityRecord
from src.blocking import BlockingIndex
from src.feature_extraction import compute_pair_features, FEATURE_NAMES
from src.dataset_builder import build_training_dataset
from src.model import EntityMatchingModel
from src.evaluation import evaluate_predictions, evaluate_blocking_recall

SAMPLE = 100_000   # bigger sample for robust error stats
BG_POOL = 75_000
random.seed(42)
np.random.seed(42)

report_lines = []
def log(s=""):
    print(s, flush=True)
    report_lines.append(s)

log("=" * 70)
log("FORENSIC ANALYSIS â€” OOF Error Categorization")
log(f"Sample: {SAMPLE:,} S1 entities")
log("=" * 70)

# â”€â”€ 1. Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
t0 = time.time()
gt_all = load_ground_truth(TRAIN_GT)
log(f"GT loaded: {len(gt_all):,} entries in {time.time()-t0:.1f}s")

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
log(f"S1 loaded: {len(s1_records):,}, needed targets: {len(needed_targets):,}")

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
log(f"Targets loaded: {len(target_records):,}")

# â”€â”€ 2. Blocking analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\nâ”€â”€ Blocking Analysis â”€â”€")
idx = BlockingIndex(max_token_freq=3000, min_token_len=3)
idx.build_index(target_records)

candidates = {}
cand_sizes = []
for s1_id, s1_rec in s1_records.items():
    c = idx.get_candidates_for_s1(s1_rec, max_candidates=40)
    candidates[s1_id] = c
    cand_sizes.append(len(c))

br = evaluate_blocking_recall(train_gt, candidates)
log(f"Blocking recall: {br['blocking_recall']*100:.2f}% ({br['retrieved_links']:,}/{br['total_true_links']:,})")
log(f"Avg candidates per S1: {np.mean(cand_sizes):.1f}, Median: {np.median(cand_sizes):.0f}, Max: {max(cand_sizes)}")
log(f"Zero-candidate S1 count: {sum(1 for c in cand_sizes if c==0):,}")

# Breakdown: unreachable positives
unreachable_s1 = 0
for s1_id, gt_set in train_gt.items():
    for tid in gt_set:
        if tid not in target_records:
            unreachable_s1 += 1
log(f"Positive links not in target pool (BG truncation): {unreachable_s1:,}")

# Blocking miss analysis â€” FN at blocking stage
blocking_misses = defaultdict(int)
for s1_id, gt_set in train_gt.items():
    cand_set = set(candidates.get(s1_id, []))
    for tid in gt_set:
        if tid in target_records and tid not in cand_set:
            s1_r = s1_records[s1_id]
            t_r = target_records[tid]
            # Categorize why it was missed
            if not s1_r.norm_name or not t_r.norm_name:
                blocking_misses["empty_name"] += 1
            elif s1_r.country.lower() != t_r.country.lower():
                blocking_misses["country_mismatch"] += 1
            elif not set(s1_r.name_tokens) & set(t_r.name_tokens):
                blocking_misses["no_token_overlap"] += 1
            elif s1_r.norm_name != t_r.norm_name:
                blocking_misses["name_variation"] += 1
            else:
                blocking_misses["other"] += 1

log(f"\nBlocking miss categories (in loaded pool):")
for cat, cnt in sorted(blocking_misses.items(), key=lambda x: -x[1]):
    log(f"  {cat}: {cnt:,}")

# â”€â”€ 3. OOF predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\nâ”€â”€ Building Training Dataset for OOF â”€â”€")
X, y, groups, pair_keys = build_training_dataset(
    s1_records, target_records, train_gt, candidates, neg_to_pos_ratio=5
)
log(f"Dataset: {len(X):,} pairs, {y.sum():,} positives, {(y==0).sum():,} negatives")

log("\nâ”€â”€ 3-Fold GroupKFold OOF â”€â”€")
gkf = GroupKFold(n_splits=3)
oof_probs = np.zeros(len(y), dtype=np.float32)
oof_preds_map = {}  # s1_id -> list of (tid, prob)

fold_metrics = []
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    t_fold = time.time()
    mdl = EntityMatchingModel()
    mdl.train(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])
    probs = mdl.predict_proba(X[val_idx])
    oof_probs[val_idx] = probs

    val_s1 = set(groups[val_idx])
    val_gt = {k: v for k, v in train_gt.items() if k in val_s1}
    val_pairs = [pair_keys[i] for i in val_idx]

    s1_pp = defaultdict(list)
    for (s1_id, tid), p in zip(val_pairs, probs):
        s1_pp[s1_id].append((tid, float(p)))

    # Grid search threshold
    best_f05, best_th, best_m = -1, 0.55, 0.33
    for th in np.arange(0.40, 0.90, 0.025):
        for m in np.arange(0.05, 0.40, 0.025):
            preds = {}
            for s1_id, true_set in val_gt.items():
                cands = s1_pp.get(s1_id, [])
                if not cands:
                    preds[s1_id] = []
                    continue
                sc = sorted(cands, key=lambda x: x[1], reverse=True)
                tp_, tp_p = sc[0]
                if tp_p < th:
                    preds[s1_id] = []
                else:
                    preds[s1_id] = [t for t, p in sc if p >= th and (tp_p - p) <= m]
            from src.evaluation import evaluate_predictions
            metrics = evaluate_predictions(val_gt, preds)
            if metrics["macro_f05"] > best_f05:
                best_f05 = metrics["macro_f05"]
                best_th, best_m = float(th), float(m)

    # Final preds with best th/m
    final_preds = {}
    for s1_id, true_set in val_gt.items():
        cands = s1_pp.get(s1_id, [])
        if not cands:
            final_preds[s1_id] = []
            continue
        sc = sorted(cands, key=lambda x: x[1], reverse=True)
        tp_, tp_p = sc[0]
        if tp_p < best_th:
            final_preds[s1_id] = []
        else:
            final_preds[s1_id] = [t for t, p in sc if p >= best_th and (tp_p - p) <= best_m]
        oof_preds_map[s1_id] = (final_preds[s1_id], s1_pp.get(s1_id, []))

    eval_res = evaluate_predictions(val_gt, final_preds)
    fold_metrics.append(eval_res)
    log(f"  Fold {fold}: F0.5={eval_res['macro_f05']:.4f}, "
        f"Prec={eval_res['macro_precision']:.4f}, Rec={eval_res['macro_recall']:.4f}, "
        f"Singleton={eval_res['singleton_accuracy']:.4f}, "
        f"Threshold={best_th:.3f}, Margin={best_m:.3f} "
        f"[{time.time()-t_fold:.1f}s]")

log(f"\n  Mean OOF F0.5: {np.mean([m['macro_f05'] for m in fold_metrics]):.4f} "
    f"+/- {np.std([m['macro_f05'] for m in fold_metrics]):.4f}")

# â”€â”€ 4. Error categorization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\nâ”€â”€ Error Category Analysis â”€â”€")

false_positives = []   # predicted match, true negative (including singleton FP)
false_negatives = []   # missed true match

for s1_id, (pred_list, cand_probs) in oof_preds_map.items():
    gt_set = train_gt.get(s1_id, set())
    pred_set = set(pred_list)
    s1_r = s1_records[s1_id]

    # FPs: predicted but not in GT
    for tid in pred_set - gt_set:
        t_r = target_records.get(tid)
        if t_r:
            false_positives.append((s1_id, tid, s1_r, t_r))

    # FNs: in GT but not predicted
    for tid in gt_set - pred_set:
        if tid in target_records:
            t_r = target_records[tid]
            in_cands = tid in {t for t, p in cand_probs}
            false_negatives.append((s1_id, tid, s1_r, t_r, in_cands))

log(f"Total False Positives: {len(false_positives):,}")
log(f"Total False Negatives: {len(false_negatives):,} (of which in candidates: {sum(1 for *_, ic in false_negatives if ic):,})")

# Categorize FPs
fp_cats = defaultdict(list)
for s1_id, tid, s1_r, t_r in false_positives:
    gt_set = train_gt.get(s1_id, set())
    # Singleton FP (true singleton incorrectly matched)
    if not gt_set:
        fp_cats["singleton_FP"].append((s1_id, tid))
    else:
        # Same name, different address
        if s1_r.norm_name == t_r.norm_name and s1_r.norm_address != t_r.norm_address:
            fp_cats["same_name_diff_addr"].append((s1_id, tid))
        # Common generic name
        elif len(s1_r.name_tokens) <= 1:
            fp_cats["short_generic_name"].append((s1_id, tid))
        # Address-only collision
        elif s1_r.norm_address and t_r.norm_address and s1_r.norm_address == t_r.norm_address:
            fp_cats["same_addr_diff_name"].append((s1_id, tid))
        else:
            fp_cats["other_FP"].append((s1_id, tid))

log(f"\nFalse Positive Categories:")
for cat, items in sorted(fp_cats.items(), key=lambda x: -len(x[1])):
    log(f"  {cat}: {len(items):,}")

# Categorize FNs
fn_cats = defaultdict(list)
for s1_id, tid, s1_r, t_r, in_cands in false_negatives:
    if not in_cands:
        fn_cats["not_in_candidates_blocking_miss"].append((s1_id, tid))
    else:
        # Check what caused the model to miss it
        from rapidfuzz import fuzz
        name_sim = fuzz.token_set_ratio(s1_r.norm_name, t_r.norm_name) / 100.0
        addr_sim = fuzz.token_set_ratio(s1_r.norm_address, t_r.norm_address) / 100.0 if s1_r.norm_address and t_r.norm_address else 0.0

        if name_sim > 0.9 and addr_sim < 0.3:
            fn_cats["high_name_low_addr_FN"].append((s1_id, tid))
        elif name_sim < 0.6:
            fn_cats["low_name_sim_FN"].append((s1_id, tid))
        elif s1_r.country.lower() != t_r.country.lower():
            fn_cats["country_mismatch_FN"].append((s1_id, tid))
        elif not s1_r.norm_address and not t_r.norm_address:
            fn_cats["both_missing_addr_FN"].append((s1_id, tid))
        else:
            fn_cats["threshold_borderline_FN"].append((s1_id, tid))

log(f"\nFalse Negative Categories:")
for cat, items in sorted(fn_cats.items(), key=lambda x: -len(x[1])):
    log(f"  {cat}: {len(items):,}")

# â”€â”€ 5. Top hard negatives â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\nâ”€â”€ Top 10 Hardest False Positives (high model confidence, wrong) â”€â”€")

# Find the cand_probs for FP pairs
fp_with_score = []
for s1_id, tid, s1_r, t_r in false_positives[:]:
    for s1_id2, (pred_list, cand_probs) in oof_preds_map.items():
        if s1_id2 == s1_id:
            for t, p in cand_probs:
                if t == tid:
                    fp_with_score.append((p, s1_id, tid, s1_r, t_r))
            break

fp_with_score.sort(reverse=True)
for p, s1_id, tid, s1_r, t_r in fp_with_score[:10]:
    log(f"  Score={p:.3f} | S1: '{s1_r.business_name[:40]}' @ '{s1_r.business_address[:30]}'")
    log(f"          -> T: '{t_r.business_name[:40]}' @ '{t_r.business_address[:30]}'")

# â”€â”€ 6. Singleton analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\nâ”€â”€ Singleton Analysis â”€â”€")
sing_fps = fp_cats.get("singleton_FP", [])
# Sample 10 singleton FPs
for s1_id, tid in sing_fps[:10]:
    s1_r = s1_records[s1_id]
    t_r = target_records.get(tid)
    if t_r:
        from rapidfuzz import fuzz
        nsim = fuzz.token_set_ratio(s1_r.norm_name, t_r.norm_name)
        log(f"  Singleton FP: '{s1_r.business_name[:40]}' -> '{t_r.business_name[:40]}' (name_tset={nsim})")

# â”€â”€ 7. Multi-match analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\nâ”€â”€ Multi-Match Analysis â”€â”€")
multi_gt = {k: v for k, v in train_gt.items() if len(v) > 1 and k in oof_preds_map}
total_multi = len(multi_gt)
correct_all = sum(
    1 for s1_id, (pred_list, _) in oof_preds_map.items()
    if len(train_gt.get(s1_id, set())) > 1 and set(pred_list) == train_gt[s1_id]
)
partial = sum(
    1 for s1_id, (pred_list, _) in oof_preds_map.items()
    if len(train_gt.get(s1_id, set())) > 1 and 0 < len(set(pred_list) & train_gt[s1_id]) < len(train_gt[s1_id])
)
log(f"Multi-match entities in OOF: {total_multi:,}")
log(f"Fully correct (all matches): {correct_all:,} ({correct_all/max(total_multi,1)*100:.1f}%)")
log(f"Partial match (some correct): {partial:,} ({partial/max(total_multi,1)*100:.1f}%)")
log(f"Completely missed: {total_multi - correct_all - partial:,} ({(total_multi-correct_all-partial)/max(total_multi,1)*100:.1f}%)")

# â”€â”€ 8. Threshold variance and marginal pairs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\nâ”€â”€ Borderline Score Analysis (scores 0.45-0.65) â”€â”€")
borderline_tp, borderline_fp, borderline_fn = 0, 0, 0
for s1_id, (pred_list, cand_probs) in oof_preds_map.items():
    gt_set = train_gt.get(s1_id, set())
    for tid, p in cand_probs:
        if 0.40 <= p <= 0.70:
            is_gt = tid in gt_set
            is_pred = tid in pred_list
            if is_gt and not is_pred:
                borderline_fn += 1
            elif not is_gt and is_pred:
                borderline_fp += 1
            elif is_gt and is_pred:
                borderline_tp += 1

log(f"Borderline pairs (0.40-0.70): TP={borderline_tp:,}, FP={borderline_fp:,}, FN={borderline_fn:,}")
log(f"FP:TP ratio in borderline: {borderline_fp/max(borderline_tp,1):.3f}")

# â”€â”€ 9. Country-specific performance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
log("\nâ”€â”€ Country-Level Performance â”€â”€")
country_f05 = defaultdict(list)
for s1_id, (pred_list, _) in oof_preds_map.items():
    gt_set = train_gt.get(s1_id, set())
    s1_r = s1_records[s1_id]
    cntry = s1_r.country.lower() or "unknown"
    from src.evaluation import compute_entity_f05
    _, _, f = compute_entity_f05(gt_set, set(pred_list))
    country_f05[cntry].append(f)

for cntry, scores in sorted(country_f05.items(), key=lambda x: -len(x[1]))[:10]:
    log(f"  {cntry}: N={len(scores):,}, mean_F0.5={np.mean(scores):.4f}, min={np.min(scores):.3f}")

# â”€â”€ Save report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
report_path = ROOT_DIR / "forensic_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"\nReport saved to: {report_path}")

