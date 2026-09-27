import argparse
import sys
import time
import os
from collections import defaultdict
from pathlib import Path

# Add project root and package root to sys.path
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
ROOT_DIR = PKG_DIR.parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List, Set, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    MATCHING_OUTPUT, CANDIDATE_OUTPUT,
    MODEL_DIR, DECISION_THRESHOLD, MARGIN_THRESHOLD, SINGLETON_THRESHOLD
)
from src.normalizer import (
    normalize_business_name,
    normalize_address,
    get_spaceless_name,
    extract_core_tokens,
    extract_address_numbers,
    extract_address_keys
)
from src.data_loader import (
    load_ground_truth,
    write_submission_file,
    EntityRecord
)
from src.blocking import BlockingIndex
from src.feature_extraction import (
    extract_features_for_candidate_pairs,
    compute_pair_features,
    FEATURE_NAMES
)
from src.dataset_builder import build_training_dataset
from src.model import EntityMatchingModel
from src.evaluation import evaluate_predictions, evaluate_blocking_recall


def load_training_sample(
    sample_size: int = 50000,
    background_pool_size: int = 50000
) -> Tuple[Dict[str, EntityRecord], Dict[str, EntityRecord], Dict[str, Set[str]]]:
    """
    Selectively load S1 training records and their ground truth matches from S2/S3.
    """
    print(f"Loading ground truth mapping...")
    gt = load_ground_truth(TRAIN_GT)

    print(f"Loading {sample_size:,} Source 1 training records...")
    s1_records = {}
    needed_target_ids = set()

    with open(TRAIN_S1, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                eid, name, addr, cntry = p[0], p[1], p[2], p[3]
                s1_records[eid] = EntityRecord(
                    entity_id=eid,
                    business_name=name,
                    business_address=addr,
                    country=cntry,
                    norm_name=normalize_business_name(name),
                    norm_address=normalize_address(addr),
                    spaceless_name=get_spaceless_name(name),
                    name_tokens=tuple(extract_core_tokens(name)),
                    address_numbers=tuple(extract_address_numbers(addr)),
                    address_keys=tuple(extract_address_keys(addr))
                )
                for tid in gt.get(eid, set()):
                    needed_target_ids.add(tid)

    # Filter GT for loaded S1 records
    train_gt = {eid: gt.get(eid, set()) for eid in s1_records}
    print(f"Loaded {len(s1_records):,} S1 records, requiring {len(needed_target_ids):,} target match records.")

    print("Loading target records (Source 2 and Source 3)...")
    target_records = {}
    for src_path in [TRAIN_S2, TRAIN_S3]:
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            next(f, None)
            bg_added = 0
            for line in f:
                eid, _, _ = line.partition("\t")
                if eid in needed_target_ids or bg_added < (background_pool_size // 2):
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 4:
                        target_records[eid] = EntityRecord(
                            entity_id=p[0],
                            business_name=p[1],
                            business_address=p[2],
                            country=p[3],
                            norm_name=normalize_business_name(p[1]),
                            norm_address=normalize_address(p[2]),
                            spaceless_name=get_spaceless_name(p[1]),
                            name_tokens=tuple(extract_core_tokens(p[1])),
                            address_numbers=tuple(extract_address_numbers(p[2])),
                            address_keys=tuple(extract_address_keys(p[2]))
                        )
                        if eid not in needed_target_ids:
                            bg_added += 1

    print(f"Loaded total of {len(target_records):,} target records.")
    return s1_records, target_records, train_gt


def tune_decision_thresholds(
    y_true_map: Dict[str, Set[str]],
    pair_keys: List[Tuple[str, str]],
    y_probs: np.ndarray,
    threshold_range: Tuple[float, float, float] = (0.45, 0.90, 0.025),
    margin_range: Tuple[float, float, float] = (0.05, 0.35, 0.025)
) -> Tuple[float, float, float]:
    """
    Grid search to find optimal (threshold, margin, singleton_threshold) maximizing macro F0.5.
    """
    s1_to_preds = defaultdict(list)
    for (s1_id, tid), prob in zip(pair_keys, y_probs):
        s1_to_preds[s1_id].append((tid, prob))

    best_f05 = -1.0
    best_params = (DECISION_THRESHOLD, MARGIN_THRESHOLD, SINGLETON_THRESHOLD)

    th_values = np.arange(threshold_range[0], threshold_range[1] + 1e-5, threshold_range[2])
    margin_values = np.arange(margin_range[0], margin_range[1] + 1e-5, margin_range[2])

    for th in th_values:
        for m in margin_values:
            preds = {}
            for s1_id, true_set in y_true_map.items():
                cand_probs = s1_to_preds.get(s1_id, [])
                if not cand_probs:
                    preds[s1_id] = []
                    continue

                sorted_cands = sorted(cand_probs, key=lambda x: x[1], reverse=True)
                top_tid, top_p = sorted_cands[0]

                if top_p < th:
                    preds[s1_id] = []
                else:
                    selected = [
                        tid for tid, p in sorted_cands
                        if p >= th and (top_p - p) <= m
                    ]
                    preds[s1_id] = selected

            metrics = evaluate_predictions(y_true_map, preds)
            if metrics["macro_f05"] > best_f05:
                best_f05 = metrics["macro_f05"]
                best_params = (float(th), float(m), float(th))

    print(f"  [Tuning] Best Validation Macro F0.5: {best_f05:.4f} with Threshold={best_params[0]:.2f}, Margin={best_params[1]:.2f}")
    return best_params


def run_pipeline(
    train_sample_size: int = 50000,
    n_splits: int = 3,
    run_inference_only: bool = False
) -> None:
    """
    Execute complete end-to-end Entity Resolution workflow.
    """
    model_path = MODEL_DIR / "matching_model.joblib"
    matching_model = EntityMatchingModel()

    best_th = DECISION_THRESHOLD
    best_m = MARGIN_THRESHOLD

    if not run_inference_only:
        print("\n=======================================================")
        print("STAGE 1: Loading Training Data & Building Ground Truth")
        print("=======================================================")
        t0 = time.time()
        s1_train, targets_train, train_gt = load_training_sample(
            sample_size=train_sample_size,
            background_pool_size=50000
        )
        print(f"Data loading completed in {time.time()-t0:.2f}s")

        print("\n=======================================================")
        print("STAGE 2: Generating Candidate Blocks (Training Set)")
        print("=======================================================")
        t0 = time.time()
        blocking_index = BlockingIndex(max_token_freq=3000, min_token_len=3)
        blocking_index.build_index(targets_train)

        candidates = {}
        for s1_id, s1_rec in s1_train.items():
            candidates[s1_id] = blocking_index.get_candidates_for_s1(s1_rec, max_candidates=40)

        recall_stats = evaluate_blocking_recall(train_gt, candidates)
        print(f"Blocking completed in {time.time()-t0:.2f}s")
        print(f"Candidate Recall Ceiling: {recall_stats['blocking_recall']*100:.2f}% ({recall_stats['retrieved_links']}/{recall_stats['total_true_links']} true matches)")

        print("\n=======================================================")
        print("STAGE 3: Feature Engineering & Dataset Construction")
        print("=======================================================")
        t0 = time.time()
        X, y, groups, pair_keys = build_training_dataset(
            s1_train, targets_train, train_gt, candidates, neg_to_pos_ratio=5
        )
        print(f"Built training matrix of {len(X):,} pairs with {X.shape[1]} features in {time.time()-t0:.2f}s")
        print(f"Class distribution: {np.sum(y == 1):,} positives, {np.sum(y == 0):,} hard negatives")

        print("\n=======================================================")
        print(f"STAGE 4: GroupKFold Cross-Validation ({n_splits} Folds)")
        print("=======================================================")
        gkf = GroupKFold(n_splits=n_splits)
        fold_f05s = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            fold_model = EntityMatchingModel()
            fold_model.train(X_tr, y_tr, X_val, y_val)
            val_probs = fold_model.predict_proba(X_val)

            val_s1_set = set(groups[val_idx])
            val_gt = {k: v for k, v in train_gt.items() if k in val_s1_set}
            val_pairs = [pair_keys[i] for i in val_idx]

            cur_th, cur_m, _ = tune_decision_thresholds(val_gt, val_pairs, val_probs)
            best_th, best_m = cur_th, cur_m

            s1_preds = defaultdict(list)
            for (s1_id, tid), p in zip(val_pairs, val_probs):
                s1_preds[s1_id].append((tid, p))

            final_preds = {}
            for s1_id in val_gt.keys():
                cand_probs = s1_preds.get(s1_id, [])
                if not cand_probs:
                    final_preds[s1_id] = []
                    continue
                sorted_c = sorted(cand_probs, key=lambda x: x[1], reverse=True)
                top_tid, top_p = sorted_c[0]
                if top_p < best_th:
                    final_preds[s1_id] = []
                else:
                    final_preds[s1_id] = [tid for tid, p in sorted_c if p >= best_th and (top_p - p) <= best_m]

            eval_res = evaluate_predictions(val_gt, final_preds)
            fold_f05s.append(eval_res["macro_f05"])
            print(f"  Fold {fold} Macro F0.5: {eval_res['macro_f05']:.4f} (Prec: {eval_res['macro_precision']:.4f}, Rec: {eval_res['macro_recall']:.4f}, Singleton Acc: {eval_res['singleton_accuracy']:.4f})")

        print(f"\nMean CV Macro F0.5: {np.mean(fold_f05s):.4f} +/- {np.std(fold_f05s):.4f}")

        print("\n=======================================================")
        print("STAGE 5: Training Full Model & Feature Importance")
        print("=======================================================")
        matching_model.train(X, y)
        matching_model.save(model_path)
        print(f"Model saved to {model_path}")
        
        importances = matching_model.get_feature_importances()
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        print("Top 10 Features by Importance:")
        for name, imp in sorted_imp[:10]:
            print(f"  - {name}: {imp:.1f}")

    else:
        print(f"Loading existing model from {model_path}...")
        matching_model.load(model_path)

    print("\n=======================================================")
    print("STAGE 6: Streaming Test Inference & File Generation")
    print("=======================================================")
    t0 = time.time()
    print("Indexing Test Target Records (Source 2 and Source 3)...")
    targets_test = {}
    test_blocking_index = BlockingIndex(max_token_freq=3000, min_token_len=3)

    for src_file in [TEST_S2, TEST_S3]:
        with open(src_file, "r", encoding="utf-8", errors="replace") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4:
                    eid, name, addr, cntry = p[0], p[1], p[2], p[3]
                    rec = EntityRecord(
                        entity_id=eid,
                        business_name=name,
                        business_address=addr,
                        country=cntry,
                        norm_name=normalize_business_name(name),
                        norm_address=normalize_address(addr),
                        spaceless_name=get_spaceless_name(name),
                        name_tokens=tuple(extract_core_tokens(name)),
                        address_numbers=tuple(extract_address_numbers(addr)),
                        address_keys=tuple(extract_address_keys(addr))
                    )
                    targets_test[eid] = rec

    print(f"Loaded {len(targets_test):,} test target records in {time.time()-t0:.2f}s")
    test_blocking_index.build_index(targets_test)
    print("Test Blocking Index built successfully.")

    # Prepare output files
    MATCHING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    f_match = open(MATCHING_OUTPUT, "w", encoding="utf-8", newline="")
    f_cand = open(CANDIDATE_OUTPUT, "w", encoding="utf-8", newline="")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    print("\nStreaming Test Source 1 & Generating Matches...")
    t0 = time.time()
    batch_size = 25000
    s1_batch = []
    total_processed = 0

    with open(TEST_S1, "r", encoding="utf-8", errors="replace") as f_s1:
        next(f_s1, None)
        for line in f_s1:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                eid, name, addr, cntry = p[0], p[1], p[2], p[3]
                s1_rec = EntityRecord(
                    entity_id=eid,
                    business_name=name,
                    business_address=addr,
                    country=cntry,
                    norm_name=normalize_business_name(name),
                    norm_address=normalize_address(addr),
                    spaceless_name=get_spaceless_name(name),
                    name_tokens=tuple(extract_core_tokens(name)),
                    address_numbers=tuple(extract_address_numbers(addr)),
                    address_keys=tuple(extract_address_keys(addr))
                )
                s1_batch.append(s1_rec)

            if len(s1_batch) >= batch_size:
                _process_and_write_batch(
                    s1_batch, test_blocking_index, targets_test, matching_model,
                    best_th, best_m, f_match, f_cand
                )
                total_processed += len(s1_batch)
                print(f"  Processed {total_processed:,} test entities...")
                s1_batch = []

        if s1_batch:
            _process_and_write_batch(
                s1_batch, test_blocking_index, targets_test, matching_model,
                best_th, best_m, f_match, f_cand
            )
            total_processed += len(s1_batch)
            print(f"  Processed {total_processed:,} test entities...")

    f_match.close()
    f_cand.close()
    print(f"\nAll {total_processed:,} test entities processed and written in {time.time()-t0:.2f}s")

    print("\n=======================================================")
    print("STAGE 7: Validation Check on Generated Submission Files")
    print("=======================================================")
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
        print("\n[SUCCESS] Pipeline successfully passed all formatting and consistency validations!")
    else:
        print(f"\n[ERROR] Validation found {len(errors)} issues: {errors}")


def _process_and_write_batch(
    s1_batch: List[EntityRecord],
    blocking_index: BlockingIndex,
    targets_test: Dict[str, EntityRecord],
    model: EntityMatchingModel,
    threshold: float,
    margin: float,
    f_match,
    f_cand
) -> None:
    """Helper to score a batch of S1 records and write results directly."""
    for s1_rec in s1_batch:
        cands = blocking_index.get_candidates_for_s1(s1_rec, max_candidates=40)
        f_cand.write(f"{s1_rec.entity_id}\t{','.join(cands)}\n")

        if not cands:
            f_match.write(f"{s1_rec.entity_id}\t\n")
            continue

        feats_list = []
        valid_cands = []
        for tid in cands:
            t_rec = targets_test.get(tid)
            if t_rec:
                feats_list.append(compute_pair_features(s1_rec, t_rec))
                valid_cands.append(tid)

        if not feats_list:
            f_match.write(f"{s1_rec.entity_id}\t\n")
            continue

        X_cand = np.array(feats_list, dtype=np.float32)
        probs = model.predict_proba(X_cand)

        paired = sorted(zip(valid_cands, probs), key=lambda x: x[1], reverse=True)
        top_tid, top_p = paired[0]

        if top_p < threshold:
            f_match.write(f"{s1_rec.entity_id}\t\n")
        else:
            selected = [tid for tid, p in paired if p >= threshold and (top_p - p) <= margin]
            f_match.write(f"{s1_rec.entity_id}\t{','.join(selected)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--sample-size", type=int, default=50000, help="Training sample size")
    parser.add_argument("--folds", type=int, default=3, help="GroupKFold CV splits")
    parser.add_argument("--inference-only", action="store_true", help="Run inference with existing model")
    args = parser.parse_args()

    run_pipeline(
        train_sample_size=args.sample_size,
        n_splits=args.folds,
        run_inference_only=args.inference_only
    )
