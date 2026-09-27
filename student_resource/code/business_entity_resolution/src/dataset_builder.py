"""
Dataset builder for training pair generation with hard negative sampling.
Includes singleton-aware negatives for better precision calibration.
"""

import random
from typing import Dict, List, Set, Tuple
import numpy as np
from .data_loader import EntityRecord
from .feature_extraction import compute_pair_features, FEATURE_NAMES


def build_training_dataset(
    s1_records: Dict[str, EntityRecord],
    target_records: Dict[str, EntityRecord],
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, List[str]],
    neg_to_pos_ratio: int = 4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Construct training feature matrix X, binary labels y, and group array for GroupKFold.

    Two types of negatives:
    1. Hard in-block negatives: candidates that look similar but are NOT true matches.
    2. Singleton negatives: for S1 entities with NO true matches, their top candidates
       are labeled 0 — teaches the model to confidently predict singletons.

    Returns:
      - X: feature matrix (N_samples, N_features)
      - y: binary labels (N_samples,)
      - groups: s1_id group identifiers (N_samples,)
      - pair_keys: list of (s1_id, target_id)
    """
    random.seed(seed)
    np.random.seed(seed)

    X_list = []
    y_list = []
    groups_list = []
    pair_keys = []

    for s1_id, s1_rec in s1_records.items():
        true_set = ground_truth.get(s1_id, set())
        cand_list = candidates.get(s1_id, [])

        # --- Positives (true matches in ground truth) ---
        pos_added = 0
        for tid in true_set:
            t_rec = target_records.get(tid)
            if t_rec:
                feats = compute_pair_features(s1_rec, t_rec)
                X_list.append(feats)
                y_list.append(1)
                groups_list.append(s1_id)
                pair_keys.append((s1_id, tid))
                pos_added += 1

        # --- Hard In-Block Negatives ---
        neg_candidates = [tid for tid in cand_list if tid not in true_set and tid in target_records]
        if pos_added > 0 and neg_candidates:
            # Standard hard negatives: similar-looking but not matching
            num_negs = min(len(neg_candidates), max(pos_added * neg_to_pos_ratio, 3))
            sampled_negs = random.sample(neg_candidates, num_negs)
            for tid in sampled_negs:
                t_rec = target_records.get(tid)
                if t_rec:
                    feats = compute_pair_features(s1_rec, t_rec)
                    X_list.append(feats)
                    y_list.append(0)
                    groups_list.append(s1_id)
                    pair_keys.append((s1_id, tid))

        elif pos_added == 0 and cand_list:
            # Singleton S1: no true matches exist — teach model to predict 0 for these
            # Take top-3 candidates (highest blocking score = hardest negatives for singletons)
            num_singleton_negs = min(len(neg_candidates), 3)
            if num_singleton_negs > 0:
                sampled_singleton_negs = neg_candidates[:num_singleton_negs]
                for tid in sampled_singleton_negs:
                    t_rec = target_records.get(tid)
                    if t_rec:
                        feats = compute_pair_features(s1_rec, t_rec)
                        X_list.append(feats)
                        y_list.append(0)
                        groups_list.append(s1_id)
                        pair_keys.append((s1_id, tid))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list)

    return X, y, groups, pair_keys
