"""
Evaluation and Metrics module for Business Entity Resolution.
Computes Macro F0.5, Precision, Recall, Singleton Accuracy, and Candidate Recall.
"""

from typing import Dict, List, Set, Tuple


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F0.5 for a single Source 1 entity.
    """
    # Case 1: True singleton (no ground truth matches)
    if not true_matches:
        if not pred_matches:
            # Correctly predicted singleton
            return 1.0, 1.0, 1.0
        else:
            # False merge on singleton
            return 0.0, 0.0, 0.0

    # Case 2: True non-singleton with empty prediction
    if not pred_matches:
        return 0.0, 0.0, 0.0

    # Case 3: Both true and predicted have entities
    tp = len(true_matches & pred_matches)
    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    denom = 0.25 * precision + recall
    if denom == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return precision, recall, f05


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Compute macro-averaged F0.5, precision, recall, and singleton performance across all S1 entities.
    """
    all_s1_ids = set(ground_truth.keys()) | set(predictions.keys())
    if not all_s1_ids:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "non_singleton_f05": 0.0,
            "total_entities": 0
        }

    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0

    singleton_count = 0
    singleton_correct = 0

    non_singleton_count = 0
    non_singleton_f05_total = 0.0

    for s1_id in all_s1_ids:
        gt_set = ground_truth.get(s1_id, set())
        pred_set = set(predictions.get(s1_id, []))

        prec, rec, f05 = compute_entity_f05(gt_set, pred_set)
        total_prec += prec
        total_rec += rec
        total_f05 += f05

        if not gt_set:
            singleton_count += 1
            if not pred_set:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            non_singleton_f05_total += f05

    n = len(all_s1_ids)
    return {
        "macro_f05": total_f05 / n,
        "macro_precision": total_prec / n,
        "macro_recall": total_rec / n,
        "singleton_accuracy": (singleton_correct / singleton_count) if singleton_count > 0 else 0.0,
        "non_singleton_f05": (non_singleton_f05_total / non_singleton_count) if non_singleton_count > 0 else 0.0,
        "total_entities": n,
        "singleton_count": singleton_count,
        "non_singleton_count": non_singleton_count
    }


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Measure blocking candidate recall (maximum reachable recall ceiling).
    """
    total_true_matches = 0
    retrieved_true_matches = 0

    for s1_id, gt_set in ground_truth.items():
        if not gt_set:
            continue
        total_true_matches += len(gt_set)
        cand_set = set(candidates.get(s1_id, []))
        retrieved_true_matches += len(gt_set & cand_set)

    candidate_recall = (retrieved_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    return {
        "blocking_recall": candidate_recall,
        "total_true_links": total_true_matches,
        "retrieved_links": retrieved_true_matches
    }
