"""Clinical evaluation metrics.

All functions assume the ``bca.data`` label contract: 1 = malignant
(positive / disease present), 0 = benign. Sensitivity and specificity are
reported separately (rather than a single "recall") because in diagnostic
screening the two carry very different costs: a missed malignant case
(false negative, hurts sensitivity) is far worse than a false alarm on a
benign case (false positive, hurts specificity).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)


def confusion_counts(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def sensitivity(y_true, y_pred) -> float:
    """Recall on the malignant class: TP / (TP + FN)."""
    c = confusion_counts(y_true, y_pred)
    denom = c["tp"] + c["fn"]
    return c["tp"] / denom if denom else 0.0


def specificity(y_true, y_pred) -> float:
    """True negative rate on the benign class: TN / (TN + FP)."""
    c = confusion_counts(y_true, y_pred)
    denom = c["tn"] + c["fp"]
    return c["tn"] / denom if denom else 0.0


def ppv(y_true, y_pred) -> float:
    """Precision: of predicted-malignant, fraction actually malignant."""
    c = confusion_counts(y_true, y_pred)
    denom = c["tp"] + c["fp"]
    return c["tp"] / denom if denom else 0.0


def npv(y_true, y_pred) -> float:
    """Of predicted-benign, fraction actually benign."""
    c = confusion_counts(y_true, y_pred)
    denom = c["tn"] + c["fn"]
    return c["tn"] / denom if denom else 0.0


def roc_auc(y_true, y_score) -> float:
    return roc_auc_score(y_true, y_score)


def pr_auc(y_true, y_score) -> float:
    return average_precision_score(y_true, y_score)


def tune_threshold(y_true, y_score, min_precision: float = 0.90) -> dict:
    """Pick the decision threshold that maximizes sensitivity subject to a
    minimum precision floor.

    Framed as a cost-sensitive choice: false negatives (missed cancer) cost
    more than false positives (an unnecessary follow-up test), so among
    thresholds that keep precision acceptable we prefer the one that catches
    the most malignant cases.

    Returns a dict with the chosen threshold, its recall/precision, and
    whether the min_precision floor was actually achievable.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) + 1 precision/recall
    # points; drop the last point (threshold=inf, recall=0) which has no
    # corresponding threshold.
    precision, recall = precision[:-1], recall[:-1]

    eligible = precision >= min_precision
    if not np.any(eligible):
        # No threshold meets the floor; fall back to the one with the
        # highest achievable precision.
        best_idx = int(np.argmax(precision))
        return {
            "threshold": float(thresholds[best_idx]),
            "recall": float(recall[best_idx]),
            "precision": float(precision[best_idx]),
            "met_min_precision": False,
        }

    eligible_idx = np.where(eligible)[0]
    best_idx = eligible_idx[np.argmax(recall[eligible_idx])]
    return {
        "threshold": float(thresholds[best_idx]),
        "recall": float(recall[best_idx]),
        "precision": float(precision[best_idx]),
        "met_min_precision": True,
    }


def evaluate(fitted_pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """The single point in the codebase where a fitted model ever sees the
    held-out test set. Computes the full clinical metric suite once."""
    y_score = fitted_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float(np.mean(y_pred == y_test)),
        "sensitivity": sensitivity(y_test, y_pred),
        "specificity": specificity(y_test, y_pred),
        "ppv": ppv(y_test, y_pred),
        "npv": npv(y_test, y_pred),
        "roc_auc": roc_auc(y_test, y_score),
        "pr_auc": pr_auc(y_test, y_score),
        "confusion": confusion_counts(y_test, y_pred),
    }
