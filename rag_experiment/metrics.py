"""Classification and retrieval metrics (paper: Acc + Balanced Acc)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from rag_experiment.labels import PAPER_LABELS, PAPER_LABELS_EXT


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict[str, Any]:
    n = len(y_pred)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "parse_rate": 0.0,
            "per_class_support": {},
            "label_space": PAPER_LABELS,
        }
    parse_ok = sum(1 for yp in y_pred if yp in PAPER_LABELS)
    support: Dict[str, int] = {}
    for yt in y_true:
        support[yt] = support.get(yt, 0) + 1
    yt_list = list(y_true)
    yp_list = list(y_pred)
    return {
        "n": n,
        "accuracy": float(accuracy_score(yt_list, yp_list)),
        "balanced_accuracy": float(
            balanced_accuracy_score(yt_list, yp_list)
        ),
        "f1_macro": float(
            f1_score(yt_list, yp_list, labels=PAPER_LABELS_EXT, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(
                yt_list, yp_list, labels=PAPER_LABELS_EXT, average="weighted", zero_division=0
            )
        ),
        "parse_rate": parse_ok / n,
        "per_class_support": support,
        "label_space": PAPER_LABELS,
    }


def compute_retrieval_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "empty_rate": 0.0,
            "mean_top1_score": None,
            "mean_context_chars": 0.0,
            "mean_hits_used": 0.0,
        }
    empty = 0
    scores: List[float] = []
    ctx_lens: List[int] = []
    hits_used: List[int] = []
    for r in records:
        ret = r.get("retrieval") or {}
        if ret.get("empty", True):
            empty += 1
        s = ret.get("top1_score")
        if s is not None:
            scores.append(float(s))
        ctx_lens.append(int(ret.get("context_chars", 0)))
        hits_used.append(int(ret.get("n_hits", 0)))
    return {
        "n": n,
        "empty_rate": empty / n,
        "mean_top1_score": (sum(scores) / len(scores)) if scores else None,
        "mean_context_chars": sum(ctx_lens) / n,
        "mean_hits_used": sum(hits_used) / n,
    }


def delta_vs_baseline(
    summary: Dict[str, Dict[str, Any]],
    baseline_key: str = "R0",
) -> Dict[str, Any]:
    base = summary.get(baseline_key)
    if not base:
        return {"baseline": baseline_key, "note": "baseline missing", "deltas": {}}
    base_acc = float(base.get("accuracy", 0.0))
    base_ba = float(base.get("balanced_accuracy", 0.0))
    base_f1 = float(base.get("f1_macro", 0.0))
    deltas: Dict[str, Any] = {}
    for k, m in summary.items():
        if k == baseline_key:
            continue
        deltas[k] = {
            "accuracy_delta": float(m.get("accuracy", 0.0)) - base_acc,
            "balanced_accuracy_delta": float(m.get("balanced_accuracy", 0.0)) - base_ba,
            "f1_macro_delta": float(m.get("f1_macro", 0.0)) - base_f1,
            "accuracy": float(m.get("accuracy", 0.0)),
            "balanced_accuracy": float(m.get("balanced_accuracy", 0.0)),
            "f1_macro": float(m.get("f1_macro", 0.0)),
        }
    return {
        "baseline": baseline_key,
        "baseline_accuracy": base_acc,
        "baseline_balanced_accuracy": base_ba,
        "deltas": deltas,
    }


def pick_best_line(summary: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not summary:
        return None
    # Prefer balanced accuracy (paper primary under class imbalance)
    return max(
        summary.keys(),
        key=lambda k: (
            float(summary[k].get("balanced_accuracy", 0.0)),
            float(summary[k].get("accuracy", 0.0)),
        ),
    )
