"""Save English-labeled figures for RAG experiment results."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from rag_experiment.labels import PAPER_LABELS_EXT


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def save_metrics_bar_chart(
    line_keys: Sequence[str],
    summary: Dict[str, Any],
    out_path: str,
) -> None:
    accs = [float(summary[k]["accuracy"]) for k in line_keys]
    bas = [float(summary[k].get("balanced_accuracy", 0.0)) for k in line_keys]
    f1m = [float(summary[k]["f1_macro"]) for k in line_keys]
    x = np.arange(len(line_keys))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(line_keys) * 1.5), 5))
    ax.bar(x - w, accs, width=w, label="Accuracy")
    ax.bar(x, bas, width=w, label="Balanced Acc.")
    ax.bar(x + w, f1m, width=w, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(list(line_keys))
    ax.set_ylabel("Score")
    ax.set_title("RAG Experiment: Metrics by Line (X-VARS 4-way)")
    ax.legend()
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_delta_accuracy_bar(
    delta_payload: Dict[str, Any],
    out_path: str,
) -> None:
    deltas = delta_payload.get("deltas") or {}
    if not deltas:
        return
    keys = list(deltas.keys())
    # Prefer balanced-accuracy delta for paper alignment
    vals = [
        float(deltas[k].get("balanced_accuracy_delta", deltas[k].get("accuracy_delta", 0.0)))
        for k in keys
    ]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.3), 4.5))
    ax.bar(keys, vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Balanced-accuracy delta vs Baseline")
    base_ba = float(delta_payload.get("baseline_balanced_accuracy", 0.0))
    ax.set_title(
        f"Balanced Acc. Gain vs {delta_payload.get('baseline', 'R0')} "
        f"(base={base_ba:.3f})"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_retrieval_empty_rate_bar(
    line_keys: Sequence[str],
    retrieval_by_line: Dict[str, Dict[str, Any]],
    out_path: str,
) -> None:
    keys = [k for k in line_keys if k in retrieval_by_line]
    if not keys:
        return
    rates = [float(retrieval_by_line[k].get("empty_rate", 0.0)) for k in keys]
    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.3), 4.5))
    ax.bar(keys, rates, color="#1f77b4")
    ax.set_ylabel("Empty retrieval rate")
    ax.set_title("RAG Retrieval Empty Rate by Line")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_png(
    trues: List[str],
    preds: List[str],
    title_suffix: str,
    out_path: str,
) -> None:
    cm = confusion_matrix(trues, preds, labels=PAPER_LABELS_EXT)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    short = [x.replace("Offence + ", "O+") for x in PAPER_LABELS_EXT]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short,
        yticklabels=short,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({title_suffix})",
    )
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
