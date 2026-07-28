"""Figures for metrics (English labels)."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from experiment.constants import LABELS_EXT


def save_metrics_bar_chart(line_keys: List[str], summary: Dict[str, Any], out_path: str) -> None:
    accs = [summary[k]["accuracy"] for k in line_keys]
    f1m = [summary[k]["f1_macro"] for k in line_keys]
    x = np.arange(len(line_keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(line_keys) * 1.2), 5))
    ax.bar(x - w / 2, accs, width=w, label="Accuracy")
    ax.bar(x + w / 2, f1m, width=w, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(line_keys)
    ax.set_ylabel("Score")
    ax.set_title("Metrics by Experiment Line")
    ax.legend()
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_png(trues: List[str], preds: List[str], title_suffix: str, out_path: str) -> None:
    cm = confusion_matrix(trues, preds, labels=LABELS_EXT)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=LABELS_EXT,
        yticklabels=LABELS_EXT,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion matrix ({title_suffix})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
