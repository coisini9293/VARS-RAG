"""Per-line directory layout under a run root."""
from __future__ import annotations

import os


def line_root(run_dir: str, line_key: str) -> str:
    return os.path.join(run_dir, line_key)


def line_paths(run_dir: str, line_key: str) -> dict[str, str]:
    root = line_root(run_dir, line_key)
    return {
        "root": root,
        "settings": os.path.join(root, "experiment_settings.json"),
        "checkpoint": os.path.join(root, "checkpoint.json"),
        "logs_dir": os.path.join(root, "logs"),
        "predictions_log": os.path.join(root, "logs", "predictions.jsonl"),
        "plots_dir": os.path.join(root, "plots"),
        "plot_confusion": os.path.join(root, "plots", "confusion_matrix.png"),
        "metrics_line": os.path.join(root, "plots", "metrics_line.json"),
    }


def summary_paths(run_dir: str) -> dict[str, str]:
    s = os.path.join(run_dir, "summary")
    return {
        "root": s,
        "metrics": os.path.join(s, "metrics.json"),
        "plots_dir": os.path.join(s, "plots"),
        "metrics_bar": os.path.join(s, "plots", "metrics_by_line.png"),
    }
