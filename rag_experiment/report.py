"""Write English analysis report (Markdown + JSON) for a RAG experiment run."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rag_experiment.labels import paper_protocol_notes
from rag_experiment.metrics import pick_best_line
from rag_experiment.presets import LINE_PRESETS


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def build_analysis(
    summary: Dict[str, Dict[str, Any]],
    retrieval_by_line: Dict[str, Dict[str, Any]],
    delta_payload: Dict[str, Any],
    *,
    lines: List[str],
    n_items: int,
    args_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    best = pick_best_line(summary)
    findings: List[str] = []
    base = summary.get("R0") or summary.get(lines[0]) if lines else None
    if base and best and best in summary:
        d_ba = float(summary[best].get("balanced_accuracy", 0.0)) - float(
            base.get("balanced_accuracy", 0.0)
        )
        d_acc = float(summary[best]["accuracy"]) - float(base["accuracy"])
        if best == (delta_payload.get("baseline") or "R0"):
            findings.append(
                "Best line is the baseline; RAG variants did not improve "
                "balanced accuracy on this run."
            )
        elif d_ba > 0.01:
            findings.append(
                f"Best line {best} improves balanced accuracy by {d_ba:+.4f} "
                f"({_pct(summary[best].get('balanced_accuracy', 0))} vs "
                f"{_pct(base.get('balanced_accuracy', 0))}); "
                f"plain accuracy delta={d_acc:+.4f}."
            )
        elif d_ba > 0:
            findings.append(
                f"Best line {best} shows a small balanced-accuracy gain ({d_ba:+.4f}); "
                "treat as tentative until more samples / seeds are run."
            )
        else:
            findings.append(
                f"Best line by balanced accuracy is {best}, but it does not beat "
                "baseline by a clear margin."
            )

    # Retrieval health
    for lk in lines:
        rs = retrieval_by_line.get(lk)
        if not rs:
            continue
        er = float(rs.get("empty_rate", 0.0))
        if er > 0.3:
            findings.append(
                f"{lk}: high empty retrieval rate ({_pct(er)}). "
                "Check encoder, embeddings.pkl, score threshold, or query language mismatch."
            )
        mean_s = rs.get("mean_top1_score")
        if mean_s is not None and float(mean_s) < 0.35:
            findings.append(
                f"{lk}: low mean Top1 similarity ({float(mean_s):.3f}); "
                "retrieved rules may be weakly related to the query."
            )

    # Compare R1 vs R3 depth (balanced accuracy — paper-aligned)
    if "R1" in summary and "R3" in summary:
        d = float(summary["R3"].get("balanced_accuracy", 0)) - float(
            summary["R1"].get("balanced_accuracy", 0)
        )
        if d > 0.01:
            findings.append(
                f"RAG-Deep (R3) outperforms RAG-Base (R1) by {d:+.4f} balanced accuracy — "
                "longer / multi-chunk context helped."
            )
        elif d < -0.01:
            findings.append(
                f"RAG-Deep (R3) underperforms RAG-Base (R1) by {d:+.4f} balanced accuracy — "
                "extra context may add noise or crowd the prompt."
            )

    if "R3" in summary and "R4" in summary:
        d = float(summary["R4"].get("balanced_accuracy", 0)) - float(
            summary["R3"].get("balanced_accuracy", 0)
        )
        if abs(d) >= 0.01:
            findings.append(
                f"Adaptive query (R4) vs fixed Deep (R3) balanced-accuracy delta: {d:+.4f}."
            )

    if not findings:
        findings.append(
            "No strong automatic finding; inspect confusion matrices and prediction logs."
        )

    recommendations: List[str] = []
    if best and best.startswith("R") and best != "R0":
        recommendations.append(
            f"Prefer configuration of {best} "
            f"({LINE_PRESETS[best].name if best in LINE_PRESETS else best}) "
            "as the default RAG setting for deployment, then re-validate on a held-out seed."
        )
    else:
        recommendations.append(
            "If RAG does not help, prioritize visual errors vs rule errors: "
            "sample false predictions and check whether the correct rule was retrieved."
        )
    recommendations.append(
        "Keep Offence labels out of retrieval queries to avoid label leakage."
    )
    recommendations.append(
        "If empty_rate is high, rebuild vector_store/embeddings.pkl and confirm "
        "sentence-transformers loads on the intended device."
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_eval_items": n_items,
        "lines": lines,
        "best_line": best,
        "summary": summary,
        "retrieval": retrieval_by_line,
        "delta_vs_baseline": delta_payload,
        "findings": findings,
        "recommendations": recommendations,
        "args": args_snapshot,
    }


def write_report_markdown(analysis: Dict[str, Any], out_path: str) -> None:
    lines = analysis.get("lines") or []
    summary = analysis.get("summary") or {}
    retrieval = analysis.get("retrieval") or {}
    delta = analysis.get("delta_vs_baseline") or {}
    best = analysis.get("best_line")

    parts: List[str] = []
    proto = paper_protocol_notes()
    parts.append("# RAG Experiment Report")
    parts.append("")
    parts.append(f"- Generated (UTC): `{analysis.get('generated_at_utc')}`")
    parts.append(f"- Evaluated items: **{analysis.get('n_eval_items')}**")
    parts.append(f"- Lines: `{', '.join(lines)}`")
    parts.append(f"- Best line (by balanced accuracy): **{best}**")
    parts.append(f"- Task: `{proto['task']}` — {proto['reference']}")
    parts.append(f"- Labels: {proto['labels']}")
    parts.append(f"- Sampling: {proto['view']}; {proto['frames']}")
    parts.append(f"- Excluded from primary set: {proto['excluded']}")
    parts.append("")
    parts.append("## Experiment Design")
    parts.append("")
    parts.append("| Line | Name | Description |")
    parts.append("|------|------|-------------|")
    for lk in lines:
        spec = LINE_PRESETS.get(lk)
        if spec:
            parts.append(f"| {lk} | {spec.name} | {spec.description} |")
        else:
            parts.append(f"| {lk} | - | custom |")
    parts.append("")
    parts.append("## Classification Metrics")
    parts.append("")
    parts.append(
        "| Line | N | Accuracy | Balanced Acc. | Macro-F1 | Weighted-F1 | Parse rate |"
    )
    parts.append(
        "|------|---|----------|---------------|----------|-------------|------------|"
    )
    for lk in lines:
        m = summary.get(lk) or {}
        parts.append(
            f"| {lk} | {m.get('n', 0)} | {float(m.get('accuracy', 0)):.4f} | "
            f"{float(m.get('balanced_accuracy', 0)):.4f} | "
            f"{float(m.get('f1_macro', 0)):.4f} | {float(m.get('f1_weighted', 0)):.4f} | "
            f"{float(m.get('parse_rate', 0)):.4f} |"
        )
    parts.append("")
    parts.append("## Delta vs Baseline")
    parts.append("")
    base_k = delta.get("baseline", "R0")
    parts.append(
        f"Baseline `{base_k}` accuracy = "
        f"**{float(delta.get('baseline_accuracy', 0.0)):.4f}**, "
        f"balanced accuracy = "
        f"**{float(delta.get('baseline_balanced_accuracy', 0.0)):.4f}**"
    )
    parts.append("")
    parts.append(
        "| Line | Accuracy | Acc. delta | Balanced Acc. | BA delta | Macro-F1 delta |"
    )
    parts.append(
        "|------|----------|------------|---------------|----------|----------------|"
    )
    for lk, d in (delta.get("deltas") or {}).items():
        parts.append(
            f"| {lk} | {float(d.get('accuracy', 0)):.4f} | "
            f"{float(d.get('accuracy_delta', 0)):+.4f} | "
            f"{float(d.get('balanced_accuracy', 0)):.4f} | "
            f"{float(d.get('balanced_accuracy_delta', 0)):+.4f} | "
            f"{float(d.get('f1_macro_delta', 0)):+.4f} |"
        )
    parts.append("")
    parts.append("## Retrieval Diagnostics")
    parts.append("")
    parts.append("| Line | Empty rate | Mean Top1 score | Mean context chars | Mean hits used |")
    parts.append("|------|------------|-----------------|--------------------|----------------|")
    for lk in lines:
        rs = retrieval.get(lk)
        if not rs:
            parts.append(f"| {lk} | n/a (no RAG) | - | - | - |")
            continue
        ms = rs.get("mean_top1_score")
        ms_s = f"{float(ms):.4f}" if ms is not None else "-"
        parts.append(
            f"| {lk} | {float(rs.get('empty_rate', 0)):.4f} | {ms_s} | "
            f"{float(rs.get('mean_context_chars', 0)):.1f} | "
            f"{float(rs.get('mean_hits_used', 0)):.2f} |"
        )
    parts.append("")
    parts.append("## Findings")
    parts.append("")
    for f in analysis.get("findings") or []:
        parts.append(f"- {f}")
    parts.append("")
    parts.append("## Recommendations")
    parts.append("")
    for r in analysis.get("recommendations") or []:
        parts.append(f"- {r}")
    parts.append("")
    parts.append("## Artifacts")
    parts.append("")
    parts.append("- `summary/metrics.json` — numeric metrics")
    parts.append("- `summary/delta_vs_baseline.json` — gains vs baseline")
    parts.append("- `summary/plots/metrics_by_line.png` — accuracy / Macro-F1 bars")
    parts.append("- `summary/plots/delta_accuracy_vs_baseline.png` — accuracy deltas")
    parts.append("- `summary/plots/retrieval_empty_rate.png` — empty retrieval rates")
    parts.append("- `<LINE>/plots/confusion_matrix.png` — per-line confusion matrix")
    parts.append("- `<LINE>/logs/predictions.jsonl` — per-sample predictions + retrieval")
    parts.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_analysis_json(analysis: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
