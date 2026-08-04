"""Build a Chinese RAG-effect analysis report from a finished run directory."""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # Prefer last record per action/clip (resume may append)
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = f"{r.get('action_id')}/{r.get('clip_idx')}"
        latest[key] = r
    return list(latest.values())


def _pair_r0_rag(
    r0: List[Dict[str, Any]], rag: List[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    m0 = {f"{r.get('action_id')}/{r.get('clip_idx')}": r for r in r0}
    out = []
    for r in rag:
        key = f"{r.get('action_id')}/{r.get('clip_idx')}"
        if key in m0:
            out.append((m0[key], r))
    return out


def write_chinese_rag_analysis(run_dir: str) -> str:
    summary_path = os.path.join(run_dir, "summary", "metrics.json")
    delta_path = os.path.join(run_dir, "summary", "delta_vs_baseline.json")
    analysis_path = os.path.join(run_dir, "summary", "analysis.json")
    summary = _load_json(summary_path) if os.path.isfile(summary_path) else {}
    delta = _load_json(delta_path) if os.path.isfile(delta_path) else {}
    analysis = _load_json(analysis_path) if os.path.isfile(analysis_path) else {}

    lines = analysis.get("lines") or sorted(summary.keys())
    r0_recs = _load_jsonl(os.path.join(run_dir, "R0", "logs", "predictions.jsonl"))

    parts: List[str] = []
    parts.append("# RAG 效果分析报告（中文）")
    parts.append("")
    parts.append(f"- 运行目录：`{run_dir}`")
    parts.append(f"- 评测样本量：{analysis.get('n_eval_items', '未知')}")
    parts.append(f"- 实验线：{', '.join(lines)}")
    parts.append(f"- 最优线（Balanced Accuracy）：**{analysis.get('best_line')}**")
    parts.append("")
    parts.append("## 1. 总体指标对比")
    parts.append("")
    parts.append("| 线 | Accuracy | Balanced Acc | Macro-F1 | Parse率 |")
    parts.append("|----|----------|--------------|----------|---------|")
    for lk in lines:
        m = summary.get(lk) or {}
        parts.append(
            f"| {lk} | {float(m.get('accuracy', 0)):.4f} | "
            f"{float(m.get('balanced_accuracy', 0)):.4f} | "
            f"{float(m.get('f1_macro', 0)):.4f} | "
            f"{float(m.get('parse_rate', 0)):.4f} |"
        )
    parts.append("")
    parts.append("相对 Baseline（R0）的提升：")
    parts.append("")
    for lk, d in (delta.get("deltas") or {}).items():
        parts.append(
            f"- **{lk}**：Accuracy {float(d.get('accuracy_delta', 0)):+.4f}，"
            f"Balanced Acc {float(d.get('balanced_accuracy_delta', 0)):+.4f}"
        )
    parts.append("")

    parts.append("## 2. 自动结论")
    parts.append("")
    for f in analysis.get("findings") or []:
        parts.append(f"- {f}")
    parts.append("")
    for r in analysis.get("recommendations") or []:
        parts.append(f"- 建议：{r}")
    parts.append("")

    # Per RAG line: retrieval + case studies vs R0
    for lk in lines:
        if lk == "R0":
            continue
        rag_recs = _load_jsonl(os.path.join(run_dir, lk, "logs", "predictions.jsonl"))
        if not rag_recs:
            continue
        ret_path = os.path.join(run_dir, lk, "plots", "retrieval_stats.json")
        rstats = _load_json(ret_path) if os.path.isfile(ret_path) else {}

        parts.append(f"## 3. {lk} 检索与问答样例")
        parts.append("")
        parts.append(
            f"- 空检索率：{float(rstats.get('empty_rate', 0)):.4f}；"
            f"平均 Top1 相似度：{rstats.get('mean_top1_score')}；"
            f"平均上下文长度：{float(rstats.get('mean_context_chars', 0)):.1f}"
        )
        parts.append("")

        pairs = _pair_r0_rag(r0_recs, rag_recs)
        # cases where RAG flips wrong->right or right->wrong
        improved = [
            (a, b)
            for a, b in pairs
            if a.get("y_pred") != a.get("y_true") and b.get("y_pred") == b.get("y_true")
        ]
        worsened = [
            (a, b)
            for a, b in pairs
            if a.get("y_pred") == a.get("y_true") and b.get("y_pred") != b.get("y_true")
        ]
        same_wrong = [
            (a, b)
            for a, b in pairs
            if a.get("y_pred") != a.get("y_true") and b.get("y_pred") != b.get("y_true")
        ]
        parts.append(
            f"- 相对 R0：纠错 {len(improved)}，引入新错 {len(worsened)}，"
            f"双方都错 {len(same_wrong)}"
        )
        parts.append("")

        def _emit_case(title: str, a: Dict[str, Any], b: Dict[str, Any]) -> None:
            parts.append(f"### {title}")
            parts.append("")
            parts.append(
                f"- Action `{a.get('action_id')}` clip `{a.get('clip_idx')}` | "
                f"真值：**{a.get('y_true')}** | "
                f"R0 预测：{a.get('y_pred')} | {lk} 预测：**{b.get('y_pred')}**"
            )
            parts.append(f"- 视频：`{b.get('video')}`")
            parts.append("")
            parts.append("**原问题（论文问题，未加 RAG）：**")
            parts.append("")
            parts.append("```text")
            parts.append(str(b.get("original_question") or a.get("original_question") or ""))
            parts.append("```")
            parts.append("")
            if b.get("classifier_text"):
                parts.append("**CLIP 分类头文本：**")
                parts.append("")
                parts.append("```text")
                parts.append(str(b.get("classifier_text")))
                parts.append("```")
                parts.append("")
            parts.append("**检索查询：**")
            parts.append("")
            parts.append("```text")
            parts.append(str(b.get("retrieval_query") or ""))
            parts.append("```")
            parts.append("")
            parts.append("**RAG 检索结果：**")
            parts.append("")
            hits = ((b.get("retrieval") or {}).get("hits")) or []
            if not hits:
                parts.append("- （空）")
            for h in hits:
                parts.append(
                    f"- rank={h.get('rank', '?')} score={h.get('score')} "
                    f"len={h.get('text_len')}"
                )
                parts.append("")
                parts.append("```text")
                parts.append(str(h.get("text") or h.get("text_preview") or "")[:800])
                parts.append("```")
                parts.append("")
            parts.append("**RAG 增强后的问题（注入规则后）：**")
            parts.append("")
            parts.append("```text")
            parts.append(str(b.get("enhanced_question") or "")[:2500])
            parts.append("```")
            parts.append("")
            parts.append("**模型原始回答（R0 / 本线）：**")
            parts.append("")
            parts.append("```text")
            parts.append("--- R0 ---")
            if a.get("raw_foul") or a.get("raw_card"):
                parts.append(f"[犯规] {str(a.get('raw_foul') or '')[:400]}")
                parts.append(f"[牌型] {str(a.get('raw_card') or '')[:400]}")
            else:
                parts.append(str(a.get("raw") or "")[:800])
            parts.append(f"--- {lk} ---")
            if b.get("raw_foul") or b.get("raw_card"):
                parts.append(f"[犯规] {str(b.get('raw_foul') or '')[:400]}")
                parts.append(f"[牌型] {str(b.get('raw_card') or '')[:400]}")
            else:
                parts.append(str(b.get("raw") or "")[:800])
            parts.append("```")
            parts.append("")

        # Up to 2 improved + 1 worsened + 1 random with retrieval
        shown = 0
        for a, b in improved[:2]:
            _emit_case(f"{lk} 纠错样例 #{shown+1}", a, b)
            shown += 1
        for a, b in worsened[:1]:
            _emit_case(f"{lk} 变差样例", a, b)
            shown += 1
        if shown == 0:
            # fallback: first sample with non-empty retrieval
            for a, b in pairs:
                if not (b.get("retrieval") or {}).get("empty", True):
                    _emit_case(f"{lk} 检索样例", a, b)
                    break

    out = os.path.join(run_dir, "summary", "RAG效果分析.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    return out


if __name__ == "__main__":
    import sys

    run_dir = sys.argv[1] if len(sys.argv) > 1 else "rag_experiment_runs/full_paper"
    path = write_chinese_rag_analysis(run_dir)
    print(path)
