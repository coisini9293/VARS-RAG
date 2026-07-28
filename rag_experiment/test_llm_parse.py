#!/usr/bin/env python3
"""
Small offline re-parse test: compare rule vs LLM label extraction on existing
R0 checkpoint raw answers (no full model inference).

Usage (from project root, conda env asd):
  python -m rag_experiment.test_llm_parse
  python -m rag_experiment.test_llm_parse --n 20 --checkpoint rag_experiment_runs/run1/R0/checkpoint.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from rag_experiment.label_parse import parse_paper_label
from rag_experiment.labels import PAPER_LABELS
from rag_experiment.llm_parse import get_llm_client, parse_paper_label_llm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default=os.path.join("rag_experiment_runs", "run1", "R0", "checkpoint.json"),
    )
    p.add_argument("--n", type=int, default=20, help="How many samples to LLM-parse")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None, help="Override Qwen model id")
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    print("[1] Testing API credentials from .env ...", flush=True)
    client = get_llm_client()
    ping = client.chat.completions.create(
        model=args.model or os.environ.get("QWEN_PARSE_MODEL", "qwen-plus"),
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0,
        max_tokens=8,
    )
    ping_txt = (ping.choices[0].message.content or "").strip()
    print(f"    API OK, reply={ping_txt!r}", flush=True)

    data = json.load(open(args.checkpoint, encoding="utf-8"))
    items = list(data.items())
    random.Random(args.seed).shuffle(items)
    # Stratify a bit: include some yellow/red truths if present
    by_true: dict[str, list] = {}
    for k, v in data.items():
        by_true.setdefault(v["y_true"], []).append((k, v))
    selected = []
    per = max(1, args.n // max(1, len(by_true)))
    for lab, rows in by_true.items():
        random.Random(args.seed).shuffle(rows)
        selected.extend(rows[:per])
    # fill up
    remain = [x for x in items if x not in selected]
    while len(selected) < min(args.n, len(items)) and remain:
        selected.append(remain.pop())
    selected = selected[: args.n]

    print(f"[2] Re-parsing n={len(selected)} from {args.checkpoint}", flush=True)
    y_true, y_rule, y_llm = [], [], []
    disagreements = 0
    for i, (ckey, rec) in enumerate(selected):
        raw = rec.get("raw") or ""
        true = rec["y_true"]
        rule = parse_paper_label(raw)
        llm = parse_paper_label_llm(raw, client=client, model=args.model)
        y_true.append(true)
        y_rule.append(rule)
        y_llm.append(llm)
        mark = "" if rule == llm else " DIFF"
        if rule != llm:
            disagreements += 1
        print(
            f"  [{i+1}/{len(selected)}] true={true!r}\n"
            f"       rule={rule!r}  llm={llm!r}{mark}\n"
            f"       raw={raw[:100]!r}",
            flush=True,
        )

    print("\n[3] Metrics on this subset", flush=True)
    print(
        f"  rule: acc={accuracy_score(y_true, y_rule):.4f} "
        f"bal={balanced_accuracy_score(y_true, y_rule):.4f} "
        f"pred_dist={dict(Counter(y_rule))}",
        flush=True,
    )
    print(
        f"  llm:  acc={accuracy_score(y_true, y_llm):.4f} "
        f"bal={balanced_accuracy_score(y_true, y_llm):.4f} "
        f"pred_dist={dict(Counter(y_llm))}",
        flush=True,
    )
    print(f"  rule vs llm disagreements: {disagreements}/{len(selected)}", flush=True)
    print("\n--- LLM classification report ---", flush=True)
    print(
        classification_report(y_true, y_llm, labels=PAPER_LABELS, zero_division=0),
        flush=True,
    )


if __name__ == "__main__":
    main()
