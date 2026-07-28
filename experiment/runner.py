"""CLI driver: per-line dirs, checkpoint resume, Rich progress, summary plots."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

import torch
import x_vars_demo_with_streamlit as xv
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.metrics import accuracy_score, f1_score

from experiment.constants import LABELS, LABELS_EXT
from experiment.cot import COT_INSTRUCTION_BLOCK
from experiment.dataset_paths import load_eval_items
from experiment.generation import predict_greedy
from experiment.knowledge_rag import fetch_rag_context
from experiment.layout import line_paths, summary_paths
from experiment.line_presets import LINE_PRESETS, LineSpec
from experiment.model_load import load_models
from experiment.plotting import save_confusion_matrix_png, save_metrics_bar_chart
from experiment.prompt_assembly import build_user_question, example_question_for_settings
from experiment.prompt_engineering import TASK_CLASSIFY_OFFENCE, prompt_templates_for_log
from experiment.self_consistency import predict_self_consistency
from experiment.video_encode import encode_video_features


def _write_line_settings(
    paths: Dict[str, str],
    spec: LineSpec,
    args: argparse.Namespace,
) -> None:
    os.makedirs(paths["logs_dir"], exist_ok=True)
    os.makedirs(paths["plots_dir"], exist_ok=True)
    payload = {
        "line": spec.key,
        "rag": asdict(spec.rag),
        "use_expert_prompt": spec.use_expert_prompt,
        "use_cot": spec.use_cot,
        "generation": {
            "max_new_tokens": spec.max_new_tokens,
            "do_sample": spec.do_sample,
            "temperature": spec.temperature,
            "n_vote_self_consistency": spec.n_vote,
            "video_token_len": args.video_token_len,
            "max_frames": args.max_frames,
            "clip_batch_size": args.clip_bs,
        },
        "prompt_templates": prompt_templates_for_log(),
        "cot_instruction_block": COT_INSTRUCTION_BLOCK,
        "rag_retrieval_query": TASK_CLASSIFY_OFFENCE,
        "assembled_user_message_example_no_rag": example_question_for_settings(spec),
        "dataset": {
            "annotations": args.annotations,
            "dataset_root": args.dataset_root,
            "max_action": args.max_action,
        },
        "seed_base": args.seed,
    }
    with open(paths["settings"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _predict(
    model,
    tokenizer,
    video_feat,
    prompt: str,
    spec: LineSpec,
    seed0: int,
) -> tuple[str, str]:
    if spec.n_vote > 1:
        return predict_self_consistency(
            model,
            tokenizer,
            video_feat,
            prompt,
            n_samples=spec.n_vote,
            temperature=spec.temperature,
            max_new_tokens=spec.max_new_tokens,
            seed0=seed0,
        )
    return predict_greedy(
        model,
        tokenizer,
        video_feat,
        prompt,
        temperature=spec.temperature,
        max_new_tokens=spec.max_new_tokens,
        do_sample=spec.do_sample,
    )


def run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=os.path.join("dataset", "Test"))
    p.add_argument("--annotations", default=os.path.join("dataset", "Test", "annotations.json"))
    p.add_argument("--output-dir", default=os.path.join("experiment_runs", "default"))
    p.add_argument("--lines", default="L0,L1,L2,L3,L4,L5,L6")
    p.add_argument("--max-action", type=int, default=300)
    p.add_argument("--video-token-len", type=int, default=200)
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--clip-bs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lines = [x.strip() for x in args.lines.split(",") if x.strip()]
    for k in lines:
        if k not in LINE_PRESETS:
            raise SystemExit(f"Unknown line {k}")

    items = load_eval_items(args.annotations, args.dataset_root, args.max_action)
    model, vision_tower_hf, tokenizer, _, device = load_models()
    clip_device = device
    rag = xv.FootballRAGSystem(device=xv.DEFAULT_RAG_DEVICE)
    rag.initialize()

    total_steps = len(lines) * len(items)
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )

    sp = summary_paths(args.output_dir)
    os.makedirs(sp["plots_dir"], exist_ok=True)

    with progress:
        task_id = progress.add_task("experiment", total=total_steps)
        for lk in lines:
            spec = LINE_PRESETS[lk]
            paths = line_paths(args.output_dir, lk)
            _write_line_settings(paths, spec, args)

            line_ck: Dict[str, Any] = {}
            if os.path.isfile(paths["checkpoint"]):
                line_ck = json.load(open(paths["checkpoint"], encoding="utf-8"))

            pred_log = open(paths["predictions_log"], "a", encoding="utf-8")

            for action_id, clip_idx, vpath, y_true in items:
                ckey = f"{action_id}/{clip_idx}"
                if ckey in line_ck:
                    progress.advance(task_id)
                    continue

                frames = xv.extract_frames_to_tensor(
                    vpath, None, max_frames=args.max_frames, device=clip_device
                )
                vf = encode_video_features(
                    frames, vision_tower_hf, clip_device, args.clip_bs, args.video_token_len
                )
                # Prefer a retrieval query tied to the classification task (not per-video text).
                # Note: for dedicated RAG ablations with adaptive queries, use rag_experiment/.
                rag_text = fetch_rag_context(rag, TASK_CLASSIFY_OFFENCE, spec.rag)
                user_q = build_user_question(spec, rag_text)
                full_prompt = xv.build_prompt(user_q, model, int(args.video_token_len))
                seed = args.seed + int(action_id) * 1000 + clip_idx
                pred, raw = _predict(model, tokenizer, vf, full_prompt, spec, seed)

                rec = {
                    "y_true": y_true,
                    "y_pred": pred,
                    "video": vpath,
                    "raw": raw,
                    "action_id": action_id,
                    "clip_idx": clip_idx,
                }
                line_ck[ckey] = {k: rec[k] for k in ("y_true", "y_pred", "video", "raw")}
                pred_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pred_log.flush()

                with open(paths["checkpoint"], "w", encoding="utf-8") as cf:
                    json.dump(line_ck, cf, ensure_ascii=False, indent=2)
                progress.advance(task_id)

            pred_log.close()

    summary: Dict[str, Any] = {}
    for lk in lines:
        paths = line_paths(args.output_dir, lk)
        line_ck: Dict[str, Any] = {}
        if os.path.isfile(paths["checkpoint"]):
            line_ck = json.load(open(paths["checkpoint"], encoding="utf-8"))
        preds: List[str] = []
        trues: List[str] = []
        parse_ok = 0
        for action_id, clip_idx, _, y_true in items:
            ckey = f"{action_id}/{clip_idx}"
            rec = line_ck.get(ckey)
            if rec is None:
                continue
            yp = rec["y_pred"]
            preds.append(yp)
            trues.append(y_true)
            if yp in LABELS:
                parse_ok += 1
        n = len(preds)
        acc = accuracy_score(trues, preds) if n else 0.0
        f1_macro = (
            f1_score(trues, preds, labels=LABELS_EXT, average="macro", zero_division=0) if n else 0.0
        )
        f1_weighted = (
            f1_score(trues, preds, labels=LABELS_EXT, average="weighted", zero_division=0)
            if n
            else 0.0
        )
        line_metrics = {
            "line": lk,
            "n": n,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "parse_rate": parse_ok / n if n else 0.0,
        }
        summary[lk] = line_metrics
        with open(paths["metrics_line"], "w", encoding="utf-8") as mf:
            json.dump(line_metrics, mf, ensure_ascii=False, indent=2)
        save_confusion_matrix_png(trues, preds, lk, paths["plot_confusion"])

    with open(sp["metrics"], "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    line_keys = list(summary.keys())
    save_metrics_bar_chart(line_keys, summary, sp["metrics_bar"])


def main() -> None:
    run()


if __name__ == "__main__":
    main()
