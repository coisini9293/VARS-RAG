"""
Backend RAG ablation runner (no Streamlit).

Features:
- Lines R0–R4 (baseline vs RAG variants)
- Rich progress bar
- Video-feature cache shared across lines
- Checkpoint resume
- English plots + Markdown/JSON analysis report
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import x_vars_demo_with_streamlit as xv
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from experiment.constants import LOCAL_CLIP_DIR
from experiment.model_load import load_models
from experiment.video_encode import encode_video_features
from rag_experiment.clip_classifier import (
    load_clip_classifier,
    predict_from_frames,
    prediction_to_dict,
)
from rag_experiment.dataset import EvalItem, load_eval_items, summarize_label_distribution
from rag_experiment.frame_sample import load_frames_tensor
from rag_experiment.knowledge_rag import fetch_rag_context
from rag_experiment.label_parse import get_parse_fn
from rag_experiment.labels import paper_protocol_notes
from rag_experiment.layout import line_paths, summary_paths
from rag_experiment.metrics import (
    compute_classification_metrics,
    compute_retrieval_stats,
    delta_vs_baseline,
)
from rag_experiment.plotting import (
    save_confusion_matrix_png,
    save_delta_accuracy_bar,
    save_metrics_bar_chart,
    save_retrieval_empty_rate_bar,
)
from rag_experiment.presets import DEFAULT_LINES, LINE_PRESETS, LineSpec
from rag_experiment.paper_features import (
    PAPER_VIDEO_TOKEN_LEN,
    encode_paper_video_features,
)
from rag_experiment.paper_prompt import DEFAULT_XVARS_CONV_MODE, build_xvars_prompt
from rag_experiment.prompts import (
    PAPER_Q_CARD,
    PAPER_Q_FOUL,
    TASK_CLASSIFY_OFFENCE_SEVERITY,
    build_user_question_single,
    prompt_templates_for_log,
)
from rag_experiment.query_builder import build_retrieval_query
from rag_experiment.report import build_analysis, write_analysis_json, write_report_markdown

# Default path used by create_features.py / Streamlit
DEFAULT_CLIP_CLASSIFIER_WEIGHTS = os.path.join("visual_encoder", "14_model.pth.tar")


def _predict_greedy(
    model,
    tokenizer,
    video_feat,
    prompt: str,
    *,
    temperature: float,
    max_new_tokens: int,
    do_sample: bool,
    parse_fn=None,
) -> tuple[str, str]:
    out = xv.model_generate_simple(
        model=model,
        tokenizer=tokenizer,
        video_spatio_temporal_features=video_feat,
        prompt=prompt,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        debug=False,
    )
    pred = parse_fn(out) if parse_fn is not None else out
    return pred, out


def _generate_paper_answers(
    model,
    tokenizer,
    video_feat: torch.Tensor,
    *,
    clf_text: str,
    rag_text: str,
    video_token_len: int,
    conv_mode: str,
    temperature: float,
    max_new_tokens: int,
    do_sample: bool,
    split_questions: bool,
) -> tuple[str, str, str]:
    """Returns (combined_raw, raw_foul, raw_card)."""
    vlen = int(video_feat.shape[1])
    if vlen != int(video_token_len):
        raise ValueError(f"video_feat len {vlen} != video_token_len {video_token_len}")

    if split_questions:
        msg_foul = build_user_question_single(PAPER_Q_FOUL, rag_text, clf_text)
        msg_card = build_user_question_single(PAPER_Q_CARD, rag_text, clf_text)
        prompt_foul = build_xvars_prompt(msg_foul, video_token_len, conv_mode=conv_mode)
        prompt_card = build_xvars_prompt(msg_card, video_token_len, conv_mode=conv_mode)
        raw_foul = xv.model_generate_simple(
            model=model,
            tokenizer=tokenizer,
            video_spatio_temporal_features=video_feat,
            prompt=prompt_foul,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
        )
        raw_card = xv.model_generate_simple(
            model=model,
            tokenizer=tokenizer,
            video_spatio_temporal_features=video_feat,
            prompt=prompt_card,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
        )
        combined = f"{raw_foul.strip()}\n\n{raw_card.strip()}".strip()
        return combined, raw_foul, raw_card

    msg = build_user_question_single(TASK_CLASSIFY_OFFENCE_SEVERITY, rag_text, clf_text)
    prompt = build_xvars_prompt(msg, video_token_len, conv_mode=conv_mode)
    raw = xv.model_generate_simple(
        model=model,
        tokenizer=tokenizer,
        video_spatio_temporal_features=video_feat,
        prompt=prompt,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
    )
    return raw, raw, ""


def _write_line_settings(
    paths: Dict[str, str],
    spec: LineSpec,
    args: argparse.Namespace,
) -> None:
    os.makedirs(paths["logs_dir"], exist_ok=True)
    os.makedirs(paths["plots_dir"], exist_ok=True)
    payload = {
        "line": spec.key,
        "name": spec.name,
        "description": spec.description,
        "rag": asdict(spec.rag),
        "generation": {
            "max_new_tokens": spec.max_new_tokens,
            "do_sample": spec.do_sample,
            "temperature": spec.temperature,
            "video_token_len": args.video_token_len,
            "max_frames": args.max_frames,
            "clip_batch_size": args.clip_bs,
            "frame_sample": args.frame_sample,
            "foul_time_sec": args.foul_time_sec,
            "use_clip_classifier": bool(args.use_clip_classifier),
            "clip_classifier_weights": args.clip_classifier_weights,
            "label_parse": args.label_parse,
            "split_questions": bool(args.split_questions),
            "conv_mode": args.conv_mode,
            "paper_features": bool(args.paper_features),
        },
        "task_prompt": TASK_CLASSIFY_OFFENCE_SEVERITY,
        "prompt_templates": prompt_templates_for_log(),
        "paper_protocol": paper_protocol_notes(),
        "dataset": {
            "annotations": args.annotations,
            "dataset_root": args.dataset_root,
            "max_action": args.max_action,
            "unit": args.unit,
            "paper_protocol": bool(args.paper_protocol),
            "prefer_main_camera": bool(args.prefer_main_camera),
            "skip_missing_video": True,
        },
        "seed_base": args.seed,
        "notes": [
            "Labels follow X-VARS 4-way Offence+Severity (not legacy Between).",
            "Ground-truth Offence/Severity never used in retrieval queries.",
            "Paper-aligned inference: create_features frames 63-87, 300 video tokens (44+256).",
            "Football conv template (xvars_football_v1) + split foul/card questions.",
            "Label parse: LLM extract (paper: ChatGPT-3.5; default DashScope Qwen).",
        ],
    }
    with open(paths["settings"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sample_cache_key(item: EvalItem, args: argparse.Namespace) -> str:
    return (
        f"{item.action_id}/{item.clip_idx}|"
        f"{args.frame_sample}|f{args.max_frames}|t{args.foul_time_sec}|"
        f"vt{args.video_token_len}|pf{int(args.paper_features)}"
    )


def _encode_cached(
    feature_cache: Dict[str, torch.Tensor],
    frame_meta_cache: Dict[str, dict],
    clf_cache: Dict[str, dict],
    item: EvalItem,
    vision_tower_hf,
    clip_device: torch.device,
    args: argparse.Namespace,
    clip_clf,
) -> Optional[torch.Tensor]:
    key = _sample_cache_key(item, args)
    if key in feature_cache:
        return feature_cache[key]
    if not item.video_exists and not os.path.isfile(item.video_path):
        return None

    frames, frame_meta = load_frames_tensor(
        item.video_path,
        mode=args.frame_sample,
        max_frames=args.max_frames,
        foul_time_sec=float(args.foul_time_sec),
        n_before=8,
        n_after=8,
        device=clip_device,
    )
    frame_meta_cache[key] = frame_meta

    vf = (
        encode_paper_video_features(frames, vision_tower_hf, clip_device, args.clip_bs)
        if args.paper_features
        else encode_video_features(
            frames, vision_tower_hf, clip_device, args.clip_bs, args.video_token_len
        )
    )
    feature_cache[key] = vf

    if clip_clf is not None and key not in clf_cache:
        pred = predict_from_frames(clip_clf, frames)
        clf_cache[key] = prediction_to_dict(pred)

    return vf


def _run_one_sample(
    *,
    model,
    tokenizer,
    video_feat: torch.Tensor,
    rag: xv.FootballRAGSystem,
    item: EvalItem,
    spec: LineSpec,
    seed: int,
    classifier_info: Optional[dict],
    use_clip_classifier: bool,
    parse_fn,
    video_token_len: int,
    conv_mode: str,
    split_questions: bool,
) -> Dict[str, Any]:
    query = build_retrieval_query(spec.rag.query_mode, item.action_meta)
    ret = fetch_rag_context(rag, query, spec.rag)
    clf_text = ""
    if use_clip_classifier and classifier_info:
        clf_text = str(classifier_info.get("prompt_text", "") or "")
    rag_text = ret.text if spec.rag.enabled else ""

    combined_raw, raw_foul, raw_card = _generate_paper_answers(
        model,
        tokenizer,
        video_feat,
        clf_text=clf_text,
        rag_text=rag_text,
        video_token_len=video_token_len,
        conv_mode=conv_mode,
        temperature=spec.temperature,
        max_new_tokens=spec.max_new_tokens,
        do_sample=spec.do_sample,
        split_questions=split_questions,
    )
    pred = parse_fn(combined_raw)

    enhanced_foul = build_user_question_single(PAPER_Q_FOUL, rag_text, clf_text)
    enhanced_card = build_user_question_single(PAPER_Q_CARD, rag_text, clf_text)

    return {
        "y_true": item.y_true,
        "y_pred": pred,
        "video": item.video_path,
        "raw": combined_raw,
        "raw_foul": raw_foul,
        "raw_card": raw_card,
        "action_id": item.action_id,
        "clip_idx": item.clip_idx,
        "camera_type": item.camera_type,
        "original_question": TASK_CLASSIFY_OFFENCE_SEVERITY,
        "paper_q_foul": PAPER_Q_FOUL,
        "paper_q_card": PAPER_Q_CARD,
        "classifier_text": clf_text,
        "retrieval_query": query if spec.rag.enabled else "",
        "rag_context": rag_text,
        "enhanced_question_foul": enhanced_foul,
        "enhanced_question_card": enhanced_card,
        "enhanced_question": (
            f"[FOUL]\n{enhanced_foul}\n\n[CARD]\n{enhanced_card}"
            if split_questions
            else build_user_question_single(TASK_CLASSIFY_OFFENCE_SEVERITY, rag_text, clf_text)
        ),
        "retrieval": {
            "enabled": bool(spec.rag.enabled),
            "empty": ret.empty,
            "top1_score": ret.top1_score,
            "context_chars": len(rag_text),
            "n_hits": len(ret.hits),
            "hits": ret.hits,
            "search_mode": getattr(rag, "search_mode", None),
        },
        "classifier": classifier_info or {},
        "seed": seed,
        "split_questions": split_questions,
        "video_token_len": video_token_len,
        "conv_mode": conv_mode,
    }


def _summarize_line(
    items: List[EvalItem],
    line_ck: Dict[str, Any],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    preds: List[str] = []
    trues: List[str] = []
    records: List[Dict[str, Any]] = []
    for item in items:
        ckey = f"{item.action_id}/{item.clip_idx}"
        rec = line_ck.get(ckey)
        if rec is None:
            continue
        preds.append(rec["y_pred"])
        trues.append(rec["y_true"])
        records.append(rec)
    return trues, preds, records


def run(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(
        description="X-VARS RAG ablation experiment (backend only, English figures/report)."
    )
    p.add_argument("--dataset-root", default=os.path.join("dataset", "Test"))
    p.add_argument(
        "--annotations", default=os.path.join("dataset", "Test", "annotations.json")
    )
    p.add_argument("--output-dir", default=os.path.join("rag_experiment_runs", "default"))
    p.add_argument("--lines", default=DEFAULT_LINES, help="Comma-separated, e.g. R0,R1,R3")
    p.add_argument("--max-action", type=int, default=300)
    p.add_argument("--video-token-len", type=int, default=PAPER_VIDEO_TOKEN_LEN)
    p.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Frames per clip (paper create_features → 16 @ 224px)",
    )
    p.add_argument("--clip-bs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--baseline",
        default="R0",
        help="Line key used as baseline for delta plots/report",
    )
    p.add_argument(
        "--unit",
        default="action",
        choices=["action", "clip"],
        help="action=one view/action (paper); clip=all multi-view clips",
    )
    p.add_argument(
        "--paper-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="4-way Offence+Severity labels; skip Between/ambiguous severity (default: on)",
    )
    p.add_argument(
        "--prefer-main-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When unit=action, prefer Main camera center (default: on)",
    )
    p.add_argument(
        "--frame-sample",
        default="paper_xvars",
        choices=["paper_xvars", "foul_centered", "uniform"],
        help="paper_xvars=create_features 63-87 @17fps; foul_centered=legacy 8+8 @3s",
    )
    p.add_argument(
        "--foul-time-sec",
        type=float,
        default=3.0,
        help="In-clip foul center in seconds (create_features convention; default 3.0)",
    )
    p.add_argument(
        "--use-clip-classifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject CLIP multi-task prediction text into the LLM prompt (default: on)",
    )
    p.add_argument(
        "--clip-classifier-weights",
        default=DEFAULT_CLIP_CLASSIFIER_WEIGHTS,
        help="Path to 14_model.pth.tar",
    )
    p.add_argument(
        "--label-parse",
        default="llm",
        choices=["rule", "llm", "hybrid"],
        help="Map free-form answers to 4-way labels. Default llm (paper: ChatGPT extract via DashScope). rule=offline smoke only.",
    )
    p.add_argument(
        "--paper-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use create_features spatio-temporal encoding (44+256=300 tokens)",
    )
    p.add_argument(
        "--split-questions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask foul and card in two generations (paper Table 4 protocol)",
    )
    p.add_argument(
        "--conv-mode",
        default=DEFAULT_XVARS_CONV_MODE,
        help="Conversation template (default: xvars_football_v1)",
    )
    args = p.parse_args(argv)

    if args.paper_features and int(args.video_token_len) != PAPER_VIDEO_TOKEN_LEN:
        print(
            f"[rag_experiment] paper_features on: override video_token_len "
            f"{args.video_token_len} -> {PAPER_VIDEO_TOKEN_LEN}",
            flush=True,
        )
        args.video_token_len = PAPER_VIDEO_TOKEN_LEN

    os.makedirs(args.output_dir, exist_ok=True)
    lines = [x.strip() for x in args.lines.split(",") if x.strip()]
    for k in lines:
        if k not in LINE_PRESETS:
            raise SystemExit(f"Unknown line {k}. Choose from: {', '.join(LINE_PRESETS)}")

    items = load_eval_items(
        args.annotations,
        args.dataset_root,
        args.max_action,
        unit=args.unit,
        paper_protocol=bool(args.paper_protocol),
        prefer_main_camera=bool(args.prefer_main_camera),
    )
    print(f"[dataset] label_dist={summarize_label_distribution(items)}", flush=True)
    print("[rag_experiment] Loading models...", flush=True)
    model, vision_tower_hf, tokenizer, _, device = load_models()
    clip_device = device

    print("[rag_experiment] Initializing FootballRAGSystem...", flush=True)
    rag = xv.FootballRAGSystem(device=xv.DEFAULT_RAG_DEVICE)
    rag_ok = rag.initialize()
    if not rag_ok:
        print(
            "[WARN] RAG initialize() returned False. "
            "R1+ lines will have empty retrieval unless text fallback still works.",
            flush=True,
        )

    clip_clf = None
    if args.use_clip_classifier:
        print("[rag_experiment] Loading CLIP classifier (14_model.pth.tar)...", flush=True)
        clip_clf = load_clip_classifier(
            weights_path=args.clip_classifier_weights,
            clip_dir=LOCAL_CLIP_DIR if os.path.isdir(LOCAL_CLIP_DIR) else os.path.join(
                "visual_encoder", "clip-vit-large-patch14"
            ),
            device=clip_device,
        )
    else:
        print("[rag_experiment] CLIP classifier text injection DISABLED", flush=True)

    print(
        f"[rag_experiment] frame_sample={args.frame_sample} "
        f"video_token_len={args.video_token_len} paper_features={args.paper_features} "
        f"split_questions={args.split_questions} conv_mode={args.conv_mode} "
        f"label_parse={args.label_parse}",
        flush=True,
    )

    parse_fn = get_parse_fn(args.label_parse)
    # Warm LLM client early so credential errors fail fast
    if args.label_parse in ("llm", "hybrid"):
        from rag_experiment.llm_parse import get_llm_client

        get_llm_client()
        print(f"[rag_experiment] LLM label parser ready ({args.label_parse})", flush=True)

    feature_cache: Dict[str, torch.Tensor] = {}
    frame_meta_cache: Dict[str, dict] = {}
    clf_cache: Dict[str, dict] = {}
    total_steps = len(lines) * len(items)
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task_id = progress.add_task("RAG experiment", total=total_steps)
        for lk in lines:
            spec = LINE_PRESETS[lk]
            paths = line_paths(args.output_dir, lk)
            _write_line_settings(paths, spec, args)
            line_ck = _load_checkpoint(paths["checkpoint"])
            pred_log = open(paths["predictions_log"], "a", encoding="utf-8")

            for item in items:
                ckey = f"{item.action_id}/{item.clip_idx}"
                progress.update(task_id, description=f"{lk} {ckey}")
                if ckey in line_ck:
                    progress.advance(task_id)
                    continue
                try:
                    vf = _encode_cached(
                        feature_cache,
                        frame_meta_cache,
                        clf_cache,
                        item,
                        vision_tower_hf,
                        clip_device,
                        args,
                        clip_clf,
                    )
                    if vf is None:
                        raise FileNotFoundError(f"video missing: {item.video_path}")
                    skey = _sample_cache_key(item, args)
                    seed = args.seed + int(item.action_id) * 1000 + item.clip_idx
                    rec = _run_one_sample(
                        model=model,
                        tokenizer=tokenizer,
                        video_feat=vf,
                        rag=rag,
                        item=item,
                        spec=spec,
                        seed=seed,
                        classifier_info=clf_cache.get(skey),
                        use_clip_classifier=bool(args.use_clip_classifier),
                        parse_fn=parse_fn,
                        video_token_len=int(args.video_token_len),
                        conv_mode=str(args.conv_mode),
                        split_questions=bool(args.split_questions),
                    )
                    rec["frame_sample"] = frame_meta_cache.get(skey, {})
                    line_ck[ckey] = rec
                    pred_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pred_log.flush()
                    with open(paths["checkpoint"], "w", encoding="utf-8") as cf:
                        json.dump(line_ck, cf, ensure_ascii=False, indent=2)
                except Exception as e:
                    err_rec = {
                        "y_true": item.y_true,
                        "y_pred": "Unknown",
                        "video": item.video_path,
                        "raw": "",
                        "action_id": item.action_id,
                        "clip_idx": item.clip_idx,
                        "error": str(e),
                        "retrieval": {"empty": True, "top1_score": None, "context_chars": 0, "n_hits": 0, "hits": []},
                    }
                    line_ck[ckey] = err_rec
                    pred_log.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                    pred_log.flush()
                    with open(paths["checkpoint"], "w", encoding="utf-8") as cf:
                        json.dump(line_ck, cf, ensure_ascii=False, indent=2)
                    print(f"[ERROR] {lk} {ckey}: {e}", flush=True)
                    traceback.print_exc()
                progress.advance(task_id)

            pred_log.close()

    # ---- Aggregate metrics / plots / report ----
    sp = summary_paths(args.output_dir)
    os.makedirs(sp["plots_dir"], exist_ok=True)

    summary: Dict[str, Any] = {}
    retrieval_by_line: Dict[str, Any] = {}

    for lk in lines:
        paths = line_paths(args.output_dir, lk)
        line_ck = _load_checkpoint(paths["checkpoint"])
        trues, preds, records = _summarize_line(items, line_ck)
        line_metrics = {"line": lk, **compute_classification_metrics(trues, preds)}
        summary[lk] = line_metrics
        with open(paths["metrics_line"], "w", encoding="utf-8") as mf:
            json.dump(line_metrics, mf, ensure_ascii=False, indent=2)

        if LINE_PRESETS[lk].rag.enabled:
            rstats = compute_retrieval_stats(records)
            retrieval_by_line[lk] = rstats
            with open(paths["retrieval_stats"], "w", encoding="utf-8") as rf:
                json.dump(rstats, rf, ensure_ascii=False, indent=2)

        if trues and preds:
            save_confusion_matrix_png(trues, preds, lk, paths["plot_confusion"])

    delta_payload = delta_vs_baseline(summary, baseline_key=args.baseline)

    with open(sp["metrics"], "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
    with open(sp["delta"], "w", encoding="utf-8") as df:
        json.dump(delta_payload, df, ensure_ascii=False, indent=2)

    save_metrics_bar_chart(lines, summary, sp["metrics_bar"])
    save_delta_accuracy_bar(delta_payload, sp["delta_bar"])
    save_retrieval_empty_rate_bar(lines, retrieval_by_line, sp["retrieval_bar"])

    args_snapshot = {
        "dataset_root": args.dataset_root,
        "annotations": args.annotations,
        "output_dir": args.output_dir,
        "lines": lines,
        "max_action": args.max_action,
        "video_token_len": args.video_token_len,
        "max_frames": args.max_frames,
        "clip_bs": args.clip_bs,
        "seed": args.seed,
        "baseline": args.baseline,
        "unit": args.unit,
        "paper_protocol": bool(args.paper_protocol),
        "prefer_main_camera": bool(args.prefer_main_camera),
        "frame_sample": args.frame_sample,
        "foul_time_sec": args.foul_time_sec,
        "use_clip_classifier": bool(args.use_clip_classifier),
        "clip_classifier_weights": args.clip_classifier_weights,
        "label_parse": args.label_parse,
        "rag_initialized": bool(rag_ok),
        "paper_protocol_notes": paper_protocol_notes(),
        "label_distribution": summarize_label_distribution(items),
    }
    analysis = build_analysis(
        summary,
        retrieval_by_line,
        delta_payload,
        lines=lines,
        n_items=len(items),
        args_snapshot=args_snapshot,
    )
    write_analysis_json(analysis, sp["report_json"])
    write_report_markdown(analysis, sp["report_md"])

    try:
        from rag_experiment.analyze_rag_cn import write_chinese_rag_analysis

        cn_path = write_chinese_rag_analysis(args.output_dir)
        print(f"Chinese RAG analysis: {cn_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Chinese RAG analysis failed: {e}", flush=True)

    print("\n=== RAG Experiment Summary ===", flush=True)
    for lk in lines:
        m = summary[lk]
        print(
            f"{lk}: n={m['n']} acc={m['accuracy']:.4f} "
            f"bal_acc={m.get('balanced_accuracy', 0):.4f} "
            f"f1_macro={m['f1_macro']:.4f} parse={m['parse_rate']:.4f}",
            flush=True,
        )
    print(f"Report: {sp['report_md']}", flush=True)
    print(f"Plots:  {sp['plots_dir']}", flush=True)
    return analysis


def main(argv: Optional[List[str]] = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
