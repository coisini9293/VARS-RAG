# Alignment with X-VARS Paper (Held et al., CVPRW 2024)

This document records how `rag_experiment/` matches — and where it still differs from — the X-VARS evaluation protocol.

## Matched (defaults)

| Aspect | Paper / original X-VARS | This repo |
|--------|-------------------------|-----------|
| Primary recognition task | Offence + Severity 4-way | Same labels |
| Severity codes | No / Yellow / Red card | Annotation `1.0` / `3.0` / `5.0` |
| Metrics | Accuracy + Balanced Accuracy | Both; best line by **balanced accuracy** |
| Views | Single view | `--unit action` + Main camera center |
| Frame count / size | 16 @ 224, 8 before + 8 after foul | `--frame-sample foul_centered` (default), center at `--foul-time-sec 3.0` |
| Classifier text | CLIP multi-task preds in LLM prompt | `--use-clip-classifier` (default on) via `14_model.pth.tar` |
| Prompt theme | “Is it a foul or not? Why?” + “What card would you give? Why?” | Same two questions (combined); order: classifier → questions → RAG → video |
| Parse | ChatGPT-3.5 extracts labels from free-form answers | `--label-parse llm` (DashScope Qwen via `.env`; paper-aligned) |

## Foul centering note

Annotation `Timestamp` is **absolute match time in milliseconds**, not an offset inside the short clip.  
Without the clip’s start time we cannot map it to a frame index.  
This repository’s `create_features.py` documents that the foul is typically at **~3 seconds** (≈ frame 75 at 25 fps) in the ~5 s clips; we use that convention (`--foul-time-sec 3.0`).

## Intentionally excluded from primary set

- `Offence == Between`
- Severity `2.0` / `4.0` / empty

## Remaining differences

1. **Human study / BLEU–ROUGE** for free-form explanations (paper) — we only auto-score discrete labels for RAG ablations.
2. **Multi-view pooling** as in VARS SOTA tables — we evaluate single-view only unless `--unit clip`.

## How to run (paper-aligned defaults)

```bash
conda activate asd
python run_rag_experiment.py \
  --output-dir rag_experiment_runs/paper_align \
  --lines R0,R1,R2,R3,R4 \
  --unit action \
  --paper-protocol \
  --prefer-main-camera \
  --frame-sample foul_centered \
  --foul-time-sec 3.0 \
  --use-clip-classifier \
  --label-parse llm \
  --max-frames 16
```

Offline re-parse test (no VLM inference; uses existing checkpoint `raw` fields):

```bash
python -m rag_experiment.test_llm_parse --n 20
```

Requires `.env` with `DASHSCOPE_API_KEY` and `QWEN_BASE_URL`.
Ablation without classifier text:

```bash
python run_rag_experiment.py --no-use-clip-classifier ...
```

Uniform frames (legacy):

```bash
python run_rag_experiment.py --frame-sample uniform ...
```
