# Alignment with X-VARS Paper (Held et al., CVPRW 2024)

## Matched (defaults after paper-align fix)

| Aspect | Paper / original X-VARS | This repo (`rag_experiment/runner.py`) |
|--------|-------------------------|----------------------------------------|
| Primary task | Offence + Severity 4-way | Same |
| Frames | 63–87 @ 25fps → 16 @ 17fps | `--frame-sample paper_xvars` |
| Video tokens | 44 temporal + 256 spatial = **300** | `--paper-features` + `video_token_len=300` |
| Conv template | Football referee assistant | `--conv-mode xvars_football_v1` |
| Questions | Foul and card **separately** | `--split-questions` (default on) |
| Classifier text | CLIP P_foul / P_sev in prompt | `--use-clip-classifier` |
| Parse | ChatGPT-3.5 on free-form text | `--label-parse llm` (DashScope Qwen) |

## Run (paper-aligned)

```bash
python run_rag_experiment.py \
  --output-dir rag_experiment_runs/paper_align \
  --lines R0,R1,R2,R3,R4 \
  --label-parse llm \
  --unit action --paper-protocol \
  --prefer-main-camera \
  --frame-sample paper_xvars \
  --paper-features \
  --split-questions \
  --conv-mode xvars_football_v1 \
  --use-clip-classifier
```

Legacy (pre-fix) behaviour: `--no-paper-features --frame-sample foul_centered --video-token-len 200 --no-split-questions --conv-mode video-chatgpt_v1`

## Remaining differences

1. No offline `PRE_CLIP_feature_clip_*.pkl` in `dataset/` — features are computed online with the same math as `create_features.py`.
2. Training used ground-truth classifier labels; inference uses CLIP predictions (as in paper Table 4 eval).
3. Human study / BLEU metrics not in RAG ablation runner.
