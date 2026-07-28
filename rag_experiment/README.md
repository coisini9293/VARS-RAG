# RAG Experiment (Backend Only)

Dedicated ablation suite to measure **whether RAG helps X-VARS** on the Test set.  
No Streamlit UI. Progress bar in the terminal. Figures and report text are in **English**.

**Default evaluation follows the X-VARS paper (CVPRW 2024) 4-way Offence+Severity protocol**  
(Accuracy + Balanced Accuracy; one Main-camera view per action). Details: [`PAPER_ALIGNMENT.md`](PAPER_ALIGNMENT.md).

## Why this folder exists

The older `experiment/` package mixes RAG with Prompt / CoT / Self-consistency.  
This package focuses only on RAG effect, with a cleaner design:

| Issue in old mixed runner | Fix here |
|---------------------------|----------|
| Same fixed English task sentence used as retrieval query for every clip | Explicit `fixed` vs `adaptive` query modes |
| No retrieval diagnostics | Logs hits, empty rate, Top1 score, context length |
| Re-encodes video features per line | Shared feature cache across R0–R4 |
| Metrics JSON only | English plots + `REPORT.md` analysis |
| Missing videos can crash a whole run | Missing clips are skipped (or recorded as errors) |

Also fixed in `FootballRAGSystem._build_index`: saving `embeddings` when the encoder is unavailable no longer references an undefined local variable.

## Experiment lines

| Line | Name | What changes |
|------|------|----------------|
| **R0** | Baseline | No RAG (control = original X-VARS text path) |
| **R1** | RAG-Base | Production-like: search Top3, merge Top1, truncate ~200 chars, fixed query |
| **R2** | RAG-TopK | Merge Top3 with per-chunk truncate, fixed query |
| **R3** | RAG-Deep | Merge Top3 full chunks, longer budget, fixed query |
| **R4** | RAG-Adaptive | Same merge as R3; query adds Action class / Contact / Bodypart (**never Offence**) |

All lines share the same classification prompt and decoding settings so the comparison isolates RAG.

## Paper-aligned task (default)

| Item | Setting |
|------|---------|
| Labels | `No offence` / `Offence + No card` / `Offence + Yellow card` / `Offence + Red card` |
| Severity map | `1.0→No card`, `3.0→Yellow`, `5.0→Red` |
| Skipped | `Between`; severity `2.0`/`4.0`/empty |
| Unit | `--unit action` (≈248 Test actions after skip; not 706 multi-view clips) |
| Camera | Prefer `Main camera center` |
| Metrics | Accuracy, **Balanced Accuracy**, Macro-F1, parse rate |
| Frames | `--frame-sample foul_centered`：犯规前后各 8 帧（中心默认 3.0s） |
| Questions | 论文原问：`Is it a foul or not? Why?` + `What card would you give? Why?` |
| Classifier text | `--use-clip-classifier`：在问题前注入 `14_model.pth.tar` 的 P_foul/P_sev 文本 |

## How to run

From the **project root** (conda env `asd` recommended):

```bash
conda activate asd
python run_rag_experiment.py --output-dir rag_experiment_runs/run1 --lines R0,R1,R2,R3,R4
```

Or:

```bash
python -m rag_experiment --output-dir rag_experiment_runs/run1 --lines R0,R1,R3
```

### Useful arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `--dataset-root` | `dataset/Test` | Video root |
| `--annotations` | `dataset/Test/annotations.json` | Labels |
| `--max-action` | `300` | Max action id (inclusive scan) |
| `--lines` | `R0,R1,R2,R3,R4` | Which lines to run |
| `--output-dir` | `rag_experiment_runs/default` | Run root (resume-safe) |
| `--video-token-len` | `200` | Visual tokens |
| `--max-frames` | `16` | Frames per clip (paper uses 16) |
| `--frame-sample` | `foul_centered` | `foul_centered` (8+8) or `uniform` |
| `--foul-time-sec` | `3.0` | In-clip foul center (create_features convention) |
| `--use-clip-classifier` / `--no-use-clip-classifier` | on | Inject CLIP prediction text into LLM prompt |
| `--clip-classifier-weights` | `visual_encoder/14_model.pth.tar` | Classifier checkpoint |
| `--unit` | `action` | `action`=one view/action; `clip`=all views |
| `--paper-protocol` / `--no-paper-protocol` | on | 4-way labels vs legacy Offence-only |
| `--prefer-main-camera` | on | Prefer Main camera center when `unit=action` |
| `--seed` | `42` | Base seed |
| `--baseline` | `R0` | Baseline for delta plots |

Same `--output-dir` supports **resume** via per-line `checkpoint.json`.

### Quick smoke (few actions)

```bash
python run_rag_experiment.py --output-dir rag_experiment_runs/smoke --lines R0,R1 --max-action 2
```

## Outputs

```text
rag_experiment_runs/<run>/
  R0/ … R4/
    experiment_settings.json
    checkpoint.json
    logs/predictions.jsonl      # y_true, y_pred, raw, retrieval hits
    plots/confusion_matrix.png  # English labels
    plots/metrics_line.json
    plots/retrieval_stats.json  # RAG lines only
  summary/
    metrics.json
    delta_vs_baseline.json
    analysis.json
    REPORT.md                   # findings + recommendations
    plots/metrics_by_line.png
    plots/delta_accuracy_vs_baseline.png
    plots/retrieval_empty_rate.png
```

## How to read results

1. Open `summary/REPORT.md` for automatic findings.
2. Check `delta_accuracy_vs_baseline.png`: green = better than R0.
3. If RAG lines have high **empty retrieval rate**, fix `vector_store/` / encoder first — accuracy deltas are not meaningful until retrieval works.
4. Inspect wrong cases in `logs/predictions.jsonl` (`retrieval.hits` vs `raw`).

## Design notes

- **No label leakage**: Offence ground truth is never put into the retrieval query.
- **Fair prompt**: Every line uses the same paper-style questions; only the “Reference rules” block differs.
- **Infrastructure reuse**: Model load / CLIP encode / label parse come from `experiment/` (shared plumbing). RAG presets, queries, report, and runner live only here.
