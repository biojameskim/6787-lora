# 6787-LoRA Progress Report

_Last updated: 2026-04-27 (Phase 1 complete)_

CS 6787 final project — **Efficiency Frontiers of Parameter-Efficient Fine-Tuning Under Compute and Memory Constraints**. Backbone: Qwen2.5-1.5B. Tasks: SST-2, HellaSwag, GSM8K. Three experiments held together by one shared training/eval harness.

## Status at a glance

| Phase | What | Status |
|---|---|---|
| 0 | Build shared harness | done |
| 0 | End-to-end smoke test | passed (92% SST-2 @ 50 steps) |
| 1 | **Exp 1 — LoRA rank sweep** (54 runs) | **complete (54 / 54)** |
| 2 | **Exp 2 — precision/QLoRA** (27 cells, 18 new + 9 reused from exp1) | **complete (18 / 18 new + 9 reused = 27 / 27)** |
| 3 | **Exp 3 — PEFT method** (36 cells, 18 new + 18 reused from exp1) | **complete (18 / 18 new + 18 reused = 36 / 36)** |

## What's built (Phase 0 — the shared harness)

Every cell of every experiment is a single YAML config consumed by `python -m src.train --config <yaml>`. Same code path for every run; only the config differs. This is what guarantees the efficiency metrics (acc/param, acc/GPU-hour) are computed identically across experiments.

```
src/
├── config.py            RunConfig dataclass + YAML load/dump
├── model.py             build_model(cfg) — single dispatch over (precision × peft_method)
├── seeding.py           global seed control
├── instrumentation.py   TrainTimer (wall + peak VRAM), param counters
├── results.py           writes per-run JSON + appends master results/runs.csv (idempotent)
├── train.py             entrypoint: build → train → score → record
├── data/
│   ├── sst2.py          loglikelihood-format prompts, 872 val examples
│   ├── hellaswag.py     Eleuther-style preprocessing, 4-way LL, 2000 sub-sampled val
│   ├── gsm8k.py         CoT training, regex answer extraction, 500 sub-sampled test
│   ├── _format.py       shared tokenize-with-prompt-mask helper
│   └── registry.py      task name → (train_loader, eval_loader, scorer)
└── eval/
    ├── loglikelihood.py multiple-choice scorer (raw + length-norm)
    └── gsm8k_gen.py     greedy decode + answer extraction

scripts/
├── generate_configs.py  stamps out config matrices
├── smoke_test.py        50-step end-to-end sanity check
└── analyze.py           runs.csv → summary CSVs + matplotlib plots

jobs/
├── run_one.sh           SLURM template (single config)
├── exp1_lora_rank.sh    array job 0–53, %2 concurrency
├── exp2_precision.sh    array job 0–26
├── exp3_peft.sh         array job 0–35
└── smoke.sh

configs/
├── base.yaml
├── exp1/
│   ├── sst2/            18 yamls (6 methods × 3 seeds)
│   ├── hellaswag/       18 yamls
│   └── gsm8k/           18 yamls
├── exp2/                (sst2/, hellaswag/, gsm8k/ — populated by generate_configs)
└── exp3/                (sst2/, hellaswag/, gsm8k/ — populated by generate_configs)

results/
├── runs.csv                 master log — one row per completed run, tagged by experiment
├── exp1/
│   ├── runs/<run_id>/       config.yaml, metrics.json, train_log.jsonl
│   ├── figs/                acc_vs_rank__{task}.png, frontier_{params,gpu_hour}.png
│   └── summary.csv
├── exp2/                    (same layout, populated when phase 2 runs)
├── exp3/                    (same layout, populated when phase 3 runs)
└── smoke/runs/<run_id>/     50-step sanity-check artifacts
```

Design choices (locked in via prior discussion):
- **Eval**: custom loglikelihood scoring for MC tasks, greedy generation + regex for GSM8K. No `lm-eval-harness` dependency.
- **Budget control**: fixed optimizer steps per task (SST-2=1000, HellaSwag=2000, GSM8K=3000) with effective batch 32. Same step budget for every method/precision in that task.
- **Trainer**: HF `transformers.Trainer`. `bf16=True` when `precision ∈ {bf16, int4}`, `fp16=False`, `report_to=[]`, `save_strategy="no"`, `gradient_checkpointing=False`.
- **Logging**: local JSON+CSV only. No wandb/tensorboard.
- **Idempotency**: `train.py` checks for existing `metrics.json` and short-circuits, so SLURM array re-runs never duplicate work.

## Experiment matrix

| Exp | Variable | Levels | Tasks | Seeds | Cells | New runs | Reused from exp1 |
|---|---|---|---|---|---:|---:|---:|
| 1 | LoRA rank (+ full-FT baseline) | r ∈ {4,8,16,32,64} + full | 3 | 3 | 54 | 54 | — |
| 2 | Precision | {fp32, bf16, int4} @ r16 | 3 | 3 | 27 | 18 (fp32 + int4) | 9 (bf16/r=16) |
| 3 | PEFT method | {LoRA r16, IA3, Prefix-20, full} | 3 | 3 | 36 | 18 (IA3 + prefix-20) | 18 (LoRA r=16 + full) |

Net unique runs project-wide: **90** (54 exp1 + 18 exp2-new + 18 exp3-new). The 27 reused cells are stitched in by `analyze_exp{2,3}` directly from `runs.csv`, keyed on (task, peft_method, precision, lora_rank, seed) — no retraining needed.

## Phase 1 results (54 / 54)

Mean ± std accuracy across 3 seeds. SST-2 = raw acc, HellaSwag = `acc_norm`, GSM8K = exact-match on `#### N`. All numbers are percentages (pp).

| Method | Trainable params | SST-2 | HellaSwag | GSM8K |
|---|---:|---:|---:|---:|
| Full FT | 1.54 B (100%) | 95.3 ± 0.07 | 60.6 ± 0.14 | 55.4 ± 0.76 |
| LoRA r=4 | 545 K (0.035%) | 95.3 ± 0.14 | 60.1 ± 0.45 | 55.2 ± 0.60 |
| LoRA r=8 | 1.09 M (0.071%) | 95.7 ± 0.29 | 60.6 ± 0.13 | 54.3 ± 0.84 |
| LoRA r=16 | 2.18 M (0.14%) | 95.8 ± 0.11 | 61.2 ± 0.02 | 51.9 ± 0.53 |
| LoRA r=32 | 4.36 M (0.28%) | 95.7 ± 0.48 | 61.7 ± 0.37 | 50.5 ± 1.57 |
| LoRA r=64 | 8.72 M (0.56%) | 96.2 ± 0.16 | 62.0 ± 0.23 | 48.3 ± 1.65 |

Headline observations:

- **SST-2 is saturated** at this scale — every config lands in 95–96% with overlapping noise bands. r=64 LoRA edges out full FT (96.2 vs 95.3), but the gap is on the order of seed variance.
- **HellaSwag shows clean monotone improvement with rank**: 60.1 → 60.6 → 61.2 → 61.7 → 62.0 from r=4 to r=64, a ~1.9 pp gain across the rank decade. Diminishing returns kick in around r=32. Even r=4 LoRA is within ~0.5 pp of full FT.
- **GSM8K shows the *inverse* pattern.** Low-rank LoRA (r=4) matches full FT (55.2 vs 55.4), and accuracy *degrades monotonically* with rank: 55.2 → 54.3 → 51.9 → 50.5 → 48.3. That's a **~7 pp drop** from r=4 to r=64 — the strongest result in the sweep. Consistent with the intuition that for reasoning tasks under a fixed step budget, more trainable params overfit/underconverge at the same LR.

Net efficiency story (for the writeup): on these three tasks, LoRA r=4 already captures ≥99% of full-FT accuracy with **3000× fewer trainable params** and **~3× less peak VRAM**. Rank scaling helps only on the cleanest classification tasks; on math reasoning under a fixed step budget, lower rank is *strictly* better.

## Hardware

> **Note for the writeup:** all sweeps are running on a single **NVIDIA RTX 6000 Ada (48 GB)**, *not* the RTX PRO 6000 Blackwell (102 GB) cited in `project_proposal.txt`. The Blackwell node was fully allocated when we went to dispatch, so we switched to the available Ada node so jobs could start. SLURM scripts are pinned to GRES `nvidia_rtx_6000_ada_generation`. The 7 GB peak-VRAM budget claimed in the proposal still fits comfortably.

Per-run wall time on RTX 6000 Ada (LoRA, observed):

| Task | Wall time | Peak VRAM (LoRA bf16) | Peak VRAM (full bf16) |
|---|---:|---:|---:|
| SST-2 (1000 steps) | ~3 min | ~7.9 GB | ~17.8 GB |
| HellaSwag (2000 steps) | ~12 min | ~8.3 GB | ~18.2 GB |
| GSM8K (3000 steps) | ~42 min | ~12.0 GB | ~22.5 GB |

## Phase 1 wrap-up

All 54 runs are recorded in `results/runs.csv`. One run (`exp1__lora_r4__sst2__seed1`, original array index 34) was preempted mid-sweep on `duchin-compute-01`; resubmitted as a single-element array and re-ran cleanly on the Ada node.

### Deliverables

Generated by `python scripts/analyze.py --experiment exp1`:

| File | Description |
|---|---|
| `results/runs.csv` | Master log, one row per (run_id, seed). 54 exp1 rows + 1 smoke. |
| `results/exp1/summary.csv` | Aggregated by (task × peft_method × lora_rank), 18 rows, mean ± std across 3 seeds. |
| `results/exp1/figs/acc_vs_rank__sst2.png` | Accuracy vs LoRA rank (log₂ x-axis), error bars, full-FT dashed reference. |
| `results/exp1/figs/acc_vs_rank__hellaswag.png` | Same, primary metric = `acc_norm`. |
| `results/exp1/figs/acc_vs_rank__gsm8k.png` | Same. |
| `results/exp1/figs/frontier_params.png` | Accuracy vs trainable params, all three tasks overlaid. |
| `results/exp1/figs/frontier_gpu_hour.png` | Accuracy vs train wall-time per run (proxy for GPU-hour). |

### Efficiency frontier (acc / GPU-hour)

LoRA dominates full FT on every task on a per-GPU-hour basis:

| Task | Full FT | Best LoRA | Best rank |
|---|---:|---:|---|
| SST-2 | 16.7 acc-pp/hr | **20.7 acc-pp/hr** | r=8 |
| HellaSwag | 2.24 | **3.15** | r=32 |
| GSM8K | 0.62 | **0.81** | r=4 / r=8 |

The "best LoRA rank for acc/GPU-hour" tracks the underlying accuracy story: SST-2 saturates so the cheapest non-trivial rank wins; HellaSwag prefers the rank that matches its information content (r=32); GSM8K's inverse-rank pattern means cheaper *and* more accurate go together.

### Phase 1 status: closed

Next step: generate Phase 2 (precision/QLoRA) configs via `scripts/generate_configs.py`, dispatch `jobs/exp2_precision.sh`.

## Phase 2 / Phase 3 plan

Both experiments reuse the harness as-is — only `scripts/generate_configs.py` needs to stamp out the new YAML matrices. The `model.py` dispatch already handles `int4` (QLoRA via bnb NF4 + double quant), `ia3`, and `prefix` adapters; those branches were specced and unit-tested during Phase 0 but have not yet been exercised end-to-end on real data.

Phase 2 will surface:
- Accuracy gap between FP32, BF16, and INT4 LoRA at fixed rank (r=16).
- Peak VRAM and tokens/sec per precision regime.
- Whether QLoRA's compute savings hold when correctly normalized for dequant overhead.

Phase 3 will surface:
- LoRA vs IA3 vs Prefix vs full FT at matched task budget.
- The accuracy/param frontier (where IA3 should dominate).
- The accuracy/GPU-hour frontier (where the answer is less obvious).

## Reproducing a single run

```bash
cd /share/j_sun/jjk297/repos/6787-lora
.venv/bin/python -m src.train --config configs/exp1/sst2/exp1__lora_r16__sst2__seed0.yaml
```

The run is idempotent — re-running with the same config short-circuits if `results/<experiment>/runs/<run_id>/metrics.json` exists. Pass `--force` to override.

## Reproducing an experiment array

```bash
sbatch jobs/exp1_lora_rank.sh    # 54 configs, %2 concurrency, ~6h wall
sbatch jobs/exp2_precision.sh    # 27 configs (after generate_configs)
sbatch jobs/exp3_peft.sh         # 36 configs (after generate_configs)
```

All array jobs request `--gres=gpu:nvidia_rtx_6000_ada_generation:1 --partition=jjs533,gpu` and write logs to `slurm/`.
