"""Generate per-experiment YAML config matrices.

Usage:
    python scripts/generate_configs.py --experiment exp1
    python scripts/generate_configs.py --experiment exp2
    python scripts/generate_configs.py --experiment exp3
    python scripts/generate_configs.py --experiment all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RunConfig, dump_yaml


# Per-task hyperparameter defaults shared across all experiments.
# Eval subsampling chosen to keep eval wall-clock under ~10 min per run.
TASK_DEFAULTS = {
    "sst2": dict(
        max_steps=1000,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        max_seq_length=128,
        eval_batch_size=16,
        eval_max_examples=None,  # full 872-example val
    ),
    "hellaswag": dict(
        max_steps=2000,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_seq_length=256,
        eval_batch_size=8,
        eval_max_examples=2000,  # subsample of 10k to keep eval fast
    ),
    "gsm8k": dict(
        max_steps=3000,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        max_seq_length=512,
        eval_batch_size=1,
        eval_max_examples=500,  # subsample of 1.32k; greedy decode is the bottleneck
    ),
}

LR_BY_METHOD = {
    "lora": 2e-4,
    "ia3": 3e-3,        # IA3 uses larger LR per the paper
    "prefix": 1e-3,
    "full": 1e-5,
}

SEEDS = [0, 1, 2]
TASKS = ["sst2", "hellaswag", "gsm8k"]
RANKS = [4, 8, 16, 32, 64]


def _base_kwargs(task: str, peft_method: str) -> dict:
    td = TASK_DEFAULTS[task]
    return dict(
        task=task,
        peft_method=peft_method,
        learning_rate=LR_BY_METHOD[peft_method],
        **td,
    )


def _task_dir(out_root: Path, experiment: str, task: str) -> Path:
    d = out_root / experiment / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def gen_exp1(out_root: Path) -> list[Path]:
    """LoRA rank sweep: 5 ranks × 3 tasks × 3 seeds + full-FT baselines."""
    paths = []
    for task in TASKS:
        out_dir = _task_dir(out_root, "exp1", task)
        for seed in SEEDS:
            for r in RANKS:
                run_id = f"exp1__lora_r{r}__{task}__seed{seed}"
                cfg = RunConfig(
                    run_id=run_id,
                    experiment="exp1",
                    seed=seed,
                    precision="bf16",
                    lora_rank=r,
                    **_base_kwargs(task, "lora"),
                )
                p = out_dir / f"{run_id}.yaml"
                dump_yaml(cfg, p)
                paths.append(p)
            run_id = f"exp1__full__{task}__seed{seed}"
            cfg = RunConfig(
                run_id=run_id,
                experiment="exp1",
                seed=seed,
                precision="bf16",
                **_base_kwargs(task, "full"),
            )
            p = out_dir / f"{run_id}.yaml"
            dump_yaml(cfg, p)
            paths.append(p)
    return paths


def gen_exp2(out_root: Path) -> list[Path]:
    """Precision regime: 3 precisions × 3 tasks × 3 seeds, fixed LoRA r=16.

    BF16/r=16 cells are reused from exp1 (same training run), so we only
    generate fp32 and int4 here. analyze_exp2 stitches in the bf16 rows
    from runs.csv.
    """
    paths = []
    for task in TASKS:
        out_dir = _task_dir(out_root, "exp2", task)
        for seed in SEEDS:
            for prec in ["fp32", "int4"]:
                run_id = f"exp2__lora_r16_{prec}__{task}__seed{seed}"
                kw = _base_kwargs(task, "lora")
                if prec == "fp32":
                    kw["per_device_train_batch_size"] = max(1, kw["per_device_train_batch_size"] // 2)
                    kw["gradient_accumulation_steps"] = kw["gradient_accumulation_steps"] * 2
                cfg = RunConfig(
                    run_id=run_id,
                    experiment="exp2",
                    seed=seed,
                    precision=prec,
                    lora_rank=16,
                    **kw,
                )
                p = out_dir / f"{run_id}.yaml"
                dump_yaml(cfg, p)
                paths.append(p)
    return paths


def gen_exp3(out_root: Path) -> list[Path]:
    """PEFT method comparison: 4 methods × 3 tasks × 3 seeds.

    LoRA r=16 (bf16) and full FT (bf16) cells are reused from exp1, so we
    only generate IA3 and prefix-tuning here. analyze_exp3 stitches in the
    lora_r16 and full rows from runs.csv.
    """
    paths = []
    method_specs = [
        ("ia3", dict()),
        ("prefix", dict(prefix_num_virtual_tokens=20)),
    ]
    for task in TASKS:
        out_dir = _task_dir(out_root, "exp3", task)
        for seed in SEEDS:
            for method, extras in method_specs:
                tag = "prefix20" if method == "prefix" else method
                run_id = f"exp3__{tag}__{task}__seed{seed}"
                cfg = RunConfig(
                    run_id=run_id,
                    experiment="exp3",
                    seed=seed,
                    precision="bf16",
                    **_base_kwargs(task, method),
                    **extras,
                )
                p = out_dir / f"{run_id}.yaml"
                dump_yaml(cfg, p)
                paths.append(p)
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=["exp1", "exp2", "exp3", "all"])
    p.add_argument("--out", default="configs", help="output directory root")
    args = p.parse_args()

    out_root = Path(args.out)
    targets = {
        "exp1": gen_exp1,
        "exp2": gen_exp2,
        "exp3": gen_exp3,
    }
    todo = list(targets.values()) if args.experiment == "all" else [targets[args.experiment]]
    total = 0
    for fn in todo:
        paths = fn(out_root)
        print(f"[{fn.__name__}] generated {len(paths)} configs")
        total += len(paths)
    print(f"[done] {total} configs total under {out_root}/")


if __name__ == "__main__":
    main()
