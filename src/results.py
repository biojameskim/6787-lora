from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import RunConfig


RESULTS_DIR = Path("results")
RUNS_CSV = RESULTS_DIR / "runs.csv"


def _experiment_from_run_id(run_id: str) -> str:
    return run_id.split("__", 1)[0]


def experiment_dir(experiment: str) -> Path:
    return RESULTS_DIR / experiment

CSV_COLUMNS = [
    "run_id", "experiment", "task", "peft_method", "precision", "lora_rank",
    "prefix_num_virtual_tokens", "seed",
    "max_steps", "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
    "max_seq_length",
    "task_acc", "task_acc_norm", "eval_n",
    "trainable_params", "total_params", "trainable_pct",
    "peak_vram_gb", "train_seconds", "eval_seconds",
    "train_tokens_per_sec",
    "acc_per_param", "acc_per_gpu_hour",
]


def run_dir(run_id: str) -> Path:
    return experiment_dir(_experiment_from_run_id(run_id)) / "runs" / run_id


def metrics_path(run_id: str) -> Path:
    return run_dir(run_id) / "metrics.json"


def already_completed(run_id: str) -> bool:
    return metrics_path(run_id).exists()


def write_run_outputs(cfg: RunConfig, metrics: dict, train_log: list[dict]) -> None:
    out_dir = run_dir(cfg.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-run config snapshot (for full reproducibility from the run dir alone).
    from .config import dump_yaml
    dump_yaml(cfg, out_dir / "config.yaml")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "train_log.jsonl", "w") as f:
        for row in train_log:
            f.write(json.dumps(row) + "\n")

    _append_runs_csv(cfg, metrics)


def _append_runs_csv(cfg: RunConfig, metrics: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not RUNS_CSV.exists()
    row = {
        "run_id": cfg.run_id,
        "experiment": cfg.experiment,
        "task": cfg.task,
        "peft_method": cfg.peft_method,
        "precision": cfg.precision,
        "lora_rank": cfg.lora_rank if cfg.lora_rank is not None else "",
        "prefix_num_virtual_tokens": cfg.prefix_num_virtual_tokens if cfg.prefix_num_virtual_tokens is not None else "",
        "seed": cfg.seed,
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_seq_length": cfg.max_seq_length,
        "task_acc": metrics["task"].get("acc", ""),
        "task_acc_norm": metrics["task"].get("acc_norm", ""),
        "eval_n": metrics["task"].get("n", ""),
        "trainable_params": metrics["params"]["trainable_params"],
        "total_params": metrics["params"]["total_params"],
        "trainable_pct": metrics["params"]["trainable_pct"],
        "peak_vram_gb": metrics["hardware"]["peak_vram_gb"],
        "train_seconds": metrics["hardware"]["train_seconds"],
        "eval_seconds": metrics["hardware"]["eval_seconds"],
        "train_tokens_per_sec": metrics["hardware"]["train_tokens_per_sec"],
        "acc_per_param": metrics["efficiency"]["acc_per_param"],
        "acc_per_gpu_hour": metrics["efficiency"]["acc_per_gpu_hour"],
    }
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow(row)
