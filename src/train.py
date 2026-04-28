from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .config import RunConfig, load_yaml
from .data.registry import get_task
from .eval.gsm8k_gen import score_gsm8k_generation
from .eval.loglikelihood import score_multiple_choice
from .instrumentation import TrainTimer, count_params
from .model import build_model
from .results import already_completed, write_run_outputs
from .seeding import set_global_seed


def _training_args(cfg: RunConfig) -> TrainingArguments:
    out_dir = Path("results") / "runs" / cfg.run_id / "trainer_out"
    return TrainingArguments(
        output_dir=str(out_dir),
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=cfg.seed,
        bf16=cfg.precision in ("bf16", "int4"),
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        # Disable gradient checkpointing — prelim showed plenty of headroom for 1.5B at our seqlen.
        gradient_checkpointing=False,
    )


def _run_eval(model, tokenizer, cfg: RunConfig) -> dict:
    task_mod = get_task(cfg.task)
    examples = task_mod.build_eval_examples(cfg)
    if cfg.task == "gsm8k":
        return score_gsm8k_generation(model, tokenizer, examples)
    return score_multiple_choice(model, tokenizer, examples)


def _primary_acc(cfg: RunConfig, task_metrics: dict) -> float:
    task_mod = get_task(cfg.task)
    return float(task_metrics[task_mod.PRIMARY_METRIC])


def _train_tokens(train_dataset, cfg: RunConfig) -> int:
    """Approximate tokens seen by the trainer (for throughput accounting)."""
    eff_bs = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    avg_seq = sum(len(x["input_ids"]) for x in train_dataset) / len(train_dataset)
    return int(cfg.max_steps * eff_bs * avg_seq)


def run(cfg: RunConfig, force: bool = False) -> dict | None:
    if already_completed(cfg.run_id) and not force:
        print(f"[skip] {cfg.run_id}: metrics.json already exists. Use --force to re-run.")
        return None

    set_global_seed(cfg.seed)
    print(f"[run_id] {cfg.run_id}")
    print(f"[config] {json.dumps(cfg.to_dict(), default=str)}")

    model, tokenizer = build_model(cfg)
    params = count_params(model)
    print(f"[params] trainable={params['trainable_params']:,} / total={params['total_params']:,} ({params['trainable_pct']:.4f}%)")

    task_mod = get_task(cfg.task)
    train_ds = task_mod.build_train_dataset(tokenizer, cfg)
    print(f"[data] {cfg.task} train rows: {len(train_ds):,}")

    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding=True)

    trainer = Trainer(
        model=model,
        args=_training_args(cfg),
        train_dataset=train_ds,
        data_collator=collator,
    )

    with TrainTimer() as t:
        trainer.train()
    train_seconds = t.elapsed
    train_peak_vram = t.peak_vram_gb
    train_log = list(trainer.state.log_history)
    train_tokens = _train_tokens(train_ds, cfg)
    train_tps = train_tokens / max(train_seconds, 1e-6)
    print(f"[train] elapsed={train_seconds:.1f}s peak_vram={train_peak_vram:.2f} GB tokens/s={train_tps:.0f}")

    # Eval (uses peak VRAM stats reset to capture eval-only memory if we want it later;
    # for now we report training peak as the headline number).
    eval_start = time.perf_counter()
    task_metrics = _run_eval(model, tokenizer, cfg)
    eval_seconds = time.perf_counter() - eval_start
    print(f"[eval] {task_metrics}")

    primary_acc = _primary_acc(cfg, task_metrics)
    metrics = {
        "task": task_metrics,
        "params": params,
        "hardware": {
            "peak_vram_gb": train_peak_vram,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "train_tokens_per_sec": train_tps,
        },
        "efficiency": {
            "primary_metric": task_mod.PRIMARY_METRIC,
            "primary_acc": primary_acc,
            "acc_per_param": primary_acc / max(params["trainable_params"], 1),
            "acc_per_gpu_hour": primary_acc / max(train_seconds / 3600.0, 1e-9),
        },
    }
    write_run_outputs(cfg, metrics, train_log)
    print(f"[done] wrote results/runs/{cfg.run_id}/")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to a RunConfig YAML")
    p.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    run(cfg, force=args.force)


if __name__ == "__main__":
    main()
