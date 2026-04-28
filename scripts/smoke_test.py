"""End-to-end smoke test for the harness.

Runs a 50-step LoRA r=4 BF16 SST-2 training + eval and asserts:
- metrics.json was written
- accuracy is meaningfully above random (acc > 0.65)

Idempotency check: re-running without --force should skip.
Untrained sanity (--zero-steps): max_steps=0, accuracy should land near random (~0.50).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/smoke_test.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RunConfig
from src.results import metrics_path
from src.train import run


def make_cfg(zero_steps: bool = False) -> RunConfig:
    suffix = "zero" if zero_steps else "trained"
    return RunConfig(
        run_id=f"smoke__sst2_lora_r4_{suffix}",
        experiment="smoke",
        seed=0,
        peft_method="lora",
        task="sst2",
        precision="bf16",
        max_steps=0 if zero_steps else 50,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_seq_length=128,
        eval_batch_size=8,
        lora_rank=4,
        eval_max_examples=200,  # small to keep runtime <5 min
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-steps", action="store_true", help="run with max_steps=0 to check eval at init")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = make_cfg(zero_steps=args.zero_steps)
    metrics = run(cfg, force=args.force)
    if metrics is None:
        # idempotent skip — load and print existing
        with open(metrics_path(cfg.run_id)) as f:
            metrics = json.load(f)
        print("[smoke] (skipped — already completed)")

    acc = metrics["task"].get("acc", 0.0)
    print(f"[smoke] task acc = {acc:.4f}")
    if args.zero_steps:
        # Random for SST-2 is 0.5; without training the model may be biased but should be ~50%.
        assert 0.3 < acc < 0.7, f"untrained acc looks suspicious: {acc:.4f}"
        print("[smoke] PASS — untrained accuracy in random-ish range")
    else:
        assert acc > 0.65, f"trained smoke acc too low: {acc:.4f}"
        print("[smoke] PASS — trained accuracy > 0.65")


if __name__ == "__main__":
    main()
