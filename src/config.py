from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RunConfig:
    run_id: str
    experiment: str
    seed: int

    # PEFT method
    peft_method: str  # "full" | "lora" | "ia3" | "prefix"
    task: str  # "sst2" | "hellaswag" | "gsm8k"
    precision: str  # "fp32" | "bf16" | "int4"

    # Training
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_seq_length: int
    eval_batch_size: int

    # Model
    model_id: str = "Qwen/Qwen2.5-1.5B"

    # PEFT-specific (only the relevant ones are used per method)
    lora_rank: Optional[int] = None
    lora_alpha_mult: int = 2
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj")
    prefix_num_virtual_tokens: Optional[int] = None
    ia3_target_modules: tuple = ("k_proj", "v_proj", "down_proj")
    ia3_feedforward_modules: tuple = ("down_proj",)

    # Training extras
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    eval_max_examples: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # YAML doesn't love tuples; convert to lists
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


def load_yaml(path: str | Path) -> RunConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # back-convert lists to tuples for the tuple fields
    for k in ("lora_target_modules", "ia3_target_modules", "ia3_feedforward_modules"):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return RunConfig(**d)


def dump_yaml(cfg: RunConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
