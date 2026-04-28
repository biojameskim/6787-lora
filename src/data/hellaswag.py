from __future__ import annotations

import re

from datasets import Dataset, load_dataset

from ..config import RunConfig
from ._format import tokenize_with_completion


# Per Eleuther lm-eval-harness: lightly clean the activity label and ctx for HellaSwag.
def _preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _format_prompt(activity_label: str, ctx: str) -> str:
    return _preprocess(f"{activity_label}: {ctx}")


def build_train_dataset(tokenizer, cfg: RunConfig) -> Dataset:
    raw = load_dataset("hellaswag", split="train", trust_remote_code=True)
    rows = []
    for ex in raw:
        label = int(ex["label"])
        prompt = _format_prompt(ex["activity_label"], ex["ctx"])
        completion = " " + _preprocess(ex["endings"][label])
        out = tokenize_with_completion(tokenizer, prompt, completion, cfg.max_seq_length)
        if out is not None:
            rows.append(out)
    return Dataset.from_list(rows)


def build_eval_examples(cfg: RunConfig) -> list[dict]:
    raw = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    out = []
    for ex in raw:
        prompt = _format_prompt(ex["activity_label"], ex["ctx"])
        choices = [" " + _preprocess(e) for e in ex["endings"]]
        out.append({
            "prompt": prompt,
            "choices": choices,
            "label": int(ex["label"]),
        })
    if cfg.eval_max_examples is not None:
        out = out[: cfg.eval_max_examples]
    return out


PRIMARY_METRIC = "acc_norm"
