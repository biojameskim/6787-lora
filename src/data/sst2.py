from __future__ import annotations

from datasets import Dataset, load_dataset

from ..config import RunConfig
from ._format import tokenize_with_completion


LABEL_TOKENS = {0: " negative", 1: " positive"}


def _format_prompt(sentence: str) -> str:
    return f"Sentence: {sentence.strip()}\nSentiment:"


def build_train_dataset(tokenizer, cfg: RunConfig) -> Dataset:
    raw = load_dataset("glue", "sst2", split="train")
    rows = []
    for ex in raw:
        prompt = _format_prompt(ex["sentence"])
        completion = LABEL_TOKENS[int(ex["label"])]
        out = tokenize_with_completion(tokenizer, prompt, completion, cfg.max_seq_length)
        if out is not None:
            rows.append(out)
    return Dataset.from_list(rows)


def build_eval_examples(cfg: RunConfig) -> list[dict]:
    raw = load_dataset("glue", "sst2", split="validation")
    out = []
    for ex in raw:
        out.append({
            "prompt": _format_prompt(ex["sentence"]),
            "choices": [LABEL_TOKENS[0], LABEL_TOKENS[1]],
            "label": int(ex["label"]),
        })
    if cfg.eval_max_examples is not None:
        out = out[: cfg.eval_max_examples]
    return out


PRIMARY_METRIC = "acc"
