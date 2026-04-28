from __future__ import annotations

import re

from datasets import Dataset, load_dataset

from ..config import RunConfig
from ._format import tokenize_with_completion


ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")


def _format_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def extract_answer(text: str) -> str | None:
    """Pull the last `#### N` from a generation. Strips commas; preserves sign and decimals."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def build_train_dataset(tokenizer, cfg: RunConfig) -> Dataset:
    raw = load_dataset("gsm8k", "main", split="train")
    rows = []
    for ex in raw:
        prompt = _format_prompt(ex["question"])
        # Keep the chain-of-thought + #### N marker so the model learns the answer convention.
        completion = " " + ex["answer"].strip()
        out = tokenize_with_completion(tokenizer, prompt, completion, cfg.max_seq_length)
        if out is not None:
            rows.append(out)
    return Dataset.from_list(rows)


def build_eval_examples(cfg: RunConfig) -> list[dict]:
    raw = load_dataset("gsm8k", "main", split="test")
    out = []
    for ex in raw:
        gold = extract_answer(ex["answer"])
        out.append({
            "prompt": _format_prompt(ex["question"]),
            "gold": gold,
        })
    if cfg.eval_max_examples is not None:
        out = out[: cfg.eval_max_examples]
    return out


PRIMARY_METRIC = "acc"
