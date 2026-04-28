from __future__ import annotations

from typing import Optional


def tokenize_with_completion(tokenizer, prompt: str, completion: str, max_length: int) -> Optional[dict]:
    """Tokenize prompt+completion as one sequence with labels masked over the prompt.

    Returns None if truncation would drop the entire completion (signal: skip example).
    The returned dict has list-typed fields and is suitable for `datasets.Dataset.from_list`.
    HF Trainer will shift labels internally — do not pre-shift here.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id

    input_ids = list(prompt_ids) + list(completion_ids) + [eos_id]
    labels = [-100] * len(prompt_ids) + list(completion_ids) + [eos_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    if all(l == -100 for l in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
