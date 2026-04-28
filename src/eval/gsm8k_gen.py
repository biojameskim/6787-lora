from __future__ import annotations

import torch
from tqdm import tqdm

from ..data.gsm8k import extract_answer


def _normalize(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    # Treat "5" and "5.0" as equal.
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return s


@torch.inference_mode()
def score_gsm8k_generation(
    model,
    tokenizer,
    examples: list[dict],
    device: str = "cuda",
    max_new_tokens: int = 256,
    keep_samples: int = 20,
) -> dict:
    """Greedy-decode each prompt, regex-extract `#### N`, compare to gold."""
    model.eval()
    correct = 0
    n = 0
    samples = []
    # Use left padding for generation to keep it simple at batch=1.
    # (Batch=1 keeps the loop straightforward; can revisit if eval becomes a bottleneck.)
    for ex in tqdm(examples, desc="GSM8K eval", leave=False):
        prompt = ex["prompt"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attn = torch.ones_like(input_ids)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[0, input_ids.shape[1] :]
        decoded = tokenizer.decode(gen, skip_special_tokens=True)
        pred = _normalize(extract_answer(decoded))
        gold = _normalize(ex["gold"])
        ok = pred is not None and pred == gold
        if ok:
            correct += 1
        n += 1
        if len(samples) < keep_samples:
            samples.append({
                "prompt": prompt,
                "gold": gold,
                "pred": pred,
                "completion": decoded,
            })
    return {"acc": correct / n, "n": n, "samples": samples}
