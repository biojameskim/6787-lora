from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.inference_mode()
def _score_choices(model, tokenizer, prompt: str, choices: list[str], device: str):
    """Return (raw_nlls, norm_nlls, choice_lens) for one example.

    All choices are batched together (small fan-out: 2 for SST-2, 4 for HellaSwag) so
    we get a free per-example speedup without needing cross-example batching machinery.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_seqs = []
    completion_lens = []
    for choice in choices:
        choice_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
        full_seqs.append(list(prompt_ids) + list(choice_ids))
        completion_lens.append(len(choice_ids))

    pad_id = tokenizer.pad_token_id
    max_len = max(len(s) for s in full_seqs)
    input_ids = torch.full((len(choices), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(full_seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, : len(seq)] = 1

    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # (B, T, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    raw_nlls = []
    norm_nlls = []
    for i, seq in enumerate(full_seqs):
        comp_len = completion_lens[i]
        seq_len = len(seq)
        # Tokens at positions [seq_len - comp_len, seq_len - 1] are the completion.
        # Their logits come from positions [seq_len - comp_len - 1, seq_len - 2].
        start = seq_len - comp_len - 1
        end = seq_len - 1
        target_positions = list(range(seq_len - comp_len, seq_len))
        targets = torch.tensor([seq[p] for p in target_positions], device=device)
        token_logp = log_probs[i, start:end, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        total = token_logp.sum().item()
        raw_nlls.append(-total)
        norm_nlls.append(-total / max(comp_len, 1))
    return raw_nlls, norm_nlls


def score_multiple_choice(model, tokenizer, examples: list[dict], device: str = "cuda") -> dict:
    """Score loglikelihood-style multiple-choice tasks.

    Each example: {"prompt": str, "choices": list[str], "label": int}.
    Returns {"acc": float, "acc_norm": float, "n": int}.
    """
    model.eval()
    correct = 0
    correct_norm = 0
    n = 0
    for ex in tqdm(examples, desc="LL eval", leave=False):
        raw, norm = _score_choices(model, tokenizer, ex["prompt"], ex["choices"], device)
        pred = int(min(range(len(raw)), key=lambda i: raw[i]))
        pred_norm = int(min(range(len(norm)), key=lambda i: norm[i]))
        if pred == ex["label"]:
            correct += 1
        if pred_norm == ex["label"]:
            correct_norm += 1
        n += 1
    return {"acc": correct / n, "acc_norm": correct_norm / n, "n": n}
