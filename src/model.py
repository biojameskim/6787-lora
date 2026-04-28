from __future__ import annotations

import torch
from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import RunConfig


def _load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Pad on the right for causal-LM training; switch to left for generation later if needed.
    tok.padding_side = "right"
    return tok


def _load_base_model(cfg: RunConfig):
    if cfg.precision == "int4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, quantization_config=bnb, device_map="cuda"
        )
        return model
    if cfg.precision == "bf16":
        return AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=torch.bfloat16, device_map="cuda"
        )
    if cfg.precision == "fp32":
        # Load in BF16 first to fit in memory headroom from the cluster, then upcast to FP32.
        # For Qwen2.5-1.5B (~3 GB BF16, ~6 GB FP32) this is comfortable.
        return AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=torch.float32, device_map="cuda"
        )
    raise ValueError(f"unknown precision: {cfg.precision}")


def _attach_peft(model, cfg: RunConfig):
    method = cfg.peft_method
    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model

    if method == "lora":
        assert cfg.lora_rank is not None, "lora_rank required for peft_method=lora"
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha_mult * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
        )
    elif method == "ia3":
        peft_cfg = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(cfg.ia3_target_modules),
            feedforward_modules=list(cfg.ia3_feedforward_modules),
        )
    elif method == "prefix":
        assert cfg.prefix_num_virtual_tokens is not None
        peft_cfg = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=cfg.prefix_num_virtual_tokens,
        )
    else:
        raise ValueError(f"unknown peft_method: {method}")

    if cfg.precision == "int4":
        # Required for LoRA over a 4-bit base: cast layernorms to FP32, enable input grads
        # so the adapter sees gradients flowing through the frozen base.
        model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, peft_cfg)


def build_model(cfg: RunConfig):
    """Materialize (model, tokenizer) per RunConfig. Single source of truth for all variants."""
    tokenizer = _load_tokenizer(cfg.model_id)
    base = _load_base_model(cfg)
    model = _attach_peft(base, cfg)
    return model, tokenizer
