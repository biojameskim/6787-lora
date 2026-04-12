"""
Preliminary Exploration: Environment Feasibility Check
CS 6787 Final Project — Efficiency Frontiers of PEFT

This script validates that our training stack works end-to-end:
1. Load a ~1B param model in BF16
2. Attach a LoRA adapter via PEFT
3. Run a single training step on dummy data
4. Report VRAM usage across configurations (LoRA ranks, QLoRA INT4)
5. Verify bitsandbytes quantization support

Usage:
  python preliminary_exploration.py               # auto-detect (tries Llama, falls back to Qwen)
  python preliminary_exploration.py --model llama  # force Llama 3.2-1B (requires HF auth)
  python preliminary_exploration.py --model qwen   # force Qwen 2.5-1.5B (no auth needed)
"""

import argparse
import torch
import gc
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

MODELS = {
    "llama": "meta-llama/Llama-3.2-1B",
    "qwen": "Qwen/Qwen2.5-1.5B",
}


def resolve_model_id(choice: str) -> str:
    """Pick a model, with auto-detection trying Llama first then falling back to Qwen."""
    if choice != "auto":
        model_id = MODELS[choice]
        print(f"Using model: {model_id}")
        return model_id

    # Auto: try Llama first (gated), fall back to Qwen (ungated)
    try:
        print("Auto-detecting model... trying Llama 3.2-1B (requires HF auth)")
        AutoTokenizer.from_pretrained(MODELS["llama"])
        print(f"Using model: {MODELS['llama']}")
        return MODELS["llama"]
    except Exception:
        print("Llama 3.2-1B not accessible (gated/auth). Falling back to Qwen 2.5-1.5B")
        print(f"Using model: {MODELS['qwen']}")
        return MODELS["qwen"]


def print_gpu_stats(label: str):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
    print(f"[{label}]")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Total:     {total:.2f} GB")
    print(f"  Free:      {total - allocated:.2f} GB")
    print()


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def single_training_step(model, tokenizer, label: str):
    """Run one forward + backward pass and report timing/memory."""
    model.train()
    dummy_input = tokenizer(
        "The quick brown fox jumps over the lazy dog.",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    ).to("cuda")
    dummy_input["labels"] = dummy_input["input_ids"].clone()

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = model(**dummy_input)
    loss = outputs.loss
    loss.backward()
    elapsed = time.time() - start

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{label} — single step]")
    print(f"  Loss:        {loss.item():.4f}")
    print(f"  Wall time:   {elapsed:.3f}s")
    print(f"  Peak VRAM:   {peak:.2f} GB")
    print()


# ─── Test 1: BF16 base model + LoRA at various ranks ────────────────────────

def test_lora_ranks(model_id: str):
    print("=" * 60)
    print("TEST 1: LoRA rank sweep (BF16)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    print_gpu_stats("Base model loaded (BF16)")

    trainable_base, total_base = count_params(base_model)
    print(f"Base model params: {total_base:,} (trainable: {trainable_base:,})")
    print()

    for rank in [4, 8, 16, 32, 64]:
        cleanup()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable, total = count_params(model)
        pct = 100 * trainable / total
        print(f"LoRA r={rank}: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")
        print_gpu_stats(f"LoRA r={rank}")
        single_training_step(model, tokenizer, f"LoRA r={rank}")
        del model

    del base_model
    cleanup()


# ─── Test 2: QLoRA (INT4 quantization + LoRA) ───────────────────────────────

def test_qlora(model_id: str):
    print("=" * 60)
    print("TEST 2: QLoRA (INT4 base + LoRA r=16)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="cuda"
    )
    print_gpu_stats("Base model loaded (INT4 / QLoRA)")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable, total = count_params(model)
    pct = 100 * trainable / total
    print(f"QLoRA r=16: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")
    print_gpu_stats("QLoRA r=16 with adapter")
    single_training_step(model, tokenizer, "QLoRA r=16")

    del model
    cleanup()


# ─── Test 3: Full fine-tuning feasibility (BF16) ────────────────────────────

def test_full_finetune(model_id: str):
    print("=" * 60)
    print("TEST 3: Full fine-tuning feasibility (BF16)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    # Ensure all params are trainable
    for param in model.parameters():
        param.requires_grad = True

    trainable, total = count_params(model)
    print(f"Full FT: {trainable:,} trainable / {total:,} total (100%)")
    print_gpu_stats("Full FT model loaded")
    single_training_step(model, tokenizer, "Full fine-tune BF16")

    del model
    cleanup()


# ─── Test 4: Verify bitsandbytes quantization modes ─────────────────────────

def test_bnb_support(model_id: str):
    print("=" * 60)
    print("TEST 4: bitsandbytes quantization support check")
    print("=" * 60)

    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")

    # INT8
    try:
        bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
        model_8bit = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config_8bit,
            device_map="cuda",
        )
        print_gpu_stats("INT8 model loaded")
        del model_8bit
        cleanup()
        print("INT8 quantization: SUPPORTED")
    except Exception as e:
        print(f"INT8 quantization: FAILED — {e}")

    # INT4 (NF4)
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="cuda"
        )
        print_gpu_stats("INT4 (NF4) model loaded")
        del model_4bit
        cleanup()
        print("INT4 (NF4) quantization: SUPPORTED")
    except Exception as e:
        print(f"INT4 (NF4) quantization: FAILED — {e}")

    # INT4 with double quantization
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_4bit_dq = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="cuda"
        )
        print_gpu_stats("INT4 (NF4 + double quant) model loaded")
        del model_4bit_dq
        cleanup()
        print("INT4 (NF4 + double quant): SUPPORTED")
    except Exception as e:
        print(f"INT4 (NF4 + double quant): FAILED — {e}")

    print()


# ─── Summary ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["llama", "qwen", "auto"], default="auto",
        help="Model to use: llama (gated), qwen (ungated), or auto (try llama, fall back to qwen)",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    model_id = resolve_model_id(args.model)
    print()

    test_lora_ranks(model_id)
    test_qlora(model_id)
    test_full_finetune(model_id)
    test_bnb_support(model_id)

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
