from __future__ import annotations

import time

import torch


class TrainTimer:
    """Context manager that captures wall time and peak VRAM."""

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
        self.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9


def count_params(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }
