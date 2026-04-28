from __future__ import annotations

from . import gsm8k, hellaswag, sst2

TASKS = {
    "sst2": sst2,
    "hellaswag": hellaswag,
    "gsm8k": gsm8k,
}


def get_task(name: str):
    if name not in TASKS:
        raise KeyError(f"unknown task: {name}; choices = {list(TASKS)}")
    return TASKS[name]
