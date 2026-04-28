"""Roll up results/runs.csv into per-experiment summary tables and frontier plots.

Usage:
    python scripts/analyze.py --experiment exp1
    python scripts/analyze.py --experiment exp2
    python scripts/analyze.py --experiment exp3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results import RUNS_CSV, experiment_dir


def _figs_dir(experiment: str) -> Path:
    d = experiment_dir(experiment) / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _summary_path(experiment: str) -> Path:
    d = experiment_dir(experiment)
    d.mkdir(parents=True, exist_ok=True)
    return d / "summary.csv"


PRIMARY_METRIC = {"sst2": "task_acc", "hellaswag": "task_acc_norm", "gsm8k": "task_acc"}


def _primary(task: str) -> tuple[str, str]:
    base = PRIMARY_METRIC[task]
    return f"{base}_mean", f"{base}_std"


def load() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise SystemExit(f"no runs file at {RUNS_CSV}")
    return pd.read_csv(RUNS_CSV)


def _agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["task_acc", "task_acc_norm", "trainable_params", "peak_vram_gb",
               "train_seconds", "train_tokens_per_sec", "acc_per_param", "acc_per_gpu_hour"]
    g = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return g.reset_index()


def analyze_exp1(df: pd.DataFrame):
    out_csv = _summary_path("exp1")
    figs_dir = _figs_dir("exp1")
    sub = df[df.experiment == "exp1"].copy()
    if sub.empty:
        print("[exp1] no rows yet"); return
    summary = _agg(sub, ["task", "peft_method", "lora_rank"])
    summary.to_csv(out_csv, index=False)
    print(f"[exp1] wrote {out_csv} ({len(summary)} rows)")

    for task in sorted(sub.task.unique()):
        mcol, scol = _primary(task)
        st = summary[(summary.task == task) & (summary.peft_method == "lora")].sort_values("lora_rank")
        if st.empty: continue
        full = summary[(summary.task == task) & (summary.peft_method == "full")]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.errorbar(st.lora_rank, st[mcol], yerr=st[scol], marker="o", capsize=3, label="LoRA")
        if not full.empty:
            ax.axhline(float(full[mcol].iloc[0]), ls="--", color="gray",
                       label=f"Full FT ({float(full[mcol].iloc[0]):.3f})")
        ax.set_xscale("log", base=2); ax.set_xticks(sub.lora_rank.dropna().unique())
        ax.set_xlabel("LoRA rank r"); ax.set_ylabel(f"{PRIMARY_METRIC[task]}")
        ax.set_title(f"Exp1: {task}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(figs_dir / f"acc_vs_rank__{task}.png", dpi=150)
        plt.close(fig)
    # frontiers: acc vs trainable params, and acc vs GPU-hour
    for fname, xlabel, xcol in [
        ("frontier_params.png", "Trainable params", "trainable_params_mean"),
        ("frontier_gpu_hour.png", "Train seconds / run", "train_seconds_mean"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for task in sorted(sub.task.unique()):
            mcol, _ = _primary(task)
            st = summary[(summary.task == task) & (summary.peft_method == "lora")].sort_values(xcol)
            if st.empty: continue
            ax.plot(st[xcol], st[mcol], marker="o", label=f"{task} (LoRA)")
            full = summary[(summary.task == task) & (summary.peft_method == "full")]
            if not full.empty:
                ax.scatter(full[xcol], full[mcol], marker="x", s=80,
                           label=f"{task} (full FT)")
        ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("Accuracy (primary metric)")
        ax.set_title(f"Exp1: accuracy vs {xlabel.lower()}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(figs_dir / fname, dpi=150)
        plt.close(fig)
    print(f"[exp1] wrote plots to {figs_dir}/")


def analyze_exp2(df: pd.DataFrame):
    out_csv = _summary_path("exp2")
    figs_dir = _figs_dir("exp2")
    # Stitch in the bf16/r=16 LoRA cells from exp1 — those are bit-for-bit the
    # same configuration as exp2's bf16 column, so we reuse them rather than retrain.
    e2 = df[df.experiment == "exp2"].copy()
    e1_share = df[(df.experiment == "exp1")
                  & (df.peft_method == "lora")
                  & (df.lora_rank == 16)
                  & (df.precision == "bf16")].copy()
    sub = pd.concat([e2, e1_share], ignore_index=True)
    if sub.empty:
        print("[exp2] no rows yet"); return
    summary = _agg(sub, ["task", "precision"])
    summary.to_csv(out_csv, index=False)
    print(f"[exp2] wrote {out_csv} ({len(summary)} rows)")

    order = ["fp32", "bf16", "int4"]
    for task in sorted(sub.task.unique()):
        st = summary[summary.task == task].set_index("precision").reindex(order)
        if st.empty: continue
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        axes[0].bar(st.index, st.task_acc_mean, yerr=st.task_acc_std, capsize=3)
        axes[0].set_ylabel("Accuracy"); axes[0].set_title("Accuracy")
        axes[1].bar(st.index, st.peak_vram_gb_mean); axes[1].set_ylabel("Peak VRAM (GB)"); axes[1].set_title("VRAM")
        axes[2].bar(st.index, st.train_tokens_per_sec_mean); axes[2].set_ylabel("tokens/s"); axes[2].set_title("Throughput")
        for ax in axes: ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle(f"Exp2: {task}"); fig.tight_layout()
        fig.savefig(figs_dir / f"precision__{task}.png", dpi=150)
        plt.close(fig)
    print(f"[exp2] wrote plots to {figs_dir}/")


def analyze_exp3(df: pd.DataFrame):
    out_csv = _summary_path("exp3")
    figs_dir = _figs_dir("exp3")
    # Stitch in the lora_r16 and full FT cells from exp1 — those are bit-for-bit
    # the same configuration as exp3's "lora_r16" and "full" columns.
    e3 = df[df.experiment == "exp3"].copy()
    e1_share = df[(df.experiment == "exp1")
                  & (df.precision == "bf16")
                  & ((df.peft_method == "full")
                     | ((df.peft_method == "lora") & (df.lora_rank == 16)))].copy()
    sub = pd.concat([e3, e1_share], ignore_index=True)
    if sub.empty:
        print("[exp3] no rows yet"); return
    def _method_tag(rid: str) -> str:
        for tag in ("lora_r16", "ia3", "prefix20", "full"):
            if f"__{tag}__" in rid: return tag
        return "?"
    sub["method_tag"] = sub.run_id.map(_method_tag)
    summary = _agg(sub, ["task", "method_tag"])
    summary.to_csv(out_csv, index=False)
    print(f"[exp3] wrote {out_csv} ({len(summary)} rows)")

    method_order = ["full", "lora_r16", "ia3", "prefix20"]
    for task in sorted(sub.task.unique()):
        st = summary[summary.task == task].set_index("method_tag").reindex(method_order)
        if st.empty: continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(st.index, st.task_acc_mean, yerr=st.task_acc_std, capsize=3)
        ax.set_ylabel("Accuracy"); ax.set_title(f"Exp3: {task}")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(figs_dir / f"methods__{task}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for task in sorted(sub.task.unique()):
        st = summary[summary.task == task]
        ax.scatter(st.trainable_params_mean, st.task_acc_mean, label=task)
        for _, row in st.iterrows():
            ax.annotate(row.method_tag, (row.trainable_params_mean, row.task_acc_mean), fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("Trainable params"); ax.set_ylabel("Accuracy")
    ax.set_title("Exp3: per-method param frontier"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(figs_dir / "frontier_params.png", dpi=150)
    plt.close(fig)
    print(f"[exp3] wrote plots to {figs_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=["exp1", "exp2", "exp3", "all"])
    args = p.parse_args()
    df = load()
    fns = {"exp1": analyze_exp1, "exp2": analyze_exp2, "exp3": analyze_exp3}
    todo = list(fns.values()) if args.experiment == "all" else [fns[args.experiment]]
    for fn in todo:
        fn(df)


if __name__ == "__main__":
    main()
