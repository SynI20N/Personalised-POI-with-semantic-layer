"""
Keep track of every evaluation run: one JSON line per run in results/runs.jsonl.

Each record holds when the run happened, which model and data, every setting, the
metrics, and the file with the per-pair ranks (for paired comparisons between runs).

Read all runs as a table:
    import pandas as pd
    runs = pd.read_json("results/runs.jsonl", lines=True)
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

BASE_PATH = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_PATH / "results"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"


def new_run_id(model):
    """Unique, sortable id, e.g. 'temn_20261116-173012'."""
    return f"{model}_{time.strftime('%Y%m%d-%H%M%S')}"


def ranking_metrics(ranks, ks=(5, 10)):
    """One positive per group: HR@k = rank < k, NDCG@k = 1 / log2(rank + 2) if rank < k."""
    ranks = np.asarray(ranks)
    metrics = {}
    for k in ks:
        hit = ranks < k
        metrics[f"HR@{k}"] = round(float(hit.mean()), 4)
        metrics[f"NDCG@{k}"] = round(float(np.where(hit, 1.0 / np.log2(ranks + 2), 0.0).mean()), 4)
    return metrics


def git_commit():
    """Current commit hash, with '+dirty' if there are uncommitted changes; None outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_PATH,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=BASE_PATH, capture_output=True, text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def save_ranks(run_id, ranks):
    """Per-pair ranks (0 = first) in the order of the test groups; returns the file path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"ranks_{run_id}.txt"
    np.savetxt(path, np.asarray(ranks), fmt="%d")
    return path


def log_run(run_id, model, settings, ranks, extra=None):
    """
    Save the ranks and append one record to results/runs.jsonl. Returns the record.

    settings: everything needed to repeat the run (data, seed, hyperparameters).
    extra:    anything else worth keeping (e.g. the loss per epoch).
    """
    record = {
        "run_id": run_id,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "git_commit": git_commit(),
        "settings": settings,
        "n_pairs": int(len(ranks)),
        "metrics": ranking_metrics(ranks),
        "ranks_file": str(save_ranks(run_id, ranks).relative_to(BASE_PATH)),
    }
    if extra:
        record.update(extra)
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(f"logged {run_id} to {RUNS_FILE}")
    return record
