"""Sweep Modal worker counts for the scale table in the pub.

Runs the full plate through ``run_demo`` on Modal L4 GPUs at each
``max_containers`` in ``configs``, then writes wall-clock time, per-image
compute stats, and throughput (images per wall-clock minute) to
``benchmark.csv`` beside this script. Numbers in @tbl-benchmark in
``index.ipynb`` come from a run of this script (plus a separate local
M3 Max timing).

Requires an authenticated Modal client and bucket access. Not part of
``make execute`` — a full sweep is slow and spends GPU time.

    uv run python scripts/benchmark.py
"""

import statistics
import time
from pathlib import Path

import pandas as pd
from analysis.runner import run_demo

DATASET_ID = "20260611_141152_330"
LIMIT = None
OUTPUT_CSV = Path(__file__).parent / "benchmark.csv"

configs = [
    ("L4", 64),
    ("L4", 16),
    ("L4", 4),
    ("L4", 1),
]

rows = []
for gpu, max_containers in configs:
    t0 = time.time()
    run = run_demo(
        DATASET_ID,
        limit=LIMIT,
        backend="modal",
        gpu=gpu,
        max_containers=max_containers,
    )

    elapsed = [r["elapsed_s"] for r in run.results]
    n_images = len(run.results)
    wall_s = time.time() - t0
    rows.append(
        {
            "backend": "modal",
            "gpu": gpu,
            "max_containers": max_containers,
            "n_images": n_images,
            "n_failures": len(run.failures),
            "wall_s": wall_s,
            "compute_s_sum": sum(elapsed) if elapsed else float("nan"),
            "compute_s_median": statistics.median(elapsed) if elapsed else float("nan"),
            "compute_s_max": max(elapsed) if elapsed else float("nan"),
            "throughput_img_per_min": n_images / wall_s * 60 if wall_s else float("nan"),
        }
    )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
