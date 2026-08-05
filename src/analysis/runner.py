"""Notebook/CLI entry point for the segmentation Modal app.

``run_segmentation`` is the single call the pub notebook uses: it launches the
Modal app ephemerally, lists ND2s under an S3 prefix, fans the segmentation worker
out across GPU containers, saves the returned overlay previews into a gitignored
local folder, and returns per-image summaries plus a combined measurements table.

Full artifacts (masks, overlays, measurement CSVs) are written back to S3 by the
workers; the small previews returned here are only for local display.
"""

from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from analysis.app import app, list_images, segment_image
from analysis.infra import DEFAULT_GPU_MAX_CONTAINERS, DEFAULT_GPU_TYPE, DEFAULT_OUTPUT_PREFIX
from analysis.segment import SegmentConfig

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("analysis.runner")


@dataclass
class SegmentationRun:
    """Result of a ``run_segmentation`` call."""

    results: list[dict[str, Any]]  # per-image summaries (with local overlay path)
    measurements: pd.DataFrame  # combined per-cell measurements across images
    results_dir: Path  # local folder holding the overlay previews
    output_prefix: str  # S3 prefix where full artifacts were written
    failures: list[dict[str, Any]] = field(default_factory=list)


def _default_results_dir() -> Path:
    """Gitignored ``scratch/segmentation_results`` under the project root."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "scratch" / "segmentation_results"


def run_segmentation(
    input_prefix: str,
    *,
    output_prefix: str | None = None,
    limit: int | None = 3,
    gpu: str | None = DEFAULT_GPU_TYPE,
    max_containers: int | None = DEFAULT_GPU_MAX_CONTAINERS,
    results_dir: str | Path | None = None,
    **config_kwargs: Any,
) -> SegmentationRun:
    """Segment ND2s under ``input_prefix`` on Modal and collect results locally.

    Args:
        input_prefix: Bucket-relative prefix containing the input ``.nd2`` files.
        output_prefix: Bucket-relative prefix for derived artifacts. Defaults to
            ``derived/segmentation/<input_prefix>``.
        limit: Max number of files to process (``None`` for all). Handy for demos.
        gpu: Modal GPU type for the workers (e.g. ``"L4"``). ``None`` keeps the
            function's default.
        max_containers: Ceiling on concurrent GPU containers.
        results_dir: Local folder for overlay previews. Defaults to a gitignored
            ``scratch/segmentation_results`` under the project root.
        **config_kwargs: Forwarded to :class:`~analysis.segment.SegmentConfig`
            (e.g. ``diameter``, ``flow_threshold``, ``cellprob_threshold``).

    Returns:
        A :class:`SegmentationRun` with per-image summaries, a combined
        measurements DataFrame, and the local previews directory.
    """
    # Modal's synchronous ``Function.map`` iterator creates its own event loop on
    # the *calling* thread. In an environment that already runs one on the main
    # thread (e.g. a Jupyter notebook) that raises ``NestedEventLoops`` (surfaced
    # as ``InvalidError``). When we detect a running loop, run the whole Modal
    # interaction on a dedicated worker thread, which has no loop of its own.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        running_loop = False
    else:
        running_loop = True

    if running_loop:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _run_segmentation,
                input_prefix,
                output_prefix=output_prefix,
                limit=limit,
                gpu=gpu,
                max_containers=max_containers,
                results_dir=results_dir,
                **config_kwargs,
            )
            return future.result()

    return _run_segmentation(
        input_prefix,
        output_prefix=output_prefix,
        limit=limit,
        gpu=gpu,
        max_containers=max_containers,
        results_dir=results_dir,
        **config_kwargs,
    )


def _run_segmentation(
    input_prefix: str,
    *,
    output_prefix: str | None = None,
    limit: int | None = 3,
    gpu: str | None = DEFAULT_GPU_TYPE,
    max_containers: int | None = DEFAULT_GPU_MAX_CONTAINERS,
    results_dir: str | Path | None = None,
    **config_kwargs: Any,
) -> SegmentationRun:
    """Synchronous core of :func:`run_segmentation` (no running event loop)."""
    import pandas as pd

    resolved_output_prefix = output_prefix or f"{DEFAULT_OUTPUT_PREFIX}/{input_prefix.strip('/')}"
    config = SegmentConfig(**config_kwargs)

    out_dir = Path(results_dir) if results_dir is not None else _default_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    logger.warning("running segmentation")

    with app.run():
        rel_paths = list_images.remote(input_prefix)
        if limit is not None:
            rel_paths = rel_paths[:limit]
        if not rel_paths:
            raise FileNotFoundError(f"No .nd2 files found under prefix '{input_prefix}'.")

        logger.warning("segmenting %d image(s) from %s", len(rel_paths), input_prefix)

        worker = segment_image
        options: dict[str, Any] = {}
        if gpu:
            options["gpu"] = gpu
        if max_containers:
            options["max_containers"] = max_containers
        if options:
            worker = segment_image.with_options(**options)

        results = worker.map(
            rel_paths,
            kwargs={"output_prefix": resolved_output_prefix, "config": config.to_dict()},
            return_exceptions=True,
        )

        for rel_path, result in zip(rel_paths, results, strict=True):
            if isinstance(result, Exception):
                logger.error("failed %s: %s", rel_path, result)
                failures.append({"input": rel_path, "error": str(result)})
                continue

            overlay_bytes = result.pop("overlay_jpg_bytes", None)
            records = result.pop("measurements_records", [])
            stem = Path(rel_path).stem

            if overlay_bytes:
                local_overlay = out_dir / f"{stem}_overlay.jpg"
                local_overlay.write_bytes(overlay_bytes)
                result["local_overlay"] = str(local_overlay)

            if records:
                df = pd.DataFrame(records)
                df.insert(0, "image", stem)
                frames.append(df)

            summaries.append(result)

    measurements = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    logger.info(
        "done: %d succeeded, %d failed; previews in %s",
        len(summaries),
        len(failures),
        out_dir,
    )
    return SegmentationRun(
        results=summaries,
        measurements=measurements,
        results_dir=out_dir,
        output_prefix=resolved_output_prefix,
        failures=failures,
    )
