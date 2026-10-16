"""Notebook/CLI entry point for the segmentation Modal app.

``run_segmentation`` is the single call the pub notebook uses: given a dataset id,
it launches the Modal app ephemerally, lists that dataset's ND2s under its ``raw/``
prefix, fans the segmentation worker out across GPU containers, saves the returned
overlay previews into a gitignored local folder, and returns per-image summaries
plus a combined measurements table.

Full artifacts (masks, overlays, measurement CSVs) are written back to the
dataset's ``derived/`` prefix by the workers; the small previews returned here are
only for local display.
"""

from __future__ import annotations
import asyncio
import html as html_lib
import logging
import os
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from analysis.app import app, list_images, segment_image
from analysis.bucket_io import derived_prefix, raw_prefix
from analysis.infra import DEFAULT_GPU_MAX_CONTAINERS, DEFAULT_GPU_TYPE
from analysis.segment import SegmentConfig

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("analysis.runner")


def _without_running_loop(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` on a worker thread when an event loop is already running.

    Modal's synchronous APIs (``Function.map`` / ``Function.remote``) spin up their own
    event loop on the *calling* thread; in an environment that already runs one on the
    main thread (e.g. a Jupyter notebook) that raises ``NestedEventLoops`` (surfaced as
    ``InvalidError``). Dispatching to a dedicated worker thread — which has no loop of
    its own — sidesteps this. Outside a running loop (plain scripts, tests) ``fn`` runs
    inline.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result()


@dataclass
class SegmentationRun:
    """Result of a ``run_segmentation`` / ``run_segmentation_local`` call."""

    results: list[dict[str, Any]]  # per-image summaries (with local overlay path)
    measurements: pd.DataFrame  # combined per-cell measurements across images
    results_dir: Path  # local folder holding the overlay previews
    output_prefix: str  # derived/ prefix (S3 key for Modal, local path for local runs)
    backend: str = "modal"  # "modal" (scaled S3 path) or "local" (CPU fallback)
    failures: list[dict[str, Any]] = field(default_factory=list)


def _default_results_dir() -> Path:
    """Gitignored ``scratch/segmentation_results`` under the project root."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "scratch" / "segmentation_results"


def _default_dataset_dir(dataset_id: str) -> Path:
    """Gitignored ``datasets/<dataset_id>`` for the local copy of the published dataset."""
    from analysis.dataset import default_dataset_dir

    return default_dataset_dir(dataset_id)


def _in_notebook() -> bool:
    """True when called from an ipykernel session (Jupyter or ``quarto execute``)."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        return False
    return ip is not None and getattr(ip, "kernel", None) is not None


class _ProgressStream:
    """File-like tqdm target that does not flood notebook / HTML output.

    Headless execute (``make execute`` / nbclient) is not a TTY, so stock tqdm
    writes a *new line* on every refresh. Interactive Jupyter applies ``\\r`` in
    the frontend, which is why the same run looks like one bar there. This stream
    reports ``isatty()`` and, in a kernel, updates a single ``display_id`` HTML
    block so the saved cell output is the last bar only.
    """

    encoding = "utf-8"

    def __init__(self, use_display: bool):
        self._use_display = use_display
        self._display_id = f"cscp-seg-progress-{id(self)}"
        self._lock = threading.Lock()
        self._last = ""
        self._closed = False
        if use_display:
            from IPython.display import HTML, display

            display(HTML("<pre style='margin:0'></pre>"), display_id=self._display_id)

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            text = s.replace("\r", "\n").strip("\n")
            if "\n" in text:
                text = text.split("\n")[-1]
            if text:
                self._last = text
                self._emit(self._last)
        return len(s)

    def flush(self) -> None:
        return None

    def _emit(self, text: str) -> None:
        if not self._use_display:
            sys.stderr.write("\r" + text)
            sys.stderr.flush()
            return
        from IPython.display import HTML, update_display

        update_display(
            HTML(f"<pre style='margin:0'>{html_lib.escape(text)}</pre>"),
            display_id=self._display_id,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._use_display or not self._last:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()


class _SegmentationBar:
    """tqdm wrapper that also closes the custom ``file`` stream."""

    def __init__(self, bar: Any, stream: _ProgressStream):
        self._bar = bar
        self._stream = stream
        self._closed = False

    def reset(self, total: int | None = None) -> None:
        self._bar.reset(total=total)

    def set_postfix_str(self, s: str) -> None:
        self._bar.set_postfix_str(s)

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bar.close()
        self._stream.close()

    def __enter__(self) -> _SegmentationBar:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _segmentation_progress(total: int, *, notebook: bool | None = None) -> _SegmentationBar:
    """Progress bar for a segmentation batch.

    Construct this on the kernel thread. Modal ``map()`` runs on a worker thread
    in Jupyter (to avoid ``NestedEventLoops``); the bar is passed in and updated
    from there. ``notebook`` should be captured on the kernel thread — worker
    threads do not see ``get_ipython()``.
    """
    from tqdm import tqdm as console_tqdm

    use_display = _in_notebook() if notebook is None else notebook
    stream = _ProgressStream(use_display=use_display)
    bar = console_tqdm(
        total=total,
        desc="Segmenting",
        unit="img",
        ncols=88,
        file=stream,
        mininterval=0.1,
    )
    return _SegmentationBar(bar, stream)


def _note_progress(
    pbar: Any,
    *,
    n_cells: int | None = None,
    error: bool = False,
    total_cells: int = 0,
) -> None:
    """Advance the bar; postfix shows the latest cell count (or a failure)."""
    if error:
        pbar.set_postfix_str(f"failed · {total_cells} cells total")
    else:
        pbar.set_postfix_str(f"{n_cells} cells · {total_cells} total")
    pbar.update(1)


def run_segmentation(
    dataset_id: str,
    *,
    limit: int | None = 3,
    gpu: str | None = DEFAULT_GPU_TYPE,
    max_containers: int | None = DEFAULT_GPU_MAX_CONTAINERS,
    results_dir: str | Path | None = None,
    **config_kwargs: Any,
) -> SegmentationRun:
    """Segment a dataset's ND2s on Modal and collect results locally.

    Reads inputs from ``datasets/<dataset_id>/raw`` and writes derived artifacts
    to ``datasets/<dataset_id>/derived`` (grouped into ``masks/``, ``overlays/``,
    and ``measurements/``).

    Args:
        dataset_id: Dataset identifier under the ``datasets/`` namespace.
        limit: Max number of files to process (``None`` for all). Handy for demos.
        gpu: Modal GPU type for the workers (e.g. ``"L4"``). ``None`` keeps the
            function's default.
        max_containers: Ceiling on concurrent GPU containers.
        results_dir: Local folder for overlay previews. Defaults to a gitignored
            ``scratch/segmentation_results`` under the project root.
        **config_kwargs: Forwarded to :class:`~analysis.segment.SegmentConfig`
            (e.g. ``diameter``, ``flow_threshold``, ``cellprob_threshold``,
            ``dog_sigma_low``, ``dog_sigma_high``).

    Returns:
        A :class:`SegmentationRun` with per-image summaries, a combined
        measurements DataFrame, and the local previews directory.
    """
    # The Modal interaction (``app.run()`` + ``Function.map``) must run without an
    # already-running event loop, so hand it to a worker thread inside Jupyter.
    # Create the bar here (kernel thread) so notebook display hooks are registered
    # before work moves off-thread; image count is filled in after listing.
    pbar = _segmentation_progress(0, notebook=_in_notebook())
    try:
        return _without_running_loop(
            _run_segmentation,
            dataset_id,
            limit=limit,
            gpu=gpu,
            max_containers=max_containers,
            results_dir=results_dir,
            pbar=pbar,
            **config_kwargs,
        )
    finally:
        pbar.close()


def _run_segmentation(
    dataset_id: str,
    *,
    limit: int | None = 3,
    gpu: str | None = DEFAULT_GPU_TYPE,
    max_containers: int | None = DEFAULT_GPU_MAX_CONTAINERS,
    results_dir: str | Path | None = None,
    pbar: Any,
    **config_kwargs: Any,
) -> SegmentationRun:
    """Synchronous core of :func:`run_segmentation` (no running event loop)."""
    import pandas as pd

    input_prefix = raw_prefix(dataset_id)
    resolved_output_prefix = derived_prefix(dataset_id)
    config = SegmentConfig(**config_kwargs)

    out_dir = Path(results_dir) if results_dir is not None else _default_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    logger.info("running segmentation")

    with app.run():
        rel_paths = list_images.remote(input_prefix)
        if limit is not None:
            rel_paths = rel_paths[:limit]
        if not rel_paths:
            raise FileNotFoundError(f"No .nd2 files found under prefix '{input_prefix}'.")

        logger.info("segmenting %d image(s) from %s", len(rel_paths), input_prefix)

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

        total_cells = 0
        pbar.reset(total=len(rel_paths))
        for rel_path, result in zip(rel_paths, results, strict=True):
            if isinstance(result, Exception):
                logger.error("failed %s: %s", rel_path, result)
                failures.append({"input": rel_path, "error": str(result)})
                _note_progress(pbar, error=True, total_cells=total_cells)
                continue

            overlay_bytes = result.pop("overlay_jpg_bytes", None)
            records = result.pop("measurements_records", [])
            stem = Path(rel_path).stem
            n_cells = int(result.get("n_cells") or 0)
            total_cells += n_cells

            if overlay_bytes:
                local_overlay = out_dir / f"{stem}_overlay.jpg"
                local_overlay.write_bytes(overlay_bytes)
                result["local_overlay"] = str(local_overlay)

            if records:
                df = pd.DataFrame(records)
                df.insert(0, "image", stem)
                frames.append(df)

            summaries.append(result)
            _note_progress(pbar, n_cells=n_cells, total_cells=total_cells)

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
        backend="modal",
        failures=failures,
    )


# Cached result of ``modal_backend_available`` (the secret lookup hits the network,
# so it is resolved at most once per session). ``None`` means "not yet checked".
_modal_backend_available: bool | None = None


def modal_backend_available() -> bool:
    """Whether the scaled Modal + S3 path can run in this environment.

    True only when the Modal client is authenticated *and* the ``aws-s3-access-secret``
    is reachable (i.e. an internal reader with bucket access). Set ``CSCP_FORCE_LOCAL``
    to force the local fallback regardless. The secret lookup contacts Modal, so the
    result is cached for reuse within a session.
    """
    global _modal_backend_available

    if os.environ.get("CSCP_FORCE_LOCAL"):
        return False

    if _modal_backend_available is not None:
        return _modal_backend_available

    available = False
    try:
        from analysis.infra import SECRET

        SECRET.hydrate()  # raises if unauthenticated or the secret is missing
        available = True
    except Exception as exc:  # auth error, missing secret, offline, etc.
        logger.info("Modal backend unavailable (%s); using local fallback", exc)

    _modal_backend_available = available
    return available


def resolve_backend(backend: str = "auto") -> str:
    """Return ``"modal"`` or ``"local"``, resolving ``"auto"`` from the environment."""
    if backend not in {"auto", "modal", "local"}:
        raise ValueError(f"backend must be 'auto', 'modal', or 'local' (got {backend!r})")
    if backend == "auto":
        return "modal" if modal_backend_available() else "local"
    return backend


def run_segmentation_local(
    dataset_id: str,
    *,
    limit: int | None = 3,
    results_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    **config_kwargs: Any,
) -> SegmentationRun:
    """Segment ND2s locally (CPU) after unpacking the published Zenodo dataset.

    The no-cloud fallback used by external readers: downloads ``{dataset_id}.zip``
    from the public Zenodo archive (full plate plus precomputed ``derived/``),
    runs the same Cellpose-SAM pipeline (:func:`analysis.segment.segment_nd2`) on
    up to ``limit`` local ND2s, writes artifacts under ``derived/``, and returns
    the same :class:`SegmentationRun` shape the notebook consumes.

    Args:
        dataset_id: Dataset identifier used to select files from the Zenodo record.
        limit: Max number of files to process (``None`` for all). Does not shrink
            the Zenodo download.
        results_dir: Local folder for overlay previews (defaults to the gitignored
            ``scratch/segmentation_results``).
        dataset_dir: Local cache dir for the unpacked dataset
            (defaults to ``datasets/<dataset_id>``).
        **config_kwargs: Forwarded to :class:`~analysis.segment.SegmentConfig`.
            ``use_gpu`` defaults to ``False`` for CPU-only reproducibility.
    """
    import pandas as pd

    from analysis.bucket_io import MASKS_SUBDIR, MEASUREMENTS_SUBDIR, OVERLAYS_SUBDIR
    from analysis.dataset import list_nd2s

    config_kwargs.setdefault("use_gpu", False)
    config = SegmentConfig(**config_kwargs)

    out_dir = Path(results_dir) if results_dir is not None else _default_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(dataset_dir) if dataset_dir is not None else _default_dataset_dir(dataset_id)
    derived_dir = base_dir / "derived"

    logger.info("running local segmentation (CPU=%s)", not config.use_gpu)
    nd2_paths = list_nd2s(dataset_id, base_dir, limit=limit)
    logger.info("segmenting %d image(s) locally", len(nd2_paths))

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    total_cells = 0
    with _segmentation_progress(len(nd2_paths)) as pbar:
        for nd2_path in nd2_paths:
            try:
                summary, df = _segment_one_local(
                    nd2_path,
                    config=config,
                    derived_dir=derived_dir,
                    previews_dir=out_dir,
                    subdirs=(MASKS_SUBDIR, OVERLAYS_SUBDIR, MEASUREMENTS_SUBDIR),
                )
            except Exception as exc:
                logger.error("failed %s: %s", nd2_path.name, exc)
                failures.append({"input": str(nd2_path), "error": str(exc)})
                _note_progress(pbar, error=True, total_cells=total_cells)
                continue

            n_cells = int(summary.get("n_cells") or 0)
            total_cells += n_cells
            _note_progress(pbar, n_cells=n_cells, total_cells=total_cells)
            summaries.append(summary)
            if df is not None and not df.empty:
                frames.append(df)

    measurements = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    logger.info(
        "done (local): %d succeeded, %d failed; previews in %s",
        len(summaries),
        len(failures),
        out_dir,
    )
    return SegmentationRun(
        results=summaries,
        measurements=measurements,
        results_dir=out_dir,
        output_prefix=str(derived_dir),
        backend="local",
        failures=failures,
    )


def _segment_one_local(
    nd2_path: Path,
    *,
    config: SegmentConfig,
    derived_dir: Path,
    previews_dir: Path,
    subdirs: tuple[str, str, str],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Segment one ND2 locally, write artifacts, and return (summary, measurements)."""
    import time

    import tifffile
    from PIL import Image

    from analysis.segment import segment_nd2

    masks_subdir, overlays_subdir, measurements_subdir = subdirs
    stem = nd2_path.stem
    started = time.time()

    result = segment_nd2(nd2_path, config=config)

    mask_path = derived_dir / masks_subdir / f"{stem}_mask.tif"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(mask_path, result.mask, metadata={"axes": "YX"})

    overlay_path = derived_dir / overlays_subdir / f"{stem}_overlay.jpg"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.overlay_rgb).convert("RGB").save(overlay_path, quality=90)

    measurements_path = derived_dir / measurements_subdir / f"{stem}_measurements.csv"
    measurements_path.parent.mkdir(parents=True, exist_ok=True)
    result.measurements.to_csv(measurements_path, index=False)

    # Overlay preview alongside the Modal path's previews, for uniform display.
    preview_path = previews_dir / f"{stem}_overlay.jpg"
    Image.fromarray(result.overlay_rgb).convert("RGB").save(preview_path, quality=90)

    elapsed = time.time() - started
    logger.info("done %s (%d cells, %.1fs)", nd2_path.name, result.n_cells, elapsed)

    summary: dict[str, Any] = {
        "input": str(nd2_path),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
        "measurements": str(measurements_path),
        "n_cells": result.n_cells,
        "pixel_size_um": result.pixel_size_um,
        "channel": result.channel_name,
        "elapsed_s": elapsed,
        "local_overlay": str(preview_path),
    }

    df: pd.DataFrame | None = None
    if not result.measurements.empty:
        df = result.measurements.copy()
        df.insert(0, "image", stem)
    return summary, df


def run_demo(
    dataset_id: str,
    *,
    limit: int | None = 3,
    backend: str = "auto",
    **kwargs: Any,
) -> SegmentationRun:
    """Run the pub demo, picking the scaled Modal path or the local fallback.

    This is the single entry point the notebook calls. With ``backend="auto"``
    (default) it runs the scaled Modal + S3 pipeline when that environment is
    available (internal readers) and otherwise falls back to the local CPU path on
    the published Zenodo dataset (external readers). Pass ``backend="modal"`` or
    ``backend="local"`` to force a path.

    Args:
        dataset_id: Dataset identifier.
        limit: Max number of files to process (``None`` for all).
        backend: ``"auto"``, ``"modal"``, or ``"local"``.
        **kwargs: Forwarded to the chosen runner.
    """
    backend = resolve_backend(backend)

    if backend == "modal":
        return run_segmentation(dataset_id, limit=limit, **kwargs)
    return run_segmentation_local(dataset_id, limit=limit, **kwargs)
