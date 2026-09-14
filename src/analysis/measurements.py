"""Load pooled per-cell measurements without re-running segmentation.

Segmentation writes one ``<image>_measurements.csv`` per image under a dataset's
``derived/<id>/measurements`` prefix. Those CSVs are the analysis-ready product, so a
reader who only wants the phenotyping table can load them directly instead of paying
for segmentation again. This is both cheaper and, in the local demo, more complete:
the CPU fallback only segments a handful of images, whereas the published table
covers the whole plate.

Sources:
  * ``"modal"`` — read the CSVs straight from the S3 bucket via a small Modal function
    (reuses the ``aws-s3-access-secret`` mount, so no local AWS credentials are
    needed). For internal readers with bucket access.
  * ``"zenodo"`` — download the published measurement CSVs from the public Zenodo
    record over HTTPS (stdlib only). For external readers.
  * ``"auto"`` — ``"modal"`` when that backend is available, otherwise ``"zenodo"``.

The returned frame matches :attr:`~analysis.runner.SegmentationRun.measurements`: one
row per cell, an ``"image"`` column (the source-image stem) plus the regionprops
feature columns.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("analysis.measurements")


def load_measurements(
    dataset_id: str,
    *,
    source: str = "auto",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load a dataset's pooled per-cell measurements from S3 or Zenodo.

    Args:
        dataset_id: Dataset identifier under the ``datasets/`` namespace.
        source: ``"auto"``, ``"modal"`` (S3 bucket), or ``"zenodo"`` (public archive).
        limit: Max number of per-image CSVs to pool (``None`` for all — the default,
            since the point of loading precomputed measurements is a cheap, complete
            table covering every well).

    Returns:
        One row per cell with an ``"image"`` column plus the regionprops features,
        matching ``SegmentationRun.measurements``.
    """
    if source not in {"auto", "modal", "zenodo"}:
        raise ValueError(f"source must be 'auto', 'modal', or 'zenodo' (got {source!r})")

    if source == "auto":
        from analysis.runner import modal_backend_available

        source = "modal" if modal_backend_available() else "zenodo"

    logger.info("loading measurements for %s from %s", dataset_id, source)
    items = _items_from_modal(dataset_id) if source == "modal" else _items_from_zenodo(dataset_id)
    return _pool(items, limit=limit)


def _pool(items: list[dict[str, Any]], *, limit: int | None) -> pd.DataFrame:
    """Concatenate ``{"image", "csv_text"}`` items into one measurements DataFrame."""
    import io

    import pandas as pd

    if limit is not None:
        items = items[:limit]

    frames: list[pd.DataFrame] = []
    for item in items:
        df = pd.read_csv(io.StringIO(item["csv_text"]))
        if df.empty:
            continue
        df.insert(0, "image", item["image"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _items_from_modal(dataset_id: str) -> list[dict[str, Any]]:
    """Read the dataset's measurement CSVs from S3 through a small Modal function."""
    from analysis.app import app, read_measurements
    from analysis.runner import _without_running_loop

    def _run() -> list[dict[str, Any]]:
        with app.run():
            return read_measurements.remote(dataset_id)

    return _without_running_loop(_run)


def _items_from_zenodo(dataset_id: str) -> list[dict[str, Any]]:
    """Download the dataset's published measurement CSVs from Zenodo and read them."""
    from analysis.bucket_io import image_stem_from_measurements_name
    from analysis.sample_data import fetch_sample_files

    csv_paths = fetch_sample_files(
        dataset_id, _default_measurements_dir(dataset_id), suffix=".csv", limit=None
    )
    return [
        {"image": image_stem_from_measurements_name(path), "csv_text": path.read_text()}
        for path in csv_paths
    ]


def _default_measurements_dir(dataset_id: str) -> Path:
    """Gitignored ``scratch/measurements/<dataset_id>`` cache for downloaded CSVs."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "scratch" / "measurements" / dataset_id.strip("/")
