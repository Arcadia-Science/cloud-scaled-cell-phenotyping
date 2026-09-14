"""Modal app: parallel single-cell segmentation of brightfield ND2s.

Fans out across GPU containers: each worker fetches one ND2 from the shared
CloudBucketMount, runs Cellpose-SAM segmentation, and writes the full artifacts
(label-mask TIFF, outline-overlay JPG, per-cell measurements CSV) back to the
bucket under a derived prefix. Each worker also returns a *small* preview payload
(overlay JPEG bytes + cell count + measurement records) so a notebook can display
results without needing local S3 credentials.
"""

from __future__ import annotations
import logging
import time
from pathlib import PurePosixPath

import modal

from analysis.infra import (
    FUNCTION_TIMEOUT_S,
    MOUNT_PATH,
    data_mount,
    image,
    meta_image,
)

# Both inputs and outputs live in the same bucket, mounted read/write.
_volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    MOUNT_PATH: data_mount
}

logger = logging.getLogger("analysis.segmentation")

app = modal.App("cscp-segmentation")


@app.function(image=meta_image, volumes=_volumes, timeout=FUNCTION_TIMEOUT_S)
def list_images(input_prefix: str, suffix: str = ".nd2") -> list[str]:
    """Return bucket-relative paths of every ND2 under ``input_prefix``."""
    from analysis.bucket_io import list_files

    rel_paths = list_files(MOUNT_PATH, input_prefix, suffix=suffix)
    logger.info("found %d images under %s", len(rel_paths), input_prefix)
    return rel_paths


@app.function(image=meta_image, volumes=_volumes, timeout=FUNCTION_TIMEOUT_S)
def read_measurements(dataset_id: str) -> list[dict]:
    """Return a dataset's per-image measurement CSVs as ``{"image", "csv_text"}`` dicts.

    Reads the CSVs the segmentation workers wrote to ``derived/<id>/measurements`` so a
    caller can pool the analysis-ready table without re-segmenting. Runs on the light
    stdlib image (no pandas); the caller concatenates the CSV text locally.
    """
    from analysis.bucket_io import read_measurement_csvs

    items = read_measurement_csvs(MOUNT_PATH, dataset_id)
    logger.info("read %d measurement CSV(s) for %s", len(items), dataset_id)
    return [{"image": image, "csv_text": text} for image, text in items]


@app.function(image=image, gpu="L4", volumes=_volumes, timeout=FUNCTION_TIMEOUT_S)
def segment_image(rel_path: str, output_prefix: str, config: dict) -> dict:
    """Segment one ND2; write mask/overlay/measurements to S3, return a small preview."""
    import tempfile
    from pathlib import Path

    import tifffile
    from PIL import Image

    from analysis import bucket_io
    from analysis.bucket_io import MASKS_SUBDIR, MEASUREMENTS_SUBDIR, OVERLAYS_SUBDIR
    from analysis.segment import SegmentConfig, segment_nd2

    cfg = SegmentConfig(**config)
    started = time.time()
    name = Path(rel_path).name
    logger.info("start %s (gpu=%s)", name, cfg.use_gpu)

    out_prefix = output_prefix.strip("/")
    stem = Path(rel_path).stem

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        local_nd2 = bucket_io.fetch_to_local(MOUNT_PATH, rel_path, tmp_dir)

        result = segment_nd2(local_nd2, config=cfg)

        # --- Write full artifacts back to the bucket, grouped by artifact type ---
        local_tif = tmp_dir / f"{stem}_mask.tif"
        tifffile.imwrite(local_tif, result.mask, metadata={"axes": "YX"})
        mask_rel = f"{out_prefix}/{MASKS_SUBDIR}/{stem}_mask.tif"
        mask_dest = bucket_io.push_from_local(local_tif, MOUNT_PATH, mask_rel)

        local_jpg = tmp_dir / f"{stem}_overlay.jpg"
        Image.fromarray(result.overlay_rgb).convert("RGB").save(local_jpg, quality=90)
        overlay_rel = f"{out_prefix}/{OVERLAYS_SUBDIR}/{stem}_overlay.jpg"
        overlay_dest = bucket_io.push_from_local(local_jpg, MOUNT_PATH, overlay_rel)

        local_csv = tmp_dir / f"{stem}_measurements.csv"
        result.measurements.to_csv(local_csv, index=False)
        measurements_rel = f"{out_prefix}/{MEASUREMENTS_SUBDIR}/{stem}_measurements.csv"
        measurements_dest = bucket_io.push_from_local(local_csv, MOUNT_PATH, measurements_rel)

        # Small preview to return over the wire for notebook display.
        overlay_bytes = local_jpg.read_bytes()

    elapsed = time.time() - started
    mask_out = str(mask_dest.relative_to(MOUNT_PATH))
    overlay_out = str(overlay_dest.relative_to(MOUNT_PATH))
    measurements_out = str(measurements_dest.relative_to(MOUNT_PATH))
    logger.info(
        "done %s -> %s (%d cells, %.1fs)",
        name,
        mask_out,
        result.n_cells,
        elapsed,
    )
    return {
        "input": rel_path,
        "mask": mask_out,
        "overlay": overlay_out,
        "measurements": measurements_out,
        "n_cells": result.n_cells,
        "pixel_size_um": result.pixel_size_um,
        "channel": result.channel_name,
        "elapsed_s": elapsed,
        "overlay_jpg_bytes": overlay_bytes,
        "measurements_records": result.measurements.to_dict(orient="records"),
    }
