"""Bucket file I/O helpers used inside Modal workers.

These operate against the ``CloudBucketMount``. mountpoint-s3 is optimized for
whole-file sequential reads/writes, so random-access readers (e.g. the nd2 reader)
should work on a local copy: fetch to scratch, process, push the results back.

All functions take an explicit ``root`` (the mount path) so they stay free of any
Modal import and remain unit-testable against a plain local directory.
"""

from __future__ import annotations
import shutil
from pathlib import Path

# Top-level namespace holding one self-contained directory per dataset. Each
# dataset keeps its immutable ND2 inputs under ``raw/`` and its regenerable
# artifacts under ``derived/`` (grouped by artifact type), so a whole dataset can
# be listed, archived, or handed off as a single unit.
DATASETS_ROOT = "datasets"

# Artifact-type subdirectories under a dataset's ``derived/`` prefix.
MASKS_SUBDIR = "masks"
OVERLAYS_SUBDIR = "overlays"
MEASUREMENTS_SUBDIR = "measurements"


def raw_prefix(dataset_id: str) -> str:
    """Bucket-relative prefix holding a dataset's raw ``.nd2`` inputs."""
    return f"{DATASETS_ROOT}/{dataset_id.strip('/')}/raw"


def derived_prefix(dataset_id: str) -> str:
    """Bucket-relative prefix holding a dataset's derived artifacts."""
    return f"{DATASETS_ROOT}/{dataset_id.strip('/')}/derived"


def measurements_prefix(dataset_id: str) -> str:
    """Bucket-relative prefix holding a dataset's per-image measurement CSVs."""
    return f"{derived_prefix(dataset_id)}/{MEASUREMENTS_SUBDIR}"


def image_stem_from_measurements_name(rel_path: str | Path) -> str:
    """Recover the source-image stem from a ``<stem>_measurements.csv`` path."""
    stem = Path(rel_path).stem
    suffix = f"_{MEASUREMENTS_SUBDIR}"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def read_measurement_csvs(root: Path | str, dataset_id: str) -> list[tuple[str, str]]:
    """Return ``(image_stem, csv_text)`` for each of a dataset's measurement CSVs.

    Reads the per-image CSVs written under ``derived/<dataset_id>/measurements`` so a
    caller can pool them into one table without re-running segmentation. Operates on a
    plain ``root`` (the mount path), so it stays Modal-free and unit-testable.
    """
    root = Path(root)
    rel_paths = list_files(root, measurements_prefix(dataset_id), suffix=".csv")
    return [(image_stem_from_measurements_name(rel), (root / rel).read_text()) for rel in rel_paths]


def list_files(root: Path | str, prefix: str, suffix: str = ".nd2") -> list[str]:
    """Return root-relative paths of every ``*{suffix}`` file under ``prefix``."""
    root = Path(root)
    search_root = root / prefix.strip("/")
    matches = sorted(search_root.rglob(f"*{suffix}"))
    return [str(p.relative_to(root)) for p in matches]


def fetch_to_local(root: Path | str, rel_path: str, dest_dir: Path | str) -> Path:
    """Copy a bucket file to local scratch and return the local path.

    Use before any random-access reader (e.g. the nd2 reader) touches the file.
    """
    root = Path(root)
    dest_dir = Path(dest_dir)
    src = root / rel_path
    local = dest_dir / Path(rel_path).name
    shutil.copyfile(src, local)
    return local


def push_from_local(local_path: Path | str, root: Path | str, out_rel: str) -> Path:
    """Copy a local file to ``root/out_rel`` in the bucket, returning the dest path."""
    root = Path(root)
    dest = root / out_rel.strip("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_path, dest)
    return dest
