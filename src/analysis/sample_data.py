"""Fetch the public sample dataset from a Zenodo archive.

Internal readers run the scaled Modal pipeline against the private S3 bucket. Every
other reader has no bucket access, so the notebook's local fallback pulls
``{dataset_id}.zip`` from a public Zenodo record over HTTPS (stdlib only, no extra
deps) and unpacks it under gitignored ``datasets/<id>/``. The zip layout is:

    raw/*.nd2
    derived/masks/*_mask.tif
    derived/measurements/*_measurements.csv
    derived/overlays/*_overlay.jpg

The Zenodo record is resolved (in priority order) from the ``CSCP_ZENODO_RECORD_ID``
environment variable, then the ``ZENODO_RECORD_ID`` constant below. Set one of them
once the archive is published; until then the helper raises a clear error rather
than silently failing.
"""

from __future__ import annotations
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("analysis.sample_data")

# Zenodo record holding the public example datasets. Fill this in once the archive
# is published (or set the CSCP_ZENODO_RECORD_ID env var to override without a code
# change). Kept as a placeholder so the local path fails loudly instead of guessing.
ZENODO_RECORD_ID: str | None = None

_ZENODO_API = "https://zenodo.org/api/records/{record_id}"


class SampleDataUnavailableError(RuntimeError):
    """Raised when the public sample archive is not configured or reachable."""


def default_cache_dir(dataset_id: str) -> Path:
    """Gitignored ``datasets/<dataset_id>`` unpack location."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "datasets" / dataset_id.strip("/")


def _resolve_record_id() -> str:
    record_id = os.environ.get("CSCP_ZENODO_RECORD_ID") or ZENODO_RECORD_ID
    if not record_id:
        raise SampleDataUnavailableError(
            "No Zenodo record configured for the local demo. Set the "
            "CSCP_ZENODO_RECORD_ID environment variable (or the ZENODO_RECORD_ID "
            "constant in analysis/sample_data.py) to the published record id, or "
            "run the notebook in an environment with Modal + S3 access to use the "
            "scaled pipeline instead."
        )
    return str(record_id)


def _list_record_files(record_id: str) -> list[dict]:
    """Return the Zenodo record's file entries (``key`` + download ``links``)."""
    url = _ZENODO_API.format(record_id=record_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
            payload = json.load(resp)
    except Exception as exc:  # network / HTTP / JSON errors
        raise SampleDataUnavailableError(
            f"Could not read Zenodo record '{record_id}' at {url}: {exc}"
        ) from exc
    return payload.get("files", [])


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def _extract_zip(zip_path: Path, dest_dir: Path, dataset_id: str) -> None:
    """Unpack ``zip_path`` into ``dest_dir``, skipping unsafe members.

    A wrapping ``{dataset_id}/`` directory is stripped so both a zip of ``raw/`` +
    ``derived/`` and a zip of ``{dataset_id}/raw/`` + ``{dataset_id}/derived/``
    unpack to the same layout.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or name.endswith("/"):
                continue
            parts = Path(name).parts
            if Path(name).is_absolute() or ".." in parts:
                logger.warning("skipping unsafe zip member %s", info.filename)
                continue
            if parts and parts[0] == dataset_id:
                parts = parts[1:]
            if not parts:
                continue
            dest = dest_dir.joinpath(*parts)
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("extracting %s -> %s", info.filename, dest)
            with zf.open(info) as src, dest.open("wb") as fh:
                shutil.copyfileobj(src, fh)


def fetch_dataset(dataset_id: str, dest_dir: Path | str | None = None) -> Path:
    """Download ``{dataset_id}.zip`` from Zenodo and unpack it into ``dest_dir``.

    Already-downloaded zip bytes and already-extracted members are reused.
    """
    dataset_id = dataset_id.strip("/")
    dest_dir = Path(dest_dir) if dest_dir is not None else default_cache_dir(dataset_id)
    zip_name = f"{dataset_id}.zip"
    record_id = _resolve_record_id()
    files = _list_record_files(record_id)
    entry = next(
        (item for item in files if Path(str(item.get("key", ""))).name == zip_name),
        None,
    )
    if entry is None:
        raise SampleDataUnavailableError(
            f"Zenodo record '{record_id}' has no file named '{zip_name}'."
        )

    links = entry.get("links") or {}
    url = links.get("self") or links.get("download")
    if not url:
        raise SampleDataUnavailableError(f"Zenodo file '{zip_name}' has no download link.")

    zip_path = dest_dir / zip_name
    expected_size = entry.get("size")
    if zip_path.exists() and (expected_size is None or zip_path.stat().st_size == expected_size):
        logger.info("using cached %s", zip_path)
    else:
        _download(url, zip_path)

    try:
        _extract_zip(zip_path, dest_dir, dataset_id)
    except zipfile.BadZipFile as exc:
        raise SampleDataUnavailableError(f"{zip_path} is not a valid zip archive.") from exc
    return dest_dir


def fetch_sample_files(
    dataset_id: str,
    dest_dir: Path | str | None = None,
    *,
    suffix: str = ".nd2",
    limit: int | None = 3,
) -> list[Path]:
    """Unpack the dataset zip and return local files matching ``suffix``.

    Args:
        dataset_id: Dataset identifier; the record must contain ``{dataset_id}.zip``.
        dest_dir: Unpack directory (defaults to ``datasets/<dataset_id>``).
        suffix: File extension to select (e.g. ``".nd2"`` or ``".csv"``).
        limit: Max number of files to return (``None`` for all).

    Returns:
        Local paths, sorted by name.
    """
    root = fetch_dataset(dataset_id, dest_dir)
    suffix_lower = suffix.lower()
    zip_name = f"{dataset_id.strip('/')}.zip"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != zip_name and path.name.lower().endswith(suffix_lower)
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SampleDataUnavailableError(f"Unpacked '{zip_name}' contains no {suffix} files.")
    return paths


def fetch_sample_nd2s(
    dataset_id: str,
    dest_dir: Path | str | None = None,
    *,
    limit: int | None = 3,
) -> list[Path]:
    """Download up to ``limit`` example ND2s for ``dataset_id`` from Zenodo.

    Thin wrapper over :func:`fetch_sample_files` for the ``.nd2`` inputs consumed by
    the local re-segmentation path.
    """
    return fetch_sample_files(dataset_id, dest_dir, suffix=".nd2", limit=limit)
