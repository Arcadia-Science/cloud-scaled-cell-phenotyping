"""Fetch the published dataset from a Zenodo archive.

Internal readers run the scaled Modal pipeline against the private S3 bucket. Every
other reader has no bucket access, so the notebook's local path pulls
``{dataset_id}.zip`` from a public Zenodo record over HTTPS (stdlib only, no extra
deps) and unpacks it under gitignored ``datasets/<id>/``. The zip is the full
dataset, not a subsample. Layout:

    raw/*.nd2
    derived/masks/*_mask.tif
    derived/measurements/*_measurements.csv
    derived/overlays/*_overlay.jpg

The Zenodo record is resolved (in priority order) from the ``CSCP_ZENODO_RECORD_ID``
environment variable, then the ``ZENODO_RECORD_ID`` constant below. The env var
overrides the constant without a code change.
"""

from __future__ import annotations
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("analysis.dataset")

# Published dataset: https://doi.org/10.5281/zenodo.23377605
# Override with CSCP_ZENODO_RECORD_ID if needed.
ZENODO_RECORD_ID: str | None = "23377605"

_ZENODO_API = "https://zenodo.org/api/records/{record_id}"


class DatasetUnavailableError(RuntimeError):
    """Raised when the public dataset archive is not configured or reachable."""


def default_dataset_dir(dataset_id: str) -> Path:
    """Gitignored ``datasets/<dataset_id>`` unpack location."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "datasets" / dataset_id.strip("/")


def _resolve_record_id() -> str:
    record_id = os.environ.get("CSCP_ZENODO_RECORD_ID") or ZENODO_RECORD_ID
    if not record_id:
        raise DatasetUnavailableError(
            "No Zenodo record configured for the local path. Set the "
            "CSCP_ZENODO_RECORD_ID environment variable (or the ZENODO_RECORD_ID "
            "constant in analysis/dataset.py) to the published record id, or "
            "run the notebook in an environment with Modal + S3 access to use the "
            "scaled pipeline instead."
        )
    return str(record_id)


def _list_record_files(record_id: str) -> list[dict]:
    """Return the Zenodo record's file entries (``key`` + download ``links``)."""
    url = _ZENODO_API.format(record_id=record_id)
    logger.info("GET %s", url)
    print(f"[zenodo] Fetching record {record_id} from {url} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
            payload = json.load(resp)
    except Exception as exc:  # network / HTTP / JSON errors
        raise DatasetUnavailableError(
            f"Could not read Zenodo record '{record_id}' at {url}: {exc}"
        ) from exc
    return payload.get("files", [])


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s -> %s", url, dest)
    print(f"[zenodo] Downloading {url} -> {dest} ...", flush=True)
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


def ensure_dataset(dataset_id: str, dest_dir: Path | str | None = None) -> Path:
    """Download ``{dataset_id}.zip`` from Zenodo if needed and unpack it into ``dest_dir``.

    Already-downloaded zip bytes and already-extracted members are reused. The archive
    is the full published dataset (raw ND2s plus derived masks, overlays, and
    measurement tables), not a subsample.
    """
    dataset_id = dataset_id.strip("/")
    dest_dir = Path(dest_dir) if dest_dir is not None else default_dataset_dir(dataset_id)
    zip_name = f"{dataset_id}.zip"
    zip_path = dest_dir / zip_name
    raw_dir = dest_dir / "raw"
    if raw_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() == ".nd2" for path in raw_dir.rglob("*")
    ):
        logger.info("using unpacked dataset at %s", dest_dir)
        print(f"[zenodo] Using unpacked dataset at {dest_dir}", flush=True)
        return dest_dir
    if zip_path.exists():
        logger.info("using cached %s", zip_path)
        print(f"[zenodo] Using cached {zip_path}", flush=True)
        try:
            _extract_zip(zip_path, dest_dir, dataset_id)
        except zipfile.BadZipFile as exc:
            raise DatasetUnavailableError(f"{zip_path} is not a valid zip archive.") from exc
        return dest_dir

    record_id = _resolve_record_id()
    files = _list_record_files(record_id)
    entry = next(
        (item for item in files if Path(str(item.get("key", ""))).name == zip_name),
        None,
    )
    if entry is None:
        raise DatasetUnavailableError(
            f"Zenodo record '{record_id}' has no file named '{zip_name}'."
        )

    links = entry.get("links") or {}
    url = links.get("content") or links.get("self") or links.get("download")
    if not url:
        raise DatasetUnavailableError(f"Zenodo file '{zip_name}' has no download link.")

    _download(url, zip_path)

    try:
        _extract_zip(zip_path, dest_dir, dataset_id)
    except zipfile.BadZipFile as exc:
        raise DatasetUnavailableError(f"{zip_path} is not a valid zip archive.") from exc
    return dest_dir


def list_dataset_files(
    dataset_id: str,
    dest_dir: Path | str | None = None,
    *,
    suffix: str = ".nd2",
    under: str = "raw",
    limit: int | None = 3,
) -> list[Path]:
    """Unpack the dataset zip and return local files matching ``suffix`` under ``under``.

    Args:
        dataset_id: Dataset identifier; the record must contain ``{dataset_id}.zip``.
        dest_dir: Unpack directory (defaults to ``datasets/<dataset_id>``).
        suffix: File extension to select (e.g. ``".nd2"`` or ``".csv"``).
        under: Directory relative to the unpack root to search
            (``"raw"`` or ``"derived/measurements"``).
        limit: Max number of files to return (``None`` for all). ``limit`` only
            slices the returned list; the full zip is still downloaded.

    Returns:
        Local paths, sorted by name.
    """
    root = ensure_dataset(dataset_id, dest_dir)
    suffix_lower = suffix.lower()
    search_root = root / under
    paths = (
        sorted(
            path
            for path in search_root.rglob("*")
            if path.is_file() and path.name.lower().endswith(suffix_lower)
        )
        if search_root.is_dir()
        else []
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise DatasetUnavailableError(
            f"Unpacked '{dataset_id.strip('/')}.zip' contains no {suffix} files under '{under}'."
        )
    return paths


def list_nd2s(
    dataset_id: str,
    dest_dir: Path | str | None = None,
    *,
    limit: int | None = 3,
) -> list[Path]:
    """Return up to ``limit`` ND2s from the published dataset's ``raw/`` directory."""
    return list_dataset_files(dataset_id, dest_dir, suffix=".nd2", under="raw", limit=limit)


def ensure_raw_nd2(
    dataset_id: str,
    filename: str,
    dest_dir: Path | str | None = None,
) -> Path:
    """Return a local path to one raw ND2, fetching it if needed.

    Order: already-unpacked file, then a single S3 copy when the Modal backend is
    available, otherwise the published Zenodo zip (the full dataset). Readers
    without bucket access never hit S3, even if an ``aws`` CLI is installed.
    """
    import subprocess

    dest_dir = Path(dest_dir) if dest_dir is not None else default_dataset_dir(dataset_id)
    dest = dest_dir / "raw" / filename
    if dest.exists():
        return dest

    from analysis.runner import modal_backend_available

    if modal_backend_available():
        from analysis.bucket_io import raw_prefix
        from analysis.infra import BUCKET

        dest.parent.mkdir(parents=True, exist_ok=True)
        src = f"s3://{BUCKET}/{raw_prefix(dataset_id)}/{filename}"
        try:
            subprocess.run(["aws", "s3", "cp", src, str(dest)], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            logger.info("S3 copy of %s failed (%s); unpacking Zenodo dataset", filename, exc)
        else:
            return dest

    ensure_dataset(dataset_id, dest_dir)
    if not dest.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_id.strip('/')}' has no raw file named '{filename}'."
        )
    return dest
