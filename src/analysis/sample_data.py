"""Fetch example ND2s from a public Zenodo archive for the local demo path.

Internal readers run the scaled Modal pipeline against the private S3 bucket. Every
other reader has no bucket access, so the notebook's local fallback pulls a few
example brightfield ND2s from a public Zenodo record over HTTPS (stdlib only, no
extra deps) and caches them under a gitignored scratch folder.

The Zenodo record is resolved (in priority order) from the ``CSCP_ZENODO_RECORD_ID``
environment variable, then the ``ZENODO_RECORD_ID`` constant below. Set one of them
once the archive is published; until then the helper raises a clear error rather
than silently failing.
"""

from __future__ import annotations
import json
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger("analysis.sample_data")

# Zenodo record holding the public example datasets. Fill this in once the archive
# is published (or set the CSCP_ZENODO_RECORD_ID env var to override without a code
# change). Kept as a placeholder so the local path fails loudly instead of guessing.
ZENODO_RECORD_ID: str | None = None

_ZENODO_API = "https://zenodo.org/api/records/{record_id}"


class SampleDataUnavailableError(RuntimeError):
    """Raised when the public sample archive is not configured or reachable."""


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


def fetch_sample_nd2s(
    dataset_id: str,
    dest_dir: Path | str,
    *,
    limit: int | None = 3,
) -> list[Path]:
    """Download up to ``limit`` example ND2s for ``dataset_id`` from Zenodo.

    Files whose names contain ``dataset_id`` are preferred; if none match, every
    ``.nd2`` in the record is considered. Already-downloaded files of the expected
    size are reused, so re-running the notebook does not re-download.

    Args:
        dataset_id: Dataset identifier, used to filter the record's files.
        dest_dir: Local directory to cache the downloaded ND2s in.
        limit: Max number of files to fetch (``None`` for all).

    Returns:
        Local paths to the downloaded ND2 files, sorted by name.
    """
    record_id = _resolve_record_id()
    dest_dir = Path(dest_dir)

    files = _list_record_files(record_id)
    nd2s = [f for f in files if str(f.get("key", "")).lower().endswith(".nd2")]
    matched = [f for f in nd2s if dataset_id in str(f.get("key", ""))]
    selected = sorted(matched or nd2s, key=lambda f: str(f.get("key", "")))
    if limit is not None:
        selected = selected[:limit]

    if not selected:
        raise SampleDataUnavailableError(
            f"Zenodo record '{record_id}' has no .nd2 files matching dataset '{dataset_id}'."
        )

    local_paths: list[Path] = []
    for entry in selected:
        key = str(entry["key"])
        expected_size = entry.get("size")
        url = entry.get("links", {}).get("self") or entry.get("links", {}).get("download")
        if not url:
            logger.warning("skipping %s: no download link in record metadata", key)
            continue

        local = dest_dir / key
        if local.exists() and (expected_size is None or local.stat().st_size == expected_size):
            logger.info("using cached %s", local)
        else:
            _download(url, local)
        local_paths.append(local)

    if not local_paths:
        raise SampleDataUnavailableError(
            f"Failed to download any sample ND2s from Zenodo record '{record_id}'."
        )
    return sorted(local_paths)
