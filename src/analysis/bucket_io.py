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
