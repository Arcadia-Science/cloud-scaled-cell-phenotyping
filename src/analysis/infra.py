"""Shared Modal infrastructure: bucket mount, secret, and container images.

The segmentation app imports its S3 wiring and Python environment from here so the
same config is reused wherever the app is launched (notebook, CLI, or a deploy).

Design notes:
  * Input ND2s and derived artifacts live in the *same* bucket, so a single
    read/write ``CloudBucketMount`` is enough (mounted at ``MOUNT_PATH``).
  * The heavy compute image is built from this project's ``pyproject.toml`` +
    ``uv.lock`` via ``uv_sync(extras=["worker"])``, so its dependency versions are
    pinned/reproducible. The ``worker`` extra carries the segmentation stack
    (cellpose + torch via ``arcadia-microscopy-tools[segmentation]``) plus the
    ND2/TIFF/JPEG I/O deps.
  * The light ``meta_image`` only carries this package's source; the listing call
    uses stdlib only (rglob over the mount), so it cold-starts a small container.
  * ``add_local_python_source("analysis")`` ships this package's latest source into
    every container. Modal requires the ``add_local_*`` layer to be last, so it is
    applied after any build steps.
"""

from __future__ import annotations
import os
from pathlib import Path

import modal

# Absolute path to this project's root (the directory holding pyproject.toml +
# uv.lock). uv_sync()'s ``uv_project_dir`` defaults to the *caller's* working
# directory, so launching from a sibling repo (or a notebook elsewhere) would make
# Modal copy the wrong pyproject.toml. Anchoring to __file__ keeps the build
# independent of where ``run_segmentation`` is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Single bucket holds both the raw ND2 inputs and the derived outputs.
BUCKET = os.environ.get("CSCP_BUCKET", "cloud-scaled-cell-phenotyping")
SECRET = modal.Secret.from_name("aws-s3-access-secret")
MOUNT_PATH = "/data"

# Default S3 prefix (relative to the bucket root) for derived artifacts.
DEFAULT_OUTPUT_PREFIX = "derived/segmentation"

DEFAULT_GPU_TYPE = "L4"
FUNCTION_TIMEOUT_S = 60 * 30

# Parallelism ceiling (max concurrent GPU containers per function).
DEFAULT_GPU_MAX_CONTAINERS = 10

# One read/write mount for the shared bucket.
data_mount = modal.CloudBucketMount(bucket_name=BUCKET, secret=SECRET)

# Heavy compute image: built from the locked project with the ``worker`` extra
# (cellpose + torch via arcadia-microscopy-tools[segmentation], plus nd2/tifffile/
# pillow). uv_sync installs the exact versions from uv.lock for reproducible builds.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(_PROJECT_ROOT), extras=["worker"])
    .add_local_python_source("analysis")
)

# Lightweight image for listing bucket contents (pure-stdlib rglob over the
# mount), so it avoids cold-starting the multi-GB compute image.
meta_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("analysis")
