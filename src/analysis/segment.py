"""Modal-free single-cell segmentation logic (Cellpose-SAM via arcadia-microscopy-tools).

Kept import-light so it can be run in a notebook or unit-tested directly: the heavy
dependencies (arcadia_microscopy_tools / cellpose / torch / nd2) are imported lazily
inside the functions, so importing this module in the notebook env stays cheap.

The pipeline reads a single-channel brightfield ND2, applies a difference-of-Gaussians
bandpass, contrast-stretches the result, segments cells with Cellpose-SAM, drops
border-touching cells, and produces:
  * an integer label mask,
  * a per-cell measurements table (regionprops, converted to microns when the pixel
    size is known), and
  * an RGB outline overlay for quick visual QC.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class SegmentConfig:
    """Cellpose-SAM parameters (forwarded to ``SegmentationModel``).

    Cellpose-SAM (cellpose v4) uses a single generalist model. ``diameter`` is the
    expected cell diameter in pixels; ``flow_threshold`` / ``cellprob_threshold`` are
    the usual quality knobs (higher cellprob -> fewer, more confident masks).
    ``percentile_low``/``percentile_high`` set the contrast-stretch window applied
    after the difference-of-Gaussians bandpass and before segmentation.
    ``dog_sigma_low``/``dog_sigma_high`` are the Gaussian sigmas (pixels) subtracted
    to form that bandpass (``None`` skips it).
    """

    diameter: float | None = None  # None -> Cellpose-SAM default (30 px)
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    percentile_low: float = 0.1
    percentile_high: float = 99.9
    dog_sigma_low: float | None = 1.6
    dog_sigma_high: float | None = 16.0
    use_gpu: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentationResult:
    """Outputs of segmenting a single image."""

    mask: np.ndarray  # uint32 label image (background=0, cells 1..N)
    overlay_rgb: np.ndarray  # (H, W, 3) uint8 outline overlay
    measurements: pd.DataFrame  # one row per cell
    n_cells: int
    pixel_size_um: float | None
    channel_name: str


def _load_brightfield_plane(local_path: Path | str) -> tuple[np.ndarray, Any, float | None]:
    """Load the single-channel 2D image, its Channel object, and pixel size from an ND2.

    Inputs are expected to be single-channel 2D brightfield images; any singleton
    axes (e.g. a stray channel axis) are squeezed away as a guard.
    """
    from arcadia_microscopy_tools import MicroscopyImage

    image = MicroscopyImage.from_nd2_path(Path(local_path))

    channel = image.channels[0]

    raw = np.asarray(image.get_channel_intensities(channel))
    raw2d = np.squeeze(raw)
    if raw2d.ndim != 2:
        raise ValueError(
            f"Expected a single-channel 2D image but got shape {raw.shape} "
            f"(squeezed to {raw2d.shape}) for {Path(local_path).name}."
        )

    pixel_size_um: float | None = None
    try:
        resolution = image.metadata.instrument.channel_metadata_list[0].resolution
        pixel_size_um = float(resolution.xy_step_um)
    except (AttributeError, IndexError, TypeError):
        pixel_size_um = None

    return raw2d, channel, pixel_size_um


def _measurements_dataframe(
    mask: np.ndarray,
    raw2d: np.ndarray,
    channel: Any,
    pixel_size_um: float | None,
) -> pd.DataFrame:
    """Build a per-cell measurements table via regionprops (microns when available)."""
    import pandas as pd
    from arcadia_microscopy_tools.masks import SegmentationMask

    # Edge cells were already removed upstream, so don't drop again here.
    seg = SegmentationMask(
        mask_image=mask.astype(np.int64),
        intensity_image_dict={channel: raw2d},
        remove_edge_cells=False,
    )

    if pixel_size_um is not None:
        properties = seg.convert_properties_to_microns(pixel_size_um)
    else:
        properties = seg.cell_properties

    return pd.DataFrame(properties)


def _difference_of_gaussians(
    image: np.ndarray, *, sigma_low: float, sigma_high: float
) -> np.ndarray:
    """Bandpass via Gaussian(sigma_low) - Gaussian(sigma_high)."""
    from scipy.ndimage import gaussian_filter

    if sigma_low <= 0 or sigma_high <= 0:
        raise ValueError(
            f"DoG sigmas must be positive (got sigma_low={sigma_low}, sigma_high={sigma_high})."
        )
    if sigma_high <= sigma_low:
        raise ValueError(
            f"DoG sigma_high must be greater than sigma_low "
            f"(got sigma_low={sigma_low}, sigma_high={sigma_high})."
        )

    img = image.astype("float32", copy=False)
    return gaussian_filter(img, sigma_low) - gaussian_filter(img, sigma_high)


def _to_uint8_grayscale(
    yx: np.ndarray, *, low_pct: float = 1.0, high_pct: float = 99.0
) -> np.ndarray:
    """Contrast-stretch a (possibly float) 2D image to ``uint8`` for display."""
    img = yx.astype("float32")
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, dtype="uint8")

    lo, hi = np.percentile(finite, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype="uint8")

    scaled = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).round().astype("uint8")


def outline_overlay(
    background: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[float, float, float] = (1.0, 1.0, 0.0),
    opacity: float = 0.6,
) -> np.ndarray:
    """Render mask outlines over the brightfield image as an RGB ``uint8`` image."""
    from cellpose.utils import masks_to_outlines
    from scipy.ndimage import binary_dilation

    gray = _to_uint8_grayscale(background)
    rgb = np.stack([gray, gray, gray], axis=-1)

    # Make the outlines thicker than the default 1-pixel boundary.
    outlines = binary_dilation(masks_to_outlines(mask), structure=np.ones((2, 2)))
    if outlines.any():
        color_u8 = np.array([c * 255.0 for c in color], dtype="float32")
        blended = (1.0 - opacity) * rgb[outlines].astype("float32") + opacity * color_u8
        rgb[outlines] = np.clip(blended, 0, 255).round().astype("uint8")

    return rgb


def segment_nd2(local_path: Path | str, *, config: SegmentConfig) -> SegmentationResult:
    """Segment a single-channel brightfield ND2 into a label mask + measurements + overlay."""
    import torch
    from arcadia_microscopy_tools.model import SegmentationModel
    from arcadia_microscopy_tools.operations import rescale_by_percentile
    from cellpose.utils import remove_edge_masks

    raw2d, channel, pixel_size_um = _load_brightfield_plane(local_path)
    channel_name = getattr(channel, "name", str(channel))

    preprocessed = raw2d.astype("float32")
    if config.dog_sigma_low is not None or config.dog_sigma_high is not None:
        if config.dog_sigma_low is None or config.dog_sigma_high is None:
            raise ValueError("dog_sigma_low and dog_sigma_high must both be set, or both be None.")
        preprocessed = _difference_of_gaussians(
            preprocessed,
            sigma_low=config.dog_sigma_low,
            sigma_high=config.dog_sigma_high,
        )

    normalized = rescale_by_percentile(
        preprocessed,
        (config.percentile_low, config.percentile_high),
    )

    device = None if config.use_gpu else torch.device("cpu")
    model = SegmentationModel(
        default_cell_diameter_px=config.diameter if config.diameter is not None else 30.0,
        default_flow_threshold=config.flow_threshold,
        default_cellprob_threshold=config.cellprob_threshold,
        device=device,
    )
    mask = model.segment(normalized.astype("float64"))

    # Drop cells touching the image border; change_index relabels survivors to 1..N.
    mask = remove_edge_masks(mask, change_index=True).astype("uint32")

    n_cells = int(mask.max())
    overlay_rgb = outline_overlay(raw2d, mask)

    if n_cells > 0:
        measurements = _measurements_dataframe(mask, raw2d, channel, pixel_size_um)
    else:
        import pandas as pd

        measurements = pd.DataFrame()

    return SegmentationResult(
        mask=mask,
        overlay_rgb=overlay_rgb,
        measurements=measurements,
        n_cells=n_cells,
        pixel_size_um=pixel_size_um,
        channel_name=channel_name,
    )
