"""Single-cell segmentation of brightfield ND2s on Modal (Cellpose-SAM).

Kept intentionally minimal (no re-exports) so that importing the package - or any
submodule, which runs this ``__init__`` - never eagerly pulls in numpy/pandas/
cellpose. This matters inside Modal: the lightweight ``list_images`` container runs
on an image without those deps, and a re-export chain here would make merely
importing ``analysis.app`` fail at boot. Import the public API by path instead,
e.g. ``from analysis.runner import run_segmentation``.
"""

__version__ = "0.1.0"


def polo_if_marco(val: str) -> str | None:
    """Placeholder kept for the Quarto syntax demo in ``examples/demo.ipynb``."""
    return "polo" if val == "marco" else None
