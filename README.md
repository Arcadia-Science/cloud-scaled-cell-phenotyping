[![Arcadia Pub](https://img.shields.io/badge/Arcadia-Pub-596F74.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDI3LjcuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCA0My4yIDQwLjQiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDQzLjIgNDAuNDsiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8c3R5bGUgdHlwZT0idGV4dC9jc3MiPgoJLnN0MHtmaWxsOm5vbmU7c3Ryb2tlOiNGRkZGRkY7c3Ryb2tlLXdpZHRoOjI7c3Ryb2tlLWxpbmVqb2luOmJldmVsO3N0cm9rZS1taXRlcmxpbWl0OjEwO30KPC9zdHlsZT4KPGc+Cgk8cG9seWdvbiBjbGFzcz0ic3QwIiBwb2ludHM9IjIxLjYsMyAxLjcsMzcuNCA0MS41LDM3LjQgCSIvPgoJPGxpbmUgY2xhc3M9InN0MCIgeDE9IjIxLjYiIHkxPSIzIiB4Mj0iMjEuNiIgeTI9IjI3LjMiLz4KCTxwb2x5bGluZSBjbGFzcz0ic3QwIiBwb2ludHM9IjEyLjIsMTkuNCAyNC42LDMwLjEgMjQuNiwzNy40IAkiLz4KCTxsaW5lIGNsYXNzPSJzdDAiIHgxPSIxNy42IiB5MT0iMTYuNyIgeDI9IjE3LjYiIHkyPSIyNC4xIi8+Cgk8bGluZSBjbGFzcz0ic3QwIiB4MT0iMjguNiIgeTE9IjE1LjIiIHgyPSIyMS43IiB5Mj0iMjIuMSIvPgoJPHBvbHlsaW5lIGNsYXNzPSJzdDAiIHBvaW50cz0iNi44LDI4LjcgMTkuNSwzNC40IDE5LjUsMzcuNCAJIi8+Cgk8bGluZSBjbGFzcz0ic3QwIiB4MT0iMzQuOCIgeTE9IjI1LjgiIHgyPSIyNC42IiB5Mj0iMzYuMSIvPgoJPGxpbmUgY2xhc3M9InN0MCIgeDE9IjI5LjciIHkxPSIyMi4yIiB4Mj0iMjkuNyIgeTI9IjMwLjkiLz4KPC9nPgo8L3N2Zz4K)](https://doi.org/10.57844/arcadia-fe2a-711e)
[![DOI](https://zenodo.org/badge/1315519366.svg)](https://doi.org/10.5281/zenodo.23547346)

# Scaling single-cell phenotyping with Cellpose and cloud GPU compute

This code repository contains or points to all materials required for creating and hosting the publication entitled, *"[Scaling single-cell phenotyping with Cellpose and cloud GPU compute](https://doi.org/10.57844/arcadia-tb5a-ft98)"*.

The publication is hosted at [this URL](https://arcadia-science.github.io/cloud-scaled-cell-phenotyping/).

![Overview of the cloud-scaled, single-cell phenotyping pipeline. Raw microscopy images are segmented with Cellpose-SAM across a GPU cluster; per-cell measurements and a plate map yield the single-cell dataset used for morphology analysis.](assets/fig1-workflow.png)

## Data Description

A two-species yeast panel of brightfield ND2 images: diploid *S. cerevisiae* BY4743 and *S. pombe* SP286, imaged across 16 wells of a 384-well plate (2×2 fields per well; 64 fields of view). The public archive ([10.5281/zenodo.23377605](https://doi.org/10.5281/zenodo.23377605); 764 MB compressed, 2.3 GB unpacked) contains the raw ND2s and precomputed `derived/` artifacts.

Without Modal + S3 access, the notebook downloads and extracts that archive into `datasets/<id>/` on the first local run. If the automatic Zenodo fetch errors or times out, download `<id>.zip` from the record into the same directory and re-run the cell; that bypasses the automatic download. For this pub, `<id>` is `20260611_141152_330`:

```
datasets/20260611_141152_330/
├── raw/                  # brightfield ND2s
└── derived/
    ├── masks/            # label TIFFs
    ├── overlays/         # outline JPEGs
    └── measurements/     # per-cell CSVs
```

## Reproduce

The publication is a Jupyter notebook (`index.ipynb`) rendered with Quarto. The hosted page is that notebook's executed output, so reproducing the pub means executing the notebook. Instructions for doing so, including environment setup, backend choice (Modal + S3 vs local CPU), and the `make execute` / `make preview` commands are available in [SETUP.qmd](pages/SETUP.qmd).

## Contribute

Please see [CONTRIBUTING.qmd](pages/CONTRIBUTING.qmd).
