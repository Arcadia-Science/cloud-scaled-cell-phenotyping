# Environment Setup Guide

This document explains how to set up the development environment for working with and reproducing notebook publications.

## Setup the code environment

This repository uses [uv](https://docs.astral.sh/uv/) to manage the computational and build environment. If you don't have it installed (check with `uv --version`), you can find operating system-specific instructions for installing it [here](https://docs.astral.sh/uv/getting-started/installation/).

When you're ready, run the following command to create the virtual environment and install all dependencies (including the local `src/` package, in editable mode):

```bash
uv sync
```

This creates a `.venv/` in the project directory and installs everything pinned in `uv.lock`. uv automatically uses Python 3.12 as specified in `.python-version`.

(As you introduce dependencies to your publication, add them with `uv add <package>`. Runtime dependencies go in `[project.dependencies]` and development-only tooling in the `dev` group of `pyproject.toml`.)

Prefix commands with `uv run` to run them inside the environment (for example, `uv run quarto preview`), or activate the environment directly with `source .venv/bin/activate`.

Finally, install the [pre-commit](https://pre-commit.com/) hooks. This is optional but recommended:

```bash
uv run pre-commit install
```

Test your installation with `make preview`. Your pub will open up in your browser.

Afterwards, create a branch to work on (don't commit to `main` directly).

## Pub Development

When working with Jupyter notebooks:

1. Edit `index.ipynb` as your main publication file
2. Use `make preview` to see changes in real-time
3. Run `make execute` before committing to update `_freeze/` with execution results

## Other Commands

The Makefile provides several useful commands for development:

- `make lint`: Run linting checks using ruff
- `make format`: Automatically format code using ruff
- `make pre-commit`: Run all pre-commit hooks
- `make test`: Run pytest tests, if they exist
