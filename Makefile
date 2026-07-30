.PHONY: lint
lint:
	uv run ruff check --exit-zero .
	uv run ruff format --check .

.PHONY: format
format:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

.PHONY: test
test:
	uv run pytest -v .

.PHONY: execute
execute:
	# Project-wide render without executing code cells.
	# Instead, rely on pre-computed results present in `_freeze/`.
	uv run quarto render

	# Now, render `index.ipynb` with code execution.
	# This will populate `_freeze/index/` with pre-computed results.
	uv run quarto render index.ipynb --execute

.PHONY: preview
preview:
	uv run quarto preview

.PHONY: execute-demo
execute-demo:
	# This command is only necessary when changes are made to the demo notebook
	# for development of the notebook pub template.
	uv run quarto render examples/demo.ipynb --execute

.PHONY: update-theme
update-theme:
	quarto update extension Arcadia-Science/notebook-pub-theme --no-prompt
