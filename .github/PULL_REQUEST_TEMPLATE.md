## What this changes

<!-- One or two sentences. -->

## Evidence

- [ ] Each new or changed formula cites its source (paper, equation, page) in the docstring.
- [ ] Published worked examples are reproduced at the precision printed, or the pull request
      says which could not be and why.
- [ ] There is a second, independent check (exact integration, seeded simulation, brute-force
      reference or limiting case), with each tolerance explained.
- [ ] Inputs that would give a plausible but wrong answer raise an error.

## Checks

- [ ] `uv run ruff check .` and `uv run ruff format --check .`
- [ ] `uv run mypy src tests`
- [ ] `uv run pytest --cov --cov-fail-under=100`
- [ ] `uv run python docs/build_api.py` if a public function or docstring changed
