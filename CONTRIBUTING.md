# Contributing

Thanks for looking. This project is small on purpose. Its value is that every number it
produces can be trusted, so the bar for changes is evidence, not volume.

## Setup

You need [uv](https://docs.astral.sh/uv/) and, for the website test, Node.js.

```bash
uv sync
uv run pre-commit install
```

## Checks

Run these before opening a pull request. CI runs the same ones on Python 3.11 to 3.14.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov --cov-fail-under=100
```

If you change a public function or its docstring, regenerate the API reference:

```bash
uv run python docs/build_api.py
```

## The verification standard

A statistical function that is subtly wrong is worse than no function. Every formula here
is held to four rules:

1. **Cite the source.** The docstring names the paper, the equation and the page.
2. **Reproduce the published numbers.** Where the source prints a worked example, a test
   reproduces it at the precision printed. Tolerances are never loosened to make a test
   pass. If a printed number can't be reproduced, say so in the pull request.
3. **Check it a second way.** Use a method that doesn't share the formula's code: exact
   integration, a seeded simulation, a brute-force reference, or a limiting case with a
   known answer. State each tolerance and why it is that size.
4. **Fail loudly.** Inputs that would produce a plausible but wrong number raise a clear
   error instead.

All randomness takes an explicit seed, so every test is deterministic.

## Where things are

| Path | What |
| --- | --- |
| `src/algobacktesting/` | The library. NumPy is the only runtime dependency. |
| `tests/` | Paper targets, independent checks, property tests, validation. |
| `docs/explainers/` | Plain-English guides, one per module. |
| `docs/paper/` | The verification report and the script that regenerates its numbers. |
| `docs/index.html` | The project website. Its calculator is tested against the library. |
| `examples/` | Runnable demonstrations. |

## Scope

Ideas beyond the current release are listed in `BACKLOG.md`. Please open an issue before
building one, so the source and the verification plan can be agreed first.

## Reporting a wrong number

That is the most useful report this project can get. Please include the inputs, the output
you got, the output you expected, and where the expected value comes from.
