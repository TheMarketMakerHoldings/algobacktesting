# Releasing

A release is cut by pushing a version tag. The `Release` workflow then runs the tests,
builds the package, publishes it to PyPI and opens a GitHub release with the built files
attached. Nothing is uploaded by hand, and no password or token is stored anywhere.

## One-time setup

1. **PyPI.** Under *Publishing*, add a pending trusted publisher: project
   `algobacktesting`, owner `TheMarketMakerHoldings`, repository `algobacktesting`,
   workflow `release.yml`, environment `pypi`.
2. **GitHub environment.** Settings → Environments → new environment `pypi`. Adding
   yourself as a required reviewer means every publish waits for one click.
3. **GitHub Pages.** Settings → Pages → deploy from branch `main`, folder `/docs`.
4. **Private vulnerability reporting.** Settings → Security → turn it on, so the link in
   `SECURITY.md` works.

## Every release

1. Pick the version number ([Semantic Versioning](https://semver.org/)).
2. Set it in three places, then refresh the lockfile:
   - `pyproject.toml` (`version`)
   - `src/algobacktesting/__init__.py` (`__version__`)
   - `CITATION.cff` (`version`, and `date-released`)

   ```bash
   uv lock
   ```

   Tests fail if the three disagree, and CI fails if the lockfile is stale.
3. In `CHANGELOG.md`, rename `[Unreleased]` to `[x.y.z] - YYYY-MM-DD` and start a new,
   empty `[Unreleased]` above it.
4. Regenerate the API reference, which prints the version:

   ```bash
   uv run python docs/build_api.py
   ```

5. First release only: in `README.md`, remove the pre-release status note, replace "Not
   yet published" under *Installation*, and add the PyPI version badge.
6. Run every check:

   ```bash
   uv run ruff check .
   ```

   ```bash
   uv run ruff format --check .
   ```

   ```bash
   uv run mypy src tests
   ```

   ```bash
   uv run pytest --cov --cov-fail-under=100
   ```

7. Commit, push, and wait for CI to pass on `main`.
8. Tag that commit and push the tag:

   ```bash
   git tag -a v0.1.0 -m "v0.1.0"
   ```

   ```bash
   git push origin v0.1.0
   ```

9. Watch the `Release` workflow. When it finishes, install the new version in a clean
   environment and run one known number:

   ```bash
   uv run --isolated --no-project --with algobacktesting python -c "import algobacktesting as a; print(a.__version__, round(a.expected_max_sharpe(100, 1/500), 4))"
   ```

   It should print the new version and `0.1132`.

## If something goes wrong

- **The workflow fails before publishing.** Nothing reached PyPI. Fix the problem, delete
  the tag (`git push --delete origin v0.1.0`, then `git tag -d v0.1.0`), and tag again.
- **A broken version reached PyPI.** A version number can never be reused. Yank the
  release on PyPI and publish a fixed patch version.
