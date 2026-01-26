# Repository Guidelines

## Project Structure & Module Organization
- `tail30_selector/` contains the core package, with submodules for `datasource/`, `indicators/`, `strategy/`, `backtest/`, and shared helpers in `utils/`.
- `tests/` holds unit tests, currently focused on indicator alignment logic.
- Root files include `README.md` and `requirements.txt`. Output files are written to `output/` at runtime.
 - CLI entrypoints live in `tail30_selector/cli.py` and `tail30_selector/__main__.py` for `python -m` usage.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs runtime dependencies.
- `python -m tail30_selector --date 2026-01-20 --mode realtime --universe all --datasource akshare` runs the selector with CLI flags.
- Use `--mode backtest` for historical evaluation; CSV results land in `output/selected_YYYYMMDD.csv`.
 - For Tushare, pass `--token` or set `TS_TOKEN` before running.

## Coding Style & Naming Conventions
- Use 4-space indentation and keep functions small and focused.
- Prefer `snake_case` for files, modules, and variables; classes should be `PascalCase`.
- No auto-formatter is configured; keep imports grouped and avoid unused variables.
 - Favor explicit DataFrame columns (e.g., `datetime`, `close`, `volume`) and document any new fields.

## Testing Guidelines
- Tests use `pytest` conventions (`test_*.py`, `test_*` functions).
- Run `pytest` from the repo root.
- Add tests for indicator math and alignment edge cases when changing selection logic.
 - Keep synthetic data small and deterministic to avoid flaky results.

## Commit & Pull Request Guidelines
- Existing history uses short, imperative English messages (e.g., “Refine volume shape…”). Follow the same style without extra prefixes.
- PRs should include a concise summary, reproducible CLI command, and note which data source (`akshare` or `tushare`) was used.
- If output changes, attach a short log snippet or sample CSV row.
 - Mention any performance impacts or added dependencies in the PR description.

## Architecture Overview
- The strategy pipeline applies indicator filters (Step1-7) in sequence and emits explanations alongside the Top 20 results.
- Data sources are pluggable via `datasource/` implementations; keep new sources aligned with the base interface.

## Security & Configuration Tips
- Tushare requires `TS_TOKEN` (or `--token`). Avoid committing tokens or local output files.

## Agent-Specific Instructions
- When automating changes, prefer small, focused edits and update tests when behavior changes.
