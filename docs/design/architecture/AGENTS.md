# Repository Guidelines

## Project Structure & Module Organization
- `infopilot.py` orchestrates the CLI scan/train/chat flow and wires the supporting modules.
- `core/data_pipeline/filefinder.py` performs cross-platform scans and writes candidate metadata to CSV during Step 1.
- `core/data_pipeline/pipeline.py` builds the cleaned corpus and topic model artifacts stored in `data/`.
- `core/search/retriever.py` and `core/conversation/lnp_chat.py` load the trained model, manage `data/cache/`, and drive the interactive chat.
- `core/data_pipeline/policies/` hosts smart folder policy schemas, examples, and the runtime policy engine used to scope scanning/indexing.
- `core/agents/meeting/` contains the meeting agent MVP (transcription/summarisation stubs) and will evolve with future cycles.
- `core/agents/photo/` contains the photo agent MVP scaffold (tagging, duplicate detection, best-shot reporting).
- `core/infra/` collects shared infrastructure helpers such as offloading strategies, audit logging, and model selection utilities.
- `data/` hosts generated corpora and models; `models/` is reserved for packaged exports.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (`.\.venv\Scripts\activate` on Windows) isolates dependencies.
- `pip install -r requirements.txt` installs the baseline extractors (필요 시 번역 옵션을 위한 `deep-translator` 포함).
- `python scripts/pipeline/infopilot.py scan --out data/found_files.csv` scans mounted drives and writes the discovery CSV.
- `python scripts/pipeline/infopilot.py train --scan_csv data/found_files.csv --corpus data/corpus.parquet` builds the corpus and updates `data/topic_model.joblib`.
- `python scripts/pipeline/infopilot.py chat --model data/topic_model.joblib --corpus data/corpus.parquet --cache data/cache` launches the semantic (embedding) chat. Use `--translate` for query translation, `--show-translation` to display cached snippet translations, and raise `--lexical-weight` when you deliberately need BM25 assistance. Combine with `--rerank` (and optional depth/model) for cross-encoder refinement when higher precision is required.
- `python scripts/pipeline/infopilot.py watch --cache data/cache --corpus data/corpus.parquet --model data/topic_model.joblib` runs the watchdog-based incremental pipeline so new/edited files are extracted, embedded, and pushed into the FAISS/BM25 index within seconds.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, type hints, and concise docstrings on public entry points.
- Use `snake_case` for functions, `UPPER_SNAKE_CASE` for constants, and `CamelCase` for classes such as `FileFinder`.
- Keep console output succinct and reuse the existing emoji/logging tone for user feedback.

## Testing Guidelines
- Adopt `pytest` for new coverage and place specs under `tests/` mirroring module names.
- Mock filesystem access or use temporary directories when exercising `FileFinder` or `Retriever`.
- Run `pytest -q` before pushing and rely on synthetic fixture CSVs instead of real customer data.
- When a project virtualenv uses Python ≥3.12, install dependencies with `pip install -r requirements.txt` (the `pyhwp` extractor is skipped automatically). For environments without GPUs, prefer CPU-only Torch wheels (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cpu`) to avoid downloading large CUDA runtimes.

## Commit & Pull Request Guidelines
- Write imperative, present-tense commit subjects (e.g., `Improve PDF extractor handling`) with short context in the body.
- Reference issues via `Refs #123` or `Fixes #123`, and exclude bulky artifacts from `data/` or `data/cache/`.
- PRs should cover scope, manual verification (`scan`, `train`, `chat`), dependency changes, and attach console screenshots when UX shifts.

## Security & Data Handling
- Scrub or anonymize sensitive file paths in logs, tests, and shared troubleshooting assets.
- Clear or regenerate `data/cache/` before distributing debug bundles to prevent leaking previews.
- Add local-only artifacts to `.gitignore` entries before committing scanned data or corpora.
