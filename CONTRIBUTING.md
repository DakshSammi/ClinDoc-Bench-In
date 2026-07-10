# Contributing

Thank you for helping improve ClinDoc-Bench-IN.

## Ground Rules

- Do not modify frozen benchmark outputs under `benchmark_v2/final_day_freeze_20260709/`.
- New experiments must write to a new output directory.
- Do not commit secrets, raw patient identifiers, or unanonymized example images.
- Keep model provenance explicit: provider, model name, version/date, runtime, and failure handling.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Checks

```bash
ruff check .
pytest
python scripts/generate_publication_assets.py
```

## Contribution Types

We welcome:

- bug fixes
- documentation improvements
- new OCR lanes
- new direct VLM lanes
- new hybrid pipelines
- dataset onboarding helpers
- evaluation audits
- submission tooling
- paper asset fixes that do not alter frozen benchmark outcomes

## Adding A Model Lane

1. Write outputs to a new directory outside the frozen benchmark tree.
2. Keep one output file per document.
3. Validate structured JSON before scoring, or validate OCR text packaging before OCR scoring.
4. Record runtime and model provenance.
5. Use `scripts/validate_submission.py` if you are packaging results as a community-style submission.

## Pull Requests

Pull requests should include:

- summary of the change
- whether frozen benchmark outputs were untouched
- commands run for validation
- privacy or data handling implications
- any new docs, templates, or workflow expectations introduced
