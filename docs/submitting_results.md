# Submitting Results

ClinDoc-Bench-IN keeps the paper benchmark frozen. New results should be packaged as community submissions instead of being merged into the frozen benchmark directories.

## Submission Layout

```text
community/submissions/<submission_name>/
├── metadata.yaml
├── predictions/
├── runtime.csv
└── README.md
```

## Track-Specific Prediction Formats

### Raw OCR

- one file per document
- filename: `<document_id>.txt`
- text encoding: UTF-8

### Direct VLM Or Hybrid

- one file per document
- filename: `<document_id>.json`
- must validate as `CanonicalRawDoc`

## Metadata

Required fields in `metadata.yaml`:

- `submission_name`
- `track`
- `model_name`
- `model_version`
- `provider`
- `license`
- `hardware`
- `benchmark_version`

See the template:

- [community/submissions/template/metadata.yaml](../community/submissions/template/metadata.yaml)

## Runtime File

`runtime.csv` must contain:

- `document_id`
- one runtime field: `runtime_seconds`, `latency_ms`, or `processing_time_ms`

## Validation

Validate the package before scoring it:

```bash
python scripts/validate_submission.py \
    --submission-dir community/submissions/your_submission
```

Successful validation confirms:

- metadata is complete
- all expected benchmark documents are present
- runtime coverage is complete
- structured predictions follow the public schema, or OCR text files are non-empty

## Structured Scoring

For `direct_vlm` or `hybrid` submissions, generate an evaluation manifest and run the structured benchmark:

```bash
python scripts/validate_submission.py \
    --submission-dir community/submissions/your_submission \
    --write-benchmark-manifest experiments/template_eval_manifest.csv

python -m src.cli.benchmark \
    --manifest experiments/template_eval_manifest.csv \
    --config configs/benchmark_defaults.yaml \
    --output-dir experiments/template_eval_reports
```

## Raw OCR Scoring

For `raw_ocr` submissions, generate an OCR handoff CSV and run the OCR evaluator:

```bash
python scripts/validate_submission.py \
    --submission-dir community/submissions/your_raw_ocr_submission \
    --write-ocr-handoff experiments/template_raw_ocr_handoff.csv

python scripts/benchmark_raw_ocr_outputs.py \
    --handoff experiments/template_raw_ocr_handoff.csv \
    --manifest benchmark_v2/data/benchmark_manifest_v2.csv \
    --engine your_ocr_engine_name \
    --output-dir experiments/template_raw_ocr_reports
```

## What This Does Not Do Yet

The validator standardizes packaging and checks readiness, but it does not yet auto-publish a community leaderboard or auto-open pull requests. Those are roadmap items, not already-finished features.
