# ClinDoc-Bench-IN

Open benchmark for Indian prescription OCR, direct vision-language extraction, and hybrid OCR-to-LLM pipelines. It ships frozen publication results, reusable evaluation code, and a community-ready submission format.

[![Dataset](https://img.shields.io/badge/Dataset-90%20patients%20%7C%20125%20documents%20%7C%20150%20images-2f6f6d)](#dataset)
[![Benchmark](https://img.shields.io/badge/Benchmark-OCR%20%7C%20Direct%20VLM%20%7C%20Hybrid-44546a)](#benchmark-tracks)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab)](#installation)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)
[![Frozen Benchmark](https://img.shields.io/badge/Frozen-v1.0%20read--only-orange)](#frozen-benchmark)
[![BDA 2026](https://img.shields.io/badge/BDA%202026-Submitted-purple)](#citation)

ClinDoc-Bench-IN is both a frozen paper benchmark and an open repository for benchmarking new OCR, VLM, and hybrid extraction systems against the same evaluation logic.

The frozen benchmark under `benchmark_v2/final_day_freeze_20260709/` is read-only. New experiments, community submissions, and future benchmark revisions must live outside that directory.

## Overview

- Reproduce the paper benchmark from frozen reports and assets.
- Run a new model lane without modifying frozen outputs.
- Validate a community submission before scoring it.
- Benchmark your own dataset using the same canonical JSON schema and evaluation rules.

## Repository Modes

ClinDoc-Bench-IN has three working modes:

| Mode | Purpose | Mutability |
| --- | --- | --- |
| Frozen benchmark | Canonical BDA 2026 reference results, provenance, statistics, and paper assets | Read-only |
| Experiments | New local runs, ablations, debugging, and unpublished comparisons | Writable |
| Community submissions | Standardized external model submissions and validation artifacts | Writable |

Recommended working areas:

```text
benchmark_v2/final_day_freeze_20260709/   # frozen publication benchmark
experiments/                              # new local experiments
community/submissions/                    # community-formatted submissions
paper_assets/                             # regenerated publication assets from frozen CSVs
```

## Installation

```bash
git clone <repository-url>
cd prescription_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For Conda users:

```bash
conda env create -f environment.yml
conda activate clindoc-bench-in
```

## Quick Start

Inspect the frozen summary:

```bash
python scripts/read_frozen_summary.py
```

Regenerate publication tables and figures from frozen CSVs without touching benchmark outputs:

```bash
python scripts/generate_publication_assets.py
```

Validate a directory of canonical JSON predictions against the benchmark schema:

```bash
python scripts/validate_outputs.py --input-dir outputs/raw_extractions/my_lane
```

Validate a community-formatted submission package:

```bash
python scripts/validate_submission.py \
    --submission-dir community/submissions/your_submission
```

## Frozen Benchmark

The canonical frozen benchmark is stored under:

- [benchmark_v2/final_day_freeze_20260709/reports/](benchmark_v2/final_day_freeze_20260709/reports/)
- [final_benchmark_report.md](benchmark_v2/final_day_freeze_20260709/reports/final_benchmark_report.md)
- [final_model_registry.csv](benchmark_v2/final_day_freeze_20260709/reports/final_model_registry.csv)
- [selected_lanes_provenance.md](benchmark_v2/final_day_freeze_20260709/reports/selected_lanes_provenance.md)

Do not rerun, overwrite, or edit anything under `benchmark_v2/final_day_freeze_20260709/`.

## Dataset

The frozen benchmark covers:

| Item | Count |
| --- | ---: |
| Patients | 90 |
| Documents | 125 |
| Images | 150 |

Benchmark-owned ground truth is stored as canonical JSON linked by manifest rows. The public repo includes sanitized example JSON files so the format is visible without exposing the private benchmark annotations.

For a fuller dataset overview, see:

- [DATASET_CARD.md](DATASET_CARD.md)
- [docs/Dataset.md](docs/Dataset.md)
- [docs/create_dataset.md](docs/create_dataset.md)

## Benchmark Tracks

| Track | Input | Output | Primary score |
| --- | --- | --- | --- |
| Raw OCR | Document images | Plain text | Token F1 |
| Direct VLM | Document images | Canonical JSON | Overall extraction score |
| Hybrid OCR+LLM | OCR text | Canonical JSON | Overall extraction score |

Raw OCR scoring focuses on transcription fidelity. Structured scoring focuses on schema validity, scalar accuracy, entity matching, hallucination rate, missing entity rate, and overall extraction quality.

More detail:

- [docs/Benchmark.md](docs/Benchmark.md)
- [docs/Evaluation.md](docs/Evaluation.md)
- [docs/OCRModels.md](docs/OCRModels.md)
- [docs/DirectVLMs.md](docs/DirectVLMs.md)
- [docs/HybridPipelines.md](docs/HybridPipelines.md)

## Canonical JSON

Structured outputs are benchmarked in a canonical JSON format defined by [`CanonicalRawDoc`](src/schemas/raw_extraction.py).

This schema is the contract between model outputs and evaluation. It exists so that OCR, direct VLM, and hybrid pipelines can be compared using one shared representation.

- Benchmark ground truth: evaluator-owned reference annotations in canonical JSON.
- Model prediction: your system's output JSON for the same document, using the same schema.
- Public examples:
  - [docs/examples/example_ground_truth_canonical.json](docs/examples/example_ground_truth_canonical.json)
  - [docs/examples/example_model_prediction_canonical.json](docs/examples/example_model_prediction_canonical.json)

Start here:

- [docs/schema.md](docs/schema.md)
- [docs/annotation_guide.md](docs/annotation_guide.md)
- [docs/GroundTruth.md](docs/GroundTruth.md)

## Benchmark Your Model

### Structured lanes

1. Produce one canonical JSON prediction file per document with filename `<document_id>.json`.
2. Record per-document runtime in `runtime.csv` for every processed document. This is required for direct VLM and hybrid lanes too.
3. Validate those prediction files against the public schema.
4. Build an evaluation manifest that points each document to your prediction JSON while the benchmark manifest points to the reference ground-truth JSON.
5. Run the structured benchmark CLI.

Example extraction command surface:

```bash
python -m src.cli.extract \
    --manifest path/to/extraction_manifest.csv \
    --backend your_backend_name \
    --config configs/backends.yaml \
    --prompts configs/prompts.yaml \
    --output-dir outputs/raw_extractions/your_lane \
    --resume
```

Example structured scoring flow using a community submission package:

```bash
python scripts/validate_submission.py \
    --submission-dir community/submissions/your_submission \
    --write-benchmark-manifest experiments/template_eval_manifest.csv

python -m src.cli.benchmark \
    --manifest experiments/template_eval_manifest.csv \
    --config configs/benchmark_defaults.yaml \
    --output-dir experiments/template_eval_reports
```

In other words: the benchmark already owns the ground truth, while your submission supplies only the prediction JSON and runtime metadata.

### Raw OCR lanes

1. Produce one UTF-8 text file per document with filename `<document_id>.txt`.
2. Record per-document runtime in `runtime.csv`.
3. Validate the submission package.
4. Generate an OCR handoff CSV and run the OCR evaluator.

Raw OCR lanes submit transcription text. Structured lanes submit canonical JSON. Both track types must include runtime metadata.

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

## Benchmark Your Dataset

If you want to benchmark a different hospital or a new document collection, the repository expects:

```text
my_dataset/
├── images/
├── annotations/
└── manifest.csv
```

Your dataset should preserve the same canonical JSON semantics even if document layouts, departments, or institutions differ.

Start here:

- [docs/create_dataset.md](docs/create_dataset.md)
- [docs/schema.md](docs/schema.md)
- [docs/annotation_guide.md](docs/annotation_guide.md)

## Community Submissions

Community submissions should not modify the frozen benchmark. Instead, place them under:

```text
community/submissions/<submission_name>/
├── metadata.yaml
├── predictions/
├── runtime.csv
└── README.md
```

Use the included templates:

- [community/README.md](community/README.md)
- [community/submissions/template/metadata.yaml](community/submissions/template/metadata.yaml)
- [docs/submitting_results.md](docs/submitting_results.md)

The repository now includes `scripts/validate_submission.py` to verify:

- required metadata fields
- per-document prediction presence
- canonical JSON schema validity for structured lanes
- text output presence for raw OCR lanes
- runtime coverage
- benchmark manifest or OCR handoff generation for downstream scoring

## Leaderboards

Frozen publication leaderboard:

- [paper_assets/tables/table_12_final_leaderboard.md](paper_assets/tables/table_12_final_leaderboard.md)

Frozen provenance and coverage:

- [final_benchmark_report.md](benchmark_v2/final_day_freeze_20260709/reports/final_benchmark_report.md)
- [selected_lanes_provenance.md](benchmark_v2/final_day_freeze_20260709/reports/selected_lanes_provenance.md)

Community submissions are standardized, but there is not yet an auto-updating public leaderboard service. The submission format and validator are the current bridge toward that workflow.

## Documentation

Documentation index:

- [docs/README.md](docs/README.md)

High-signal entry points:

- [docs/Dataset.md](docs/Dataset.md)
- [docs/Architecture.md](docs/Architecture.md)
- [docs/Benchmark.md](docs/Benchmark.md)
- [docs/Evaluation.md](docs/Evaluation.md)
- [docs/Statistics.md](docs/Statistics.md)
- [docs/ReproducingResults.md](docs/ReproducingResults.md)
- [docs/FAQ.md](docs/FAQ.md)

## Contributing

Contribution guide:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [GOVERNANCE.md](GOVERNANCE.md)
- [ROADMAP.md](ROADMAP.md)

We welcome:

- bug fixes
- documentation improvements
- new OCR lanes
- new direct VLM lanes
- new hybrid pipelines
- dataset adapters
- evaluation audits
- community submission tooling

## Citation

Please cite the benchmark using [CITATION.cff](CITATION.cff). Author names, DOI, and final paper metadata remain placeholders until camera-ready release.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgements

This repository was prepared for BDA 2026 submission and longer-term benchmark reuse. We thank the annotators, infrastructure maintainers, and model providers whose tools made the benchmark possible.
