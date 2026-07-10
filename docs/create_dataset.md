# Create A Dataset

ClinDoc-Bench-IN can be reused on a new hospital or institution dataset, but the safest route is to keep the same schema and explicit manifest-driven workflow.

## Recommended Layout

```text
my_dataset/
├── images/
├── annotations/
└── manifest.csv
```

## Workflow

1. Collect images and page bundles.
2. Anonymize every image before external sharing.
3. Assign stable `patient_id` and `document_id` values.
4. Create canonical JSON annotations.
5. Validate the annotations.
6. Prepare extraction and evaluation manifests.
7. Run OCR, direct VLM, or hybrid lanes.
8. Score results with the repository evaluators.

## Manifest Shapes In This Repository

The repository currently uses more than one manifest shape, depending on the stage:

### Extraction manifest

Used by `python -m src.cli.extract`.

Important columns:

- `document_id`
- `image_path`
- `patient_id`
- `prescription_id`
- optional dataset descriptors such as `speciality` or `institution_template`

### Structured evaluation manifest

Used by `python -m src.cli.benchmark`.

Required columns:

- `document_id`
- `gt_path`
- `prediction_path`

Common extra columns:

- `image_path`
- `patient_id`
- `prescription_id`
- `page_number`
- `split`

### Raw OCR handoff CSV

Used by `python scripts/benchmark_raw_ocr_outputs.py`.

Required columns:

- `document_id`
- `ocr_text_path`
- `ocr_engine`

Recommended extra columns:

- `runtime`
- `status`

## Public Reference

Use the frozen benchmark manifest as the best reference for document-level metadata:

- [benchmark_v2/data/benchmark_manifest_v2.csv](/Computational5/daksh/_gnn_/Daksh/prescription_pipeline/benchmark_v2/data/benchmark_manifest_v2.csv)

## Validation

If your new dataset produces structured predictions:

```bash
python scripts/validate_outputs.py --input-dir outputs/raw_extractions/your_lane
```

If you package the run as a community submission:

```bash
python scripts/validate_submission.py --submission-dir community/submissions/your_submission
```

## Important Caveat

The repository does not yet provide a single one-command dataset conversion pipeline for arbitrary external datasets. The public path today is:

- prepare your images and canonical JSON
- build a manifest compatible with the extraction or evaluation CLI you want to run
- validate outputs
- score with the benchmark scripts

That keeps the workflow real and reproducible, even though it is not yet a fully automated dataset onboarding system.
