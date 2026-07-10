# Canonical JSON Schema

ClinDoc-Bench-IN compares different systems by mapping them into one shared document representation: `CanonicalRawDoc`.

The authoritative implementation lives in:

- [`src/schemas/raw_extraction.py`](../src/schemas/raw_extraction.py)

Two different file roles use this exact same schema:

| File role | Owned by | Purpose |
| --- | --- | --- |
| Ground-truth canonical JSON | Benchmark maintainers | Reference annotation used for scoring |
| Prediction canonical JSON | Model submitter | System output compared against the reference |

Public examples:

- [example_ground_truth_canonical.json](examples/example_ground_truth_canonical.json)
- [example_model_prediction_canonical.json](examples/example_model_prediction_canonical.json)
- [community ground-truth template](../community/ground_truth_template/README.md)

## Why Canonical JSON Exists

Without a shared schema, OCR text, direct VLM outputs, and hybrid pipelines are not directly comparable. Canonical JSON makes the benchmark evaluate:

- what was extracted
- where information is missing
- where information was hallucinated
- whether the output is structurally usable

## Benchmark Flow

```text
Image
  -> OCR text or direct VLM response
  -> Canonical JSON
  -> Scalar and entity evaluation
  -> Aggregate metrics and ranking
```

## Minimal Structured Skeleton

```json
{
  "schema_version": "raw_rx_v2",
  "document_id": "p70",
  "patient_information": {},
  "encounter_information": {},
  "complaints_or_diagnosis": [],
  "observations": [],
  "medications": [],
  "procedures": [],
  "advice": [],
  "follow_up": null,
  "allergy_mentions": [],
  "other_notes": [],
  "lab_observations": [],
  "metadata": {
    "model_name": "example_model",
    "backend_name": "example_backend"
  }
}
```

The benchmark compares prediction JSON against ground-truth JSON with the same top-level structure. Runtime is stored separately in `runtime.csv`; it is required for structured lanes as well as raw OCR lanes.

## Core Field Groups

- `patient_information`: patient name, age, gender, identifier, phone, address
- `encounter_information`: date, department, hospital, doctor, visit context
- `complaints_or_diagnosis`, `observations`, `procedures`, `advice`: evidence-backed entity lists
- `medications`: structured medication lines with optional name, dosage, route, frequency, duration, instruction, and timing
- `follow_up`: review date or interval
- `lab_observations`: test results when present
- `metadata`: model provenance, timing, prompt version, warnings, and schema version

## Required Versus Optional

`document_id` and the top-level schema shape must always be present.

Most clinical fields are optional because prescriptions are sparse, incomplete, and heterogeneous. Absence should be represented explicitly by empty lists, `null`, or empty sub-objects, not by inventing content.

## Validation

Structured outputs can be validated with:

```bash
python scripts/validate_outputs.py --input-dir outputs/raw_extractions/your_lane
```

Community submission packages can be validated with:

```bash
python scripts/validate_submission.py --submission-dir community/submissions/your_submission
```
