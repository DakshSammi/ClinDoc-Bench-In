# Ground Truth Annotation Guide

This guide describes the public annotation conventions used by ClinDoc-Bench-IN.

## Naming And Identity

- `document_id`: unique benchmark-facing identifier, for example `p70` or `p73_4`
- `patient_id`: patient-level identifier shared across related documents
- multi-page or multi-document cases keep stable patient grouping while each page bundle gets its own `document_id`

## File Conventions

- source images live under `prescriptions/`
- ground truth JSON lives under `raw_ground_truths/` or `benchmark/ground_truths/`
- structured predictions should be named `<document_id>.json`
- raw OCR predictions should be named `<document_id>.txt`

## Annotation Principles

- preserve clinically meaningful wording whenever possible
- do not hallucinate missing content to make the JSON look fuller
- keep ambiguous content in raw text fields instead of over-normalizing
- represent absent sections as empty lists or empty optional fields
- keep the benchmark schema stable even when the document layout varies

## Required Top-Level Structure

All structured outputs must follow `CanonicalRawDoc`.

Important field groups:

- `patient_information`
- `encounter_information`
- `complaints_or_diagnosis`
- `observations`
- `medications`
- `procedures`
- `advice`
- `follow_up`
- `allergy_mentions`
- `other_notes`
- `lab_observations`
- `metadata`

## Medications

Medication lines should preserve the raw line and then decompose fields only when visible evidence supports it.

Useful fields:

- `raw_line_text`
- `raw_name`
- `raw_dosage`
- `raw_route`
- `raw_frequency`
- `raw_duration`
- `raw_instruction`
- `raw_timing`

## Scalars Versus Entities

Use scalar-style fields for patient and encounter metadata. Use list-style entity fields for repeated or open-ended evidence such as complaints, observations, advice, and medications.

## Optional And Uncertain Fields

- use empty strings, `null`, or empty lists rather than guessing
- uncertainty belongs in raw evidence, not fabricated normalized content
- preserve page numbers and evidence snippets when available

## Abbreviations

Clinical abbreviations should usually remain in the raw evidence fields. If an expansion is obvious and useful, it may appear in a decomposed field, but the raw form should still be preserved.

## Validation

Before scoring, validate annotations or predictions:

```bash
python scripts/validate_outputs.py --input-dir path/to/json_dir
```

For benchmark submissions, use:

```bash
python scripts/validate_submission.py --submission-dir community/submissions/your_submission
```
