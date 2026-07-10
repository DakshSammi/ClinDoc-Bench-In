# Ground Truth

Ground truth annotations are canonical JSON records describing patient, encounter, clinical findings, medications, advice, follow-up, and other prescription fields.

## Annotation Goals

- Preserve clinically meaningful information.
- Normalize values only when normalization is justified by the source document.
- Keep uncertain or absent fields explicit.
- Support structured comparison across OCR, VLM, and hybrid model outputs.

## Validation

Ground truth should be validated before benchmark execution. Invalid ground truth should not be silently accepted unless a specific audit override is used for debugging.

The environment flag `BENCHMARK_V2_ALLOW_INVALID_GT=1` exists only as an audit escape hatch and should not be used for final scoring.
