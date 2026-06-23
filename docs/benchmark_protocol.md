# Benchmark Protocol

ClinDoc-Bench-IN evaluates Indian outpatient prescription understanding on a private 53-record benchmark.

## Families

1. Raw OCR/document parser baselines: OCR engines produce text or layout text. Paper-facing metrics are OCR/token F1 and text similarity. CER/WER are supplementary diagnostics.
2. Direct VLM raw extraction baselines: local VLMs directly produce canonical structured JSON from images.
3. OCR-to-JSON pipelines: OCR text is handed to a local LLM, which must produce canonical structured JSON.
4. Semantic-enhanced extraction benchmark: valid Stage 1B structured outputs are enriched with evidence-backed semantic entities and relations.

## Structured Metrics

Primary structured metrics are schema parse success, scalar exact accuracy, scalar lenient accuracy, entity exact F1, entity lenient F1, hallucination rate, missing entity rate, annotation-gap rate, and overall extraction score.

## OCR Metrics

Primary OCR metrics are OCR/token F1 and text similarity. OCR-only systems are not labelled schema-valid or schema-invalid.

## Safety

Raw data, raw annotations, raw OCR/model outputs, failed cases, logs, and API credentials remain private and are not committed.
