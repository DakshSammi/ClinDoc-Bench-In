# Stage 1B Final Sanity Check

Generated: 2026-06-19T16:56:29

- [x] **PASS**: Full systems use the 53-record denominator; Docling (43), Surya (25), Marker (1), qwen2.5 (41), and pending Server 2 results are explicitly partial/excluded.
- [x] **PASS**: OCR-only metrics and structured metrics are separated into different table sections.
- [x] **PASS**: qwen2.5 is excluded from structured success because of wrong JSON shape.
- [x] **PASS**: Old Server 2 Qwen3 streaming/partial artifacts are not substituted for the recovered compact final run.
- [x] **PASS**: Missing entity rate is included for every completed structured system.
- [x] **PASS**: Annotation-gap rate is included where available and blank with a caveat for frozen direct metrics.
- [x] **PASS**: Raw OCR uses OCR/token F1 and text similarity as primary metrics; CER/WER are supplementary.
- [x] **PASS**: No paid API result is included as a completed benchmark result.
- [x] **PASS**: Every partial, diagnostic, pending, low-quality, or excluded result is labelled.
