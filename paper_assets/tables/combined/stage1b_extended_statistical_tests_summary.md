# Stage 1B Extended Statistical Tests Summary

Generated: 2026-06-26T15:07:50

## Included full-53 systems
- Structured: Qwen3-VL 8B, LLaVA 13B, GLM-OCR + qwen3:8b, docTR + qwen3:8b, TrOCR + qwen3:8b, EasyOCR + qwen3:8b, Surya + qwen3:8b, Docling + qwen3:8b.
- Raw OCR: GLM-OCR, docTR, TrOCR, EasyOCR, Surya, Docling.

## Excluded from primary paired tests
- Marker raw OCR and Marker + qwen3 partial: partial/interim coverage only.
- Qwen2.5-VL raw OCR and structured rows: imported package provides coverage/runtime status but not per-document benchmark scores needed for paired testing.
- Internal Qwen3-27B recovered-plus: aggregate row imported, but no per-document compatible metric table is available on Server 1 for paired tests.

## Methods
- Paired bootstrap 95% confidence intervals for system-level means.
- Wilcoxon signed-rank tests for paired continuous per-document metrics.
- Exact-binomial McNemar tests for paired schema-valid success/failure.
- Friedman tests over common full-53 document sets where SciPy is available.
- Holm-Bonferroni correction for pairwise p-values.

## Metric families
- Structured: overall_extraction_score, entity_lenient_f1.
- Raw OCR: token_f1, text_similarity.

## Outputs
- stage1b_extended_bootstrap_ci.csv
- stage1b_extended_pairwise_tests.csv
- stage1b_extended_mcnemar_tests.csv
- stage1b_extended_friedman_tests.csv
