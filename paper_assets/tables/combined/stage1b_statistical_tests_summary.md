# Stage 1B Statistical Tests Summary

## Inputs
- Structured: Qwen3-VL 8B (53 documents)
- Structured: LLaVA 13B (53 documents)
- Structured: GLM-OCR + qwen3:8b (53 documents)
- Structured: docTR + qwen3:8b (53 documents)
- Structured: TrOCR + qwen3:8b (53 documents)
- Raw OCR: GLM-OCR (53 documents)
- Raw OCR: docTR (53 documents)
- Raw OCR: TrOCR (53 documents)
- Raw OCR: Docling (43 documents)
- Raw OCR: Surya (25 documents)

## Methods
- Paired bootstrap 95% confidence intervals for system-level means.
- Wilcoxon signed-rank tests for paired continuous per-document metrics.
- Exact-binomial McNemar tests for paired schema-valid success/failure.
- Friedman tests for multiple systems over common documents where SciPy is available.
- Holm-Bonferroni correction for pairwise p-values.

## Friedman Tests
- Structured overall extraction score: common N=53, p=4.6812459447620063e-29
- Raw OCR token F1: common N=25, p=8.853643493507091e-15

## Outputs
- `stage1b_bootstrap_ci.csv`
- `stage1b_pairwise_tests.csv`
- `stage1b_mcnemar_tests.csv`

Bootstrap rows: 15
Pairwise rows: 30
McNemar rows: 10
