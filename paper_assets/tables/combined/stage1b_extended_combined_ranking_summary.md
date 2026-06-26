# Stage 1B Extended Combined Ranking Summary

Generated: 2026-06-26T18:48:23

## Headline findings
- Best direct VLM by available overall score: Internal Qwen3-27B compact recovered-plus, overall 0.4039 with 53/53 schema-valid.
- Best local direct Ollama VLM baseline: Qwen3-VL 8B, overall 0.3549 with 48/53 schema-valid.
- Best OCR-to-JSON pipeline: GLM-OCR + qwen3:8b, overall 0.3628 with 50/53 schema-valid.
- Best raw OCR by OCR/token F1: GLM-OCR, 0.2464 on the full 53-record denominator.
- Fastest raw OCR among full-53 lanes: GLM-OCR, 2.6471 seconds per document.
- Most reliable schema-valid structured system: Internal Qwen3-27B recovered-plus, 53/53.

## Important caveats
- Several low-recall systems report 0.0 hallucination because they mostly omit entities rather than invent them.
- Qwen2.5-VL raw OCR achieves a relatively high token F1 (0.4449) but is coverage-limited at 52/53.
- Qwen2.5-VL structured achieves entity_lenient_f1=0.0000 across all documents (no entity extraction).
- Marker is partial (19/53) and excluded from full-53 comparisons.
- Internal Qwen3-27B recovered-plus scored fields were not recomputed after the p45_2 retry.
