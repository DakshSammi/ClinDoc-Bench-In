# Stage 1B Extended Metric Provenance Audit — Server 1

Generated: 2026-06-26T18:48:23

## Raw OCR rows

| System | Source OCR output file | Per-document metric CSV | Evaluator script | Rows | Denominator | Tokenization | Intended evaluator |
|---|---|---|---:|---:|:---|---|---|
| GLM-OCR | Server 1 GLM-OCR full benchmark | reports/stage1b_raw_ocr_benchmark_glm_ocr/per_document_ocr_scores.csv | scripts/benchmark_raw_ocr_outputs.py | 53 | 53 | word-level tokenization (whitespace+punct) | yes |
| docTR | Server 2 OCR handoff | reports/stage1b_raw_ocr_benchmark_doctr/per_document_ocr_scores.csv | scripts/benchmark_raw_ocr_outputs.py | 53 | 53 | word-level tokenization (whitespace+punct) | yes |
| TrOCR | Server 2 OCR handoff | reports/stage1b_raw_ocr_benchmark_trocr/per_document_ocr_scores.csv | scripts/benchmark_raw_ocr_outputs.py | 53 | 53 | word-level tokenization (whitespace+punct) | yes |
| Docling | Server 2 OCR handoff | reports/stage1b_extended_raw_ocr_benchmark_docling/per_document_ocr_scores.csv | scripts/benchmark_raw_ocr_outputs.py | 53 | 53 | word-level tokenization (whitespace+punct) | yes |
| Surya | Server 2 OCR handoff | reports/stage1b_extended_raw_ocr_benchmark_surya/per_document_ocr_scores.csv | scripts/benchmark_raw_ocr_outputs.py | 53 | 53 | word-level tokenization (whitespace+punct) | yes |
| EasyOCR | Server 2 OCR handoff | reports/stage1b_extended_raw_ocr_benchmark_easyocr/per_document_ocr_scores.csv | scripts/benchmark_raw_ocr_outputs.py | 53 | 53 | word-level tokenization (whitespace+punct) | yes |
| Qwen2.5-VL raw OCR | Server 2 recovery run | reports/stage1b_extended_qwen25vl_raw_ocr_per_document_scores.csv | Server 2 canonical raw OCR evaluator (same metric logic) | 53 | 53 | word-level tokenization (whitespace+punct) | yes |
| Marker | Server 2 OCR handoff (19/53) | reports/stage1b_extended_raw_ocr_benchmark_marker/per_document_ocr_scores.csv | scripts/benchmark_raw_ocr_outputs.py | 19 | 19 | word-level tokenization (whitespace+punct) | yes, partial only |

### Known discrepancy: Server 1 vs Server 2 raw OCR values

Earlier Server 2 raw OCR summary had different (generally higher) token F1 values for docTR, Surya, and EasyOCR compared to the final Server 1 raw OCR table. Investigation:
- Server 1 values were computed locally from the imported Server 2 OCR handoff using the Stage 1B raw OCR evaluator.
- Server 2 summary may have used a different evaluator version or different tokenization/normalization.
- **Final Server 1 values are canonical** and supersede earlier Server 2 summary values.
- The Server 1 per-document CSVs are the authoritative source for raw OCR metrics.

## Structured / direct / hybrid rows

| System | Source prediction/metrics | Evaluator script | Attempted N | Schema-valid | Failures type | Included in primary stats? |
|---|---:|---:|---:|---|---|
| Internal Qwen3-27B recovered-plus | stage1b_extended_qwen3_27b_merged_plus_metrics.csv | Compact-to-canonical adapter | 53 | 53 | model failures (p45_2 retry fixed) | No (aggregate import only) |
| Qwen3-VL 8B | Server 1 full direct VLM benchmark | scripts/benchmark_structured_json_outputs.py | 53 | 48 | model failures (5 schema parse failures) | Yes |
| LLaVA 13B | Server 1 full direct VLM benchmark | scripts/benchmark_structured_json_outputs.py | 53 | 27 | model failures (26 schema parse failures) | Yes |
| Qwen2.5-VL structured | Server 2 recovery run | Compact-to-canonical adapter from benchmark_structured_json_outputs.py | 53 | 45 | model/schema failures (8 failures, 1 unrecovered) | No (coverage-limited 52/53) |
| GLM-OCR + qwen3:8b | Server 1 OCR-to-JSON pipeline | scripts/benchmark_structured_json_outputs.py | 53 | 50 | model failures (3 schema parse failures) | Yes |
| docTR + qwen3:8b | Server 2 handoff + Server 1 qwen3 | scripts/benchmark_structured_json_outputs.py | 53 | 49 | model failures (4 schema parse failures) | Yes |
| TrOCR + qwen3:8b | Server 2 handoff + Server 1 qwen3 | scripts/benchmark_structured_json_outputs.py | 53 | 48 | model failures (5 schema parse failures) | Yes |
| EasyOCR + qwen3:8b | Server 1 background hybrid run | scripts/benchmark_structured_json_outputs.py | 53 | 52 | model failures (1 schema parse failure) | Yes |
| Surya + qwen3:8b | Server 1 background hybrid run | scripts/benchmark_structured_json_outputs.py | 53 | 52 | model failures (1 schema parse failure) | Yes |
| Docling + qwen3:8b | Server 1 background hybrid run | scripts/benchmark_structured_json_outputs.py | 53 | 52 | model failures (1 schema parse failure) | Yes |
| Marker + qwen3:8b | Server 1 background hybrid run (partial) | scripts/benchmark_structured_json_outputs.py | 19 | 19 | partial (34 docs not attempted) | No (partial 19/53) |

## Statistical test audit
- Raw OCR primary paired tests: use token_f1 and text_similarity from 6 full-53 systems.
- Structured primary paired tests: use overall_extraction_score and entity_lenient_f1 from 8 full-53 systems.
- Qwen2.5-VL excluded from primary full-53 tests (52/53 coverage).
- Marker excluded from primary full-53 tests (19/53 partial).
- Internal Qwen3-27B excluded from primary full-53 tests (no per-document scores on Server 1).
- Holm-Bonferroni correction applied within each pairwise test family.

## Final decision
- Server 1 raw OCR values are canonical and supersede earlier Server 2 summary.
- Qwen2.5-VL per-document scores from Server 2 recovery are accepted as computed.
- Marker remains partial at 19/53 (handoff-limited).
- All aggregate tables, statistics, and reports regenerated from canonical Server 1 data.
