# Stage 1B Extended Final Benchmark Status

Generated: 2026-06-26T15:07:43

## Server 1 background hybrid run
- PID 18576 is no longer running.
- extended_full_run.log shows EasyOCR Full, Surya Full, Docling Full, and Marker Partial lanes were launched.
- Consolidated completed outputs:
  - EasyOCR + qwen3:8b: 52/53 schema-valid, overall 0.2851.
  - Surya + qwen3:8b: 52/53 schema-valid, overall 0.2832.
  - Docling + qwen3:8b: 52/53 schema-valid, overall 0.2868.
  - Marker + qwen3 partial: 19/53 schema-valid, overall 0.2336.

## Completed systems for paper-facing tables
- Raw OCR full-53: GLM-OCR, docTR, TrOCR, Docling, Surya, EasyOCR.
- Direct structured: Internal Qwen3-27B recovered-plus coverage row, Qwen3-VL 8B, LLaVA 13B, Qwen2.5-VL coverage row.
- Hybrid OCR-to-JSON full-53: GLM-OCR + qwen3:8b, docTR + qwen3:8b, TrOCR + qwen3:8b, EasyOCR + qwen3:8b, Surya + qwen3:8b, Docling + qwen3:8b.

## Partial / interim systems
- Marker raw OCR: 19/53 usable imported rows only.
- Marker + qwen3 partial: 19/53 schema-valid over the imported partial lane.

## Excluded / blocked systems
- Qwen2.5 OCR-to-JSON wrong-schema lane: excluded from structured success.
- Tesseract, PaddleOCR, Firenze, Moondream: blocked or not included as final systems.
- Qwen3 full-capability smoke: 2/5 valid, excluded from final paper tables.
- Paid API systems remain excluded from completed benchmark results.

## Qwen2.5-VL caveat
- Structured coverage: 47/53 successful; 6 no_images cases treated as data-availability gaps.
- Raw OCR coverage: 48/53 successful; 5 no_images cases treated as data-availability gaps.
- Imported Server 2 package provides coverage/runtime status but not the per-document benchmark exports required for primary paired statistics on Server 1.

## Internal Qwen3 recovered-plus caveat
- stage1b_extended_qwen3_27b_merged_plus_metrics.csv upgrades schema-valid coverage to 53/53 after the p45_2 retry.
- The imported recovered-plus package does not include a fresh CanonicalRawDoc recomputation, so overall/scalar/entity fields in the final table are carried forward from the latest imported recovered row (52/53) and labelled accordingly.

## Imported Server 2 verification
- present: stage1b_extended_server2_transfer_manifest.md
- present: stage1b_extended_qwen25vl_structured_metrics.csv
- present: stage1b_extended_qwen25vl_raw_ocr_metrics.csv
- present: stage1b_extended_server2_final_status.md
- present: stage1b_extended_qwen25vl_smoke_summary.md
- present: stage1b_extended_qwen25vl_structured_full_summary.md
- present: stage1b_extended_qwen25vl_raw_full_summary.md
- present: stage1b_extended_marker_completion_diagnosis.md
- present: stage1b_extended_server2_statistical_test_inputs_ready.md
- present: stage1b_extended_server2_structured_benchmark_summary.md
- present: stage1b_extended_qwen3_27b_merged_plus_metrics.csv

## Included in primary full-53 statistical tests
- Structured: Qwen3-VL 8B, LLaVA 13B, GLM-OCR + qwen3:8b, docTR + qwen3:8b, TrOCR + qwen3:8b, EasyOCR + qwen3:8b, Surya + qwen3:8b, Docling + qwen3:8b.
- Raw OCR: GLM-OCR, docTR, TrOCR, EasyOCR, Surya, Docling.

## Excluded from primary full-53 statistical tests
- Internal Qwen3 recovered-plus: aggregate import only, no per-document compatible score table on Server 1.
- Qwen2.5-VL raw and structured rows: coverage/runtime imported, but no per-document score tables available locally.
- Marker raw and Marker + qwen3: partial/interim only.
