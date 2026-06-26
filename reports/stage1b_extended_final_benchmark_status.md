# Stage 1B Extended Final Benchmark Status

Generated: 2026-06-26T18:48:23

## Final coverage for every system

### Raw OCR
- GLM-OCR: 53/53 full. token F1=0.2464, text_sim=0.1874.
- docTR: 53/53 full. token F1=0.1980, text_sim=0.1851.
- TrOCR: 53/53 full. token F1=0.0082, text_sim=0.0098.
- Docling: 53/53 full. token F1=0.1536, text_sim=0.1529.
- Surya: 53/53 full. token F1=0.2362, text_sim=0.1873.
- EasyOCR: 53/53 full. token F1=0.1606, text_sim=0.1728.
- Qwen2.5-VL raw OCR: 52/53 coverage-limited. token F1=0.4449, text_sim=0.1636.
- Marker: 19/53 partial. token F1=0.1221, text_sim=0.1264.

### Direct structured / VLM
- Internal Qwen3-27B recovered-plus: 53/53 schema-valid. Overall=0.4039.
- Qwen3-VL 8B: 48/53 schema-valid. Overall=0.3549.
- LLaVA 13B: 27/53 schema-valid. Overall=0.2483.
- Qwen2.5-VL structured: 52/53 coverage, 45/53 schema-valid. Overall=0.2731.

### Hybrid pipelines
- GLM-OCR + qwen3:8b: 50/53 schema-valid. Overall=0.3628.
- docTR + qwen3:8b: 49/53 schema-valid. Overall=0.3296.
- TrOCR + qwen3:8b: 48/53 schema-valid. Overall=0.2777.
- EasyOCR + qwen3:8b: 52/53 schema-valid. Overall=0.2851.
- Surya + qwen3:8b: 52/53 schema-valid. Overall=0.2832.
- Docling + qwen3:8b: 52/53 schema-valid. Overall=0.2868.
- Marker + qwen3:8b: 19/53 partial. Overall=0.2336.

## Final Qwen2.5-VL status
- Raw OCR: 52/53 documents recovered via Server 2 recovery. Per-document scores computed via canonical raw OCR evaluator. Mean token_f1=0.4449 over 53 documents (including 8 failed=0). Excluded from primary full-53 paired tests because coverage is 52/53.
- Structured: 52/53 documents recovered; 45/53 schema-valid (8 schema failures). Mean overall=0.2731, entity_lenient_f1=0.0000. Excluded from primary full-53 paired tests.

## Final Marker status
- Marker raw OCR: 19/53 usable rows in Server 1 handoff. 14 additional documents on Server 2 were timeout failures (not handoff artifacts). Marker excluded from primary full-53 tests.
- Marker + qwen3:8b: 19/53 schema-valid partial. Not rerun (handoff-limited).

## Systems included in primary full-53 stats
- Raw OCR: GLM-OCR, docTR, TrOCR, EasyOCR, Surya, Docling (6 systems, all 53/53).
- Structured: Qwen3-VL 8B, LLaVA 13B, GLM-OCR+qwen3, docTR+qwen3, TrOCR+qwen3, EasyOCR+qwen3, Surya+qwen3, Docling+qwen3 (8 systems, all attempted 53 docs).

## Coverage-limited / partial systems
- Qwen2.5-VL raw OCR (52/53 coverage-limited, included in table but not primary stats).
- Qwen2.5-VL structured (52/53 coverage-limited, 45/53 schema-valid, included in table but not primary stats).
- Marker raw OCR (19/53 partial, included in table but not primary stats).
- Marker + qwen3:8b (19/53 partial, included in table but not primary stats).
- Internal Qwen3-27B recovered-plus (53/53 aggregate, included in table but not primary stats; no per-document table on Server 1).

## Exact caveats
- Qwen2.5-VL raw OCR per-document scores computed via Server 2 canonical evaluator; values supersede any earlier Server 2 summary.
- Qwen2.5-VL structured uses compact-to-canonical adapter; entity_lenient_f1=0.0000 reflects no entity extraction.
- Marker handoff limited to 19/53 on Server 1; Server 2 reconciled 39 usable but handoff was not fully re-exported.
- Internal Qwen3-27B recovered-plus scored fields are from the earlier import, not recomputed after p45_2 retry.
- Server 1 raw OCR values are canonical and supersede earlier Server 2 summary values.

## Metric provenance audit summary
- All full-53 raw OCR per-document scores are from local Server 1 benchmark runs using the Stage 1B raw OCR evaluator (scripts/benchmark_raw_ocr_outputs.py).
- Qwen2.5-VL raw OCR per-document scores are from the Server 2 recovery run, evaluated with the same canonical raw OCR evaluator.
- All hybrid structured per-document scores are from local Server 1 runs using the canonical structured benchmark helper (scripts/benchmark_structured_json_outputs.py).
- Qwen2.5-VL structured per-document scores used the compact-to-canonical adapter from the structured benchmark script.
- Marker rows remain partial because the imported handoff provided only 19 usable Marker OCR outputs.
- Server 1 values supersede earlier Server 2 raw OCR summary values where discrepancies exist.
