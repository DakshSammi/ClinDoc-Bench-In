# Stage 1B Extended Combined Ranking Summary

Generated: 2026-06-26T15:07:43

## Headline findings
- Best direct VLM by available overall score: Internal Qwen3-27B compact recovered-plus coverage row, carrying forward the latest imported recovered score of 0.4039 while the recovered-plus import upgrades schema-valid coverage to 53/53.
- Best local direct Ollama VLM baseline: Qwen3-VL 8B, overall 0.3549 with 48/53 schema-valid.
- Best OCR-to-JSON pipeline: GLM-OCR + qwen3:8b, overall 0.3628 with 50/53 schema-valid.
- Best raw OCR by OCR/token F1: GLM-OCR, 0.2464 on the full 53-record denominator.
- Fastest raw OCR among full-53 lanes: GLM-OCR, 2.6471 seconds per document.
- Most reliable schema-valid structured system by available coverage row: Internal Qwen3-27B recovered-plus, 53/53. Among fully local scored lanes, EasyOCR/Surya/Docling + qwen3 each reached 52/53, but with near-zero entity F1 and very high missing-entity rates.

## Important caveats
- Several low-recall systems report 0.0 hallucination because they mostly omit entities rather than invent them.
- The worst missing-entity rates are Marker + qwen3 partial (1.0000), EasyOCR + qwen3 (0.9993), Surya + qwen3 (0.9993), Docling + qwen3 (0.9993), and TrOCR + qwen3 (0.9992).
- Qwen2.5-VL no_images failures are treated as data-availability gaps, not model-performance failures.
- Qwen2.5-VL is shown in paper-facing tables with coverage, but it is excluded from primary full-53 paired tests because the imported Server 2 package lacks per-document benchmark scores on Server 1.
- Internal Qwen3-27B recovered-plus scored fields were not recomputed after the p45_2 retry in the imported package; that row is explicitly labelled as a combined-source consolidation.
