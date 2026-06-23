# Stage 1B Final Ranking Summary

Generated: 2026-06-19T16:56:29

## Rankings

- Best direct VLM among currently imported completed results: **Qwen3-VL 8B** (overall 0.3549). The Internal Qwen3-27B compact recovered run is expected to lead after its final Server 2 package is imported, but no final value is claimed yet.
- Best OCR-to-JSON pipeline: **GLM-OCR + qwen3:8b** (overall 0.3628; 50/53 valid).
- Best raw OCR by OCR/token F1: **GLM-OCR** (0.2464) among full 53-record baselines. Surya scored 0.2395 on a partial 25-record subset.
- Fastest raw OCR: **GLM-OCR**, averaging 2.6471 seconds/document.
- Most reliable structured schema output: **GLM-OCR + qwen3:8b**, 50/53 valid (0.9434).
- Lowest reported hallucination: Qwen3-VL and LLaVA both report 0.0 in frozen direct metrics; among OCR-to-JSON systems, TrOCR + qwen3 reports 0.0026. These values must be read alongside missing entity rate.

## Missing-Entity Risk

- LLaVA: 1.0000.
- TrOCR + qwen3: 0.9992.
- docTR + qwen3: 0.9386.
- Qwen3-VL: 0.9219.
- GLM-OCR + qwen3: 0.8851.

## Caveats

- High schema validity does not imply complete clinical extraction.
- Annotation-gap rate is available for canonical OCR-to-JSON lanes but not the frozen direct VLM compatibility results.
- Docling and Surya OCR results are partial; Marker has only one record.
- qwen2.5 OCR-to-JSON is excluded because it returned the wrong JSON schema shape.
- No paid API result is treated as a completed benchmark result.
