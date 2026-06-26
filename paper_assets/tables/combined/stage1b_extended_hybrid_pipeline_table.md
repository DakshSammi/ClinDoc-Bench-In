# Stage 1B Extended Hybrid OCR-to-JSON Table

Generated: 2026-06-26T18:48:23

## Full-53 systems (primary comparison)

| System | N | Schema-valid | Schema parse | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing | Annotation gap | Overall | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLM-OCR + qwen3:8b | 50/53 | 50 | 0.9434 | 0.5800 | 0.6586 | 0.0244 | 0.0809 | 0.0337 | 0.8851 | 0.7772 | 0.3628 | 23.3147 | full_53_hybrid_pipeline | Accepted Server 1 OCR-to-JSON pipeline. |
| docTR + qwen3:8b | 49/53 | 49 | 0.9245 | 0.5146 | 0.5831 | 0.0140 | 0.0473 | 0.0287 | 0.9386 | 0.8042 | 0.3296 | 18.9745 | full_53_hybrid_pipeline | Imported Server 2 OCR handoff evaluated locally. |
| TrOCR + qwen3:8b | 48/53 | 48 | 0.9057 | 0.4330 | 0.4345 | 0.0009 | 0.0009 | 0.0026 | 0.9992 | 0.4314 | 0.2777 | 47.3863 | full_53_hybrid_pipeline_low_quality | Imported Server 2 OCR handoff evaluated locally. |
| EasyOCR + qwen3:8b | 52/53 | 52 | 0.9811 | 0.4327 | 0.4327 | 0.0008 | 0.0008 | 0.0000 | 0.9993 | 0.0000 | 0.2851 | 20.6328 | full_53_hybrid_pipeline | Consolidated from the completed background hybrid run. |
| Surya + qwen3:8b | 52/53 | 52 | 0.9811 | 0.4231 | 0.4231 | 0.0008 | 0.0008 | 0.0000 | 0.9993 | 0.0000 | 0.2832 | 21.1282 | full_53_hybrid_pipeline | Consolidated from the completed background hybrid run. |
| Docling + qwen3:8b | 52/53 | 52 | 0.9811 | 0.4396 | 0.4409 | 0.0008 | 0.0008 | 0.0000 | 0.9993 | 0.0000 | 0.2868 | 12.0035 | full_53_hybrid_pipeline | Consolidated from the completed background hybrid run. |

## Partial / Interim Hybrid Lane

| System | N | Schema-valid | Schema parse | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing | Annotation gap | Overall | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Marker + qwen3:8b partial | 19/53 | 19 | 0.3585 | 0.4887 | 0.4887 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2336 | 9.5529 | partial_interim_hybrid | Only a partial/import-limited lane was available (19/53). Keep separate from full-53 comparisons. |

EasyOCR, Surya, and Docling + qwen3 were consolidated from the completed background hybrid run. Marker + qwen3 remains partial/interim only (19/53).
