# Stage 1B Combined Final Paper Table

Generated: 2026-06-19T16:56:29

## 1. Raw OCR Baselines

| System | N | Non-empty | OCR/token F1 | Text similarity | Avg runtime s | Label |
|---|---|---|---|---|---|---|
| glm-ocr:latest | 53 | 53 | 0.2464 | 0.1874 | 2.6471 | complete_raw_ocr |
| docTR | 53 | 53 | 0.198 | 0.1851 | 9.1378 | complete_raw_ocr |
| TrOCR | 53 | 53 | 0.0082 | 0.0098 | 6.6103 | complete_raw_ocr |
| Docling | 43 | 43 | 0.148 | 0.148 | 47.758 | partial_43_of_53 |
| Surya | 25 | 25 | 0.2395 | 0.2034 | 86.752 | partial_25_of_53 |

Marker is omitted from the paper table because only 1 usable imported record was available.

## 2. Direct VLM Structured Extraction

| System | N | Schema-valid | Schema success | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing entity | Annotation gap | Overall | Avg runtime s | Label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Internal Qwen3-27B compact recovered |  |  |  |  |  |  |  |  |  |  |  |  | pending; do not substitute old streaming/partial run |
| qwen3-vl:8b-instruct | 53 | 48 | 0.9057 | 0.6191 | 0.6935 | 0.0117 | 0.053 | 0.0 | 0.9219 |  | 0.3549 | 19.7501 | complete_local_direct_baseline |
| llava:13b | 53 | 27 | 0.5094 | 0.4815 | 0.4868 | 0.0 | 0.0 | 0.0 | 1.0 |  | 0.2483 | 7.1591 | complete_diagnostic |

The Internal Qwen3-27B compact recovered row remains pending until the final Server 2 package is imported. Older streaming-failure and partial artifacts are not substituted.

## 3. OCR-to-JSON Structured Pipelines

| System | N | Schema-valid | Schema success | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing entity | Annotation gap | Overall | Avg runtime s | Label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLM-OCR + qwen3:8b | 53 | 50 | 0.9434 | 0.58 | 0.6586 | 0.0244 | 0.0809 | 0.0337 | 0.8851 | 0.7772 | 0.3628 | 23.3147 | complete_best_ocr_to_json |
| docTR + qwen3:8b | 53 | 49 | 0.9245 | 0.5146 | 0.5831 | 0.014 | 0.0473 | 0.0287 | 0.9386 | 0.8042 | 0.3296 | 18.9745 | complete |
| TrOCR + qwen3:8b | 53 | 48 | 0.9057 | 0.433 | 0.4345 | 0.0009 | 0.0009 | 0.0026 | 0.9992 | 0.4314 | 0.2777 | 47.3863 | complete_low_quality |
| GLM-OCR + qwen2.5:14b | 41 | 0 | 0.0 |  |  |  |  |  |  |  |  | 7.7354 | excluded_wrong_json_shape |

The qwen2.5 lane is shown only as an excluded engineering result; parseable but non-canonical JSON is not a structured success.
