# Stage 1B Final Paper Table: Server 1 Side

Generated: 2026-06-19T16:03:29

## Raw OCR Baselines

| System | N | Non-empty | OCR F1 | Text similarity | Runtime s |
|---|---|---|---|---|---|
| glm-ocr:latest | 53 | 53 | 0.2464 | 0.1874 | 2.6471 |
| docTR | 53 | 53 | 0.198 | 0.1851 | 9.1378 |
| TrOCR | 53 | 53 | 0.0082 | 0.0098 | 6.6103 |
| Docling | 43 | 43 | 0.148 | 0.148 | 47.758 |
| Surya | 25 | 25 | 0.2395 | 0.2034 | 86.752 |

## Direct VLM Structured Extraction

| System | N | Valid | Schema | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing | Annotation gap | Overall | Runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl:8b-instruct | 53 | 48 | 0.9057 | 0.6191 | 0.6935 | 0.0117 | 0.053 | 0.0 | 0.9219 |  | 0.3549 | 19.7501 |
| llava:13b | 53 | 27 | 0.5094 | 0.4815 | 0.4868 | 0.0 | 0.0 | 0.0 | 1.0 |  | 0.2483 | 7.1591 |

## OCR-to-JSON Structured Pipelines

| System | N | Valid | Schema | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing | Annotation gap | Overall | Runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLM-OCR + qwen3:8b | 53 | 50 | 0.9434 | 0.58 | 0.6586 | 0.0244 | 0.0809 | 0.0337 | 0.8851 | 0.7772 | 0.3628 | 23.3147 |
| docTR + qwen3:8b | 53 | 49 | 0.9245 | 0.5146 | 0.5831 | 0.014 | 0.0473 | 0.0287 | 0.9386 | 0.8042 | 0.3296 | 18.9745 |
| TrOCR + qwen3:8b | 53 | 48 | 0.9057 | 0.433 | 0.4345 | 0.0009 | 0.0009 | 0.0026 | 0.9992 | 0.4314 | 0.2777 | 47.3863 |

Excluded engineering result: `GLM-OCR + qwen2.5:14b` produced parseable but non-canonical JSON (`schema_invalid_wrong_json_shape`) and is not counted as a structured success.
