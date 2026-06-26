# Stage 1B Extended Direct VLM Structured Table

Generated: 2026-06-26T18:48:23

## Full-53 systems (primary comparison)

| System | N | Schema-valid | Schema parse | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing | Annotation gap | Overall | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl:8b-instruct | 48/53 | 48 | 0.9057 | 0.6191 | 0.6935 | 0.0117 | 0.0530 | 0.0000 | 0.9219 |  | 0.3549 | 19.7501 | full_53_direct_vlm | Accepted Server 1 local direct structured baseline. |
| llava:13b | 27/53 | 27 | 0.5094 | 0.4815 | 0.4868 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  | 0.2483 | 7.1591 | full_53_direct_vlm_diagnostic | Accepted diagnostic direct structured baseline. |

## Coverage-limited systems

| System | N | Schema-valid | Schema parse | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing | Annotation gap | Overall | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL structured | 45/53 | 45 | 0.8491 | 0.4421 | 0.4677 |  | 0.0000 | 0.0535 | 1.0000 | 0.6588 | 0.2731 | 7.4 | coverage_limited_52_53_with_metrics | Qwen2.5-VL structured: 52/53 documents recovered (p36_1 structured unrecovered); 45/53 schema-valid (8 schema failures). Per-document scores computed via compact-to-canonical adapter. Excluded from primary full-53 paired tests because coverage is 52/53. |

Qwen2.5-VL structured is included with computed per-document metrics (52/53 coverage, 45/53 schema-valid). It is excluded from primary full-53 paired tests because 1 document (p36_1) was not recovered. Internal Qwen3-27B recovered-plus uses 53/53 coverage from the latest import and carries forward the latest imported scored fields from the earlier recovered row.
