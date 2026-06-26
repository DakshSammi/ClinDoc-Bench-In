# Stage 1B Extended Direct VLM Structured Table

Generated: 2026-06-26T15:07:43

| System | N | Schema-valid | Schema parse | Scalar exact | Scalar lenient | Entity exact F1 | Entity lenient F1 | Hallucination | Missing | Annotation gap | Overall | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Internal Qwen3-27B compact recovered-plus | 53/53 | 53 | 1.0000 | 0.5714 | 0.6415 | 0.0639 | 0.1786 | 0.1246 | 0.6869 | 0.4147 | 0.4039 |  | recovered_plus_coverage_with_frozen_scored_fields | Coverage and schema-validity are from stage1b_extended_qwen3_27b_merged_plus_metrics.csv (53/53 after p45_2 retry). Scored extraction fields remain the latest imported recovered values from paper_assets/tables/server2/stage1b_server2_ppt_aligned_metrics.csv because the recovered-plus import did not include a fresh CanonicalRawDoc recomputation. |
| qwen3-vl:8b-instruct | 48/53 | 48 | 0.9057 | 0.6191 | 0.6935 | 0.0117 | 0.0530 | 0.0000 | 0.9219 |  | 0.3549 | 19.7501 | full_53_local_direct_vlm | Accepted Server 1 local direct structured baseline. |
| llava:13b | 27/53 | 27 | 0.5094 | 0.4815 | 0.4868 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  | 0.2483 | 7.1591 | full_53_diagnostic_direct_vlm | Accepted diagnostic direct structured baseline. |
| Qwen2.5-VL structured | 47/53 | 47 | 0.8868 |  |  |  |  |  |  |  |  | 6.6 | coverage_only_imported_status | 6 no_images cases are data-availability gaps, not model failures; the imported Server 2 package included success/runtime status but not a per-document canonical benchmark export for scalar/entity metrics. |

Qwen2.5-VL structured is included with coverage and runtime because the imported package did not include the per-document canonical score export needed to recompute scalar/entity metrics on Server 1. Internal Qwen3-27B recovered-plus uses 53/53 coverage from the latest import and carries forward the latest imported scored fields from the earlier recovered row.
