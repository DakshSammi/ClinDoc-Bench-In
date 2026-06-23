# Server 1 / Server 2 Code Audit

Server 2 import could not be completed because SSH required password authentication in a non-interactive session. The clean repository currently uses Server 1 code only, with Server 2 import instructions in `docs/server2_import_blocked.md`.

| Server 1 path | Server 2 path | Chosen repo path | Decision | Reason |
|---|---|---|---|---|
| `scripts/benchmark_raw_ocr_outputs.py` | not imported | `scripts/benchmark_raw_ocr_outputs.py` | keep | Current raw OCR evaluator used for Stage 1B OCR/token F1 and text similarity. |
| `scripts/benchmark_stage1b_structured_outputs.py` | not imported | `scripts/benchmark_stage1b_structured_outputs.py` | keep | Current Server 1 canonical structured JSON evaluator. |
| `scripts/benchmark_semantic_outputs.py` | not imported | `scripts/benchmark_semantic_outputs.py` | keep | Current Stage 1C evidence-backed semantic benchmark evaluator. |
| `scripts/run_stage1b_ocr_to_json_from_handoff.py` | not imported | `scripts/run_stage1b_ocr_to_json_from_handoff.py` | keep | Current OCR-to-JSON handoff runner. |
| `scripts/run_stage1b_ocr_to_json_canonical.py` | not imported | `scripts/run_stage1b_ocr_to_json_canonical.py` | keep | Canonical prompt/schema runner for Server 1 OCR-to-JSON lanes. |
| `scripts/stage1b_server2_qwen3_compact_resume.py` | blocked | pending import | pending | Prefer Server 2 latest version after manual import. |
| `scripts/export_server2_ocr_handoff.py` | blocked | pending import | pending | Prefer Server 2 latest version after manual import. |

Server-specific absolute paths should be passed as CLI arguments or kept in `configs/servers/*.yaml` before public release. No raw images, annotations, OCR text, logs, model outputs, or archives were intentionally copied into the clean repository.
