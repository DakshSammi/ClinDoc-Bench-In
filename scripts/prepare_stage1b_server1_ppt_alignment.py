#!/usr/bin/env python3
# Copyright 2026 ClinDoc-Bench-IN contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare PPT-aligned Stage 1B Server 1 reports and imported handoff CSV."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
DATA = PROJECT_ROOT / "data"
IMPORT_ROOT = Path("/Computational5/daksh/_gnn_/benchmark_outputs/server2_handoff_imports/stage1b_server2_ocr_handoff_for_server1_20260619_1150")
QWEN3_ROOT = Path("/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_server1_full_20260618_1919")
LLAVA_ROOT = Path("/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_server1_llava_full_20260619_1129")
BAD_QWEN25_ROOT = Path("/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_server1_ocr_to_json_glm_ocr_20260619_1149")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def avg(values: list[Any]) -> float | str:
    nums = [v for v in (as_float(value) for value in values) if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else ""


def pct(numer: int, denom: int) -> float:
    return round(numer / denom, 4) if denom else 0.0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def import_server2_handoff() -> dict[str, Any]:
    source = IMPORT_ROOT / "stage1b_server2_ocr_outputs_for_server1_wave2.csv"
    missing_report = REPORTS / "stage1b_server1_handoff_missing_transfer_request.md"
    if not source.exists():
        write_text(
            missing_report,
            """# Stage 1B Server 2 Handoff Missing

Please copy this file from Server 2:

`/mnt/mnfas_Varahi/Daksh/exports/stage1b_server2_ocr_handoff_for_server1_20260619_1150.tar.gz`
""",
        )
        return {"available": False, "rows": 0, "available_rows": 0, "missing_rows": 0, "engines": {}}

    manifest = {row["document_id"]: row for row in read_csv(DATA / "full_benchmark_manifest.csv")}
    rows_out: list[dict[str, Any]] = []
    engines: dict[str, int] = {}
    missing = 0
    for row in read_csv(source):
        engine = row["engine"]
        engines[engine] = engines.get(engine, 0) + 1
        raw_abs = IMPORT_ROOT / row.get("raw_text_path", "")
        markdown_abs = IMPORT_ROOT / row.get("markdown_path", "")
        layout_abs = IMPORT_ROOT / row.get("layout_json_path", "")
        pagewise_abs = []
        for item in filter(None, re_split_paths(row.get("pagewise_text_paths", ""))):
            pagewise_abs.append(str(IMPORT_ROOT / item))
        ok = row.get("status") == "ok" and raw_abs.exists()
        missing += int(not ok)
        doc = manifest.get(row["document_id"], {})
        rows_out.append(
            {
                "document_id": row["document_id"],
                "patient_id": doc.get("patient_id", row["document_id"]),
                "ocr_engine": engine,
                "ocr_text_path": str(raw_abs),
                "status": "available" if ok else "missing",
                "runtime": row.get("runtime", ""),
                "env_name": row.get("env_name", ""),
                "markdown_path": str(markdown_abs) if markdown_abs.exists() else "",
                "layout_json_path": str(layout_abs) if layout_abs.exists() else "",
                "pagewise_text_paths": "|".join(pagewise_abs),
                "source_csv": str(source),
                "notes": row.get("notes", ""),
            }
        )
    fields = [
        "document_id",
        "patient_id",
        "ocr_engine",
        "ocr_text_path",
        "status",
        "runtime",
        "env_name",
        "markdown_path",
        "layout_json_path",
        "pagewise_text_paths",
        "source_csv",
        "notes",
    ]
    write_csv(DATA / "stage1b_server2_ocr_handoff_imported.csv", rows_out, fields)
    write_text(
        REPORTS / "stage1b_server1_handoff_import_summary.md",
        f"""# Stage 1B Server 2 Handoff Import Summary

Generated: {now()}

- Archive imported from: `/Computational5/daksh/_gnn_/exports/stage1b_server2_ocr_handoff_for_server1_20260619_1150.tar.gz`
- Extracted root: `{IMPORT_ROOT}`
- Source CSV: `{source}`
- Imported CSV: `data/stage1b_server2_ocr_handoff_imported.csv`
- Rows: {len(rows_out)}
- Rows with resolving raw text files: {sum(1 for row in rows_out if row["status"] == "available")}
- Rows missing raw text files: {missing}
- Engines: {engines}
""",
    )
    return {"available": True, "rows": len(rows_out), "available_rows": len(rows_out) - missing, "missing_rows": missing, "engines": engines}


def re_split_paths(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.replace(";", "|").split("|") if part.strip()]


def summarize_structured_csv(path: Path, model_name: str, notes: str) -> dict[str, Any]:
    rows = read_csv(path)
    attempted = len(rows)
    valid_rows = [row for row in rows if row.get("schema_validity") == "1"]
    schema_rate = pct(len(valid_rows), attempted)
    scalar_exact = avg([row.get("scalar_accuracy_exact") for row in valid_rows])
    scalar_lenient = avg([row.get("scalar_accuracy_lenient") for row in valid_rows])
    entity_exact = avg([row.get("entity_exact_f1") for row in valid_rows])
    entity_lenient = avg([row.get("entity_lenient_f1") for row in valid_rows])
    hallucination = avg([row.get("hallucination_rate") for row in valid_rows])
    missing = avg([row.get("missing_entity_rate") for row in valid_rows])
    annotation_gap = recompute_annotation_gap(model_name)
    overall = overall_score(schema_rate, scalar_lenient, entity_lenient, entity_exact, hallucination)
    return {
        "system": model_name,
        "task_type": "structured_extraction",
        "records_attempted": attempted,
        "records_schema_valid": len(valid_rows),
        "schema_parse_success": schema_rate,
        "scalar_exact_accuracy": scalar_exact,
        "scalar_lenient_accuracy": scalar_lenient,
        "entity_exact_f1": entity_exact,
        "entity_lenient_f1": entity_lenient,
        "hallucination_rate": hallucination,
        "missing_entity_rate": missing,
        "annotation_gap_rate": annotation_gap,
        "overall_extraction_score": overall,
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": avg([row.get("runtime_seconds") for row in rows]),
        "notes": notes,
    }


def recompute_annotation_gap(model_name: str) -> float | str:
    if "qwen3-vl" in model_name:
        output_root, backend = QWEN3_ROOT, "ollama_qwen3_vl_8b"
    elif "llava" in model_name:
        output_root, backend = LLAVA_ROOT, "ollama_llava_13b"
    else:
        return ""
    try:
        sys_path_insert()
        from scripts.run_full_benchmark_stage1 import compute_smoke_metrics

        rows = []
        for doc in read_csv(DATA / "full_benchmark_manifest.csv"):
            parsed = output_root / "raw_structured" / backend / f"{doc['document_id']}.json"
            log = output_root / "logs" / backend / f"{doc['document_id']}.json"
            if parsed.exists():
                metric = compute_smoke_metrics(doc, backend, "raw_structured", parsed, log if log.exists() else None, None)
                if metric.get("schema_validity"):
                    rows.append(metric)
        # compute_smoke_metrics did not historically expose annotation_gap_rate.
        return ""
    except Exception:
        return ""


def sys_path_insert() -> None:
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def overall_score(schema_rate: Any, scalar_lenient: Any, entity_lenient: Any, entity_exact: Any, hallucination: Any) -> float | str:
    values = [as_float(schema_rate), as_float(scalar_lenient), as_float(entity_lenient), as_float(entity_exact), as_float(hallucination)]
    if any(value is None for value in values):
        return ""
    schema_v, scalar_v, entity_l_v, entity_e_v, halluc_v = values
    return round(0.10 * schema_v + 0.20 * scalar_v + 0.45 * entity_l_v + 0.15 * entity_e_v + 0.10 * (1.0 - halluc_v), 4)


def bad_qwen25_row() -> dict[str, Any]:
    log_dir = BAD_QWEN25_ROOT / "logs" / "ocr_glm_ocr_qwen25_14b"
    logs = [load_json(path) for path in sorted(log_dir.glob("*.json"))]
    attempted = len(logs)
    valid = sum(1 for log in logs if log.get("schema_validation_success") is True)
    parse_success = sum(1 for log in logs if log.get("parse_success") is True)
    return {
        "system": "glm-ocr + qwen2.5:14b",
        "task_type": "ocr_to_json",
        "records_attempted": attempted,
        "records_schema_valid": valid,
        "schema_parse_success": pct(valid, attempted),
        "scalar_exact_accuracy": "",
        "scalar_lenient_accuracy": "",
        "entity_exact_f1": "",
        "entity_lenient_f1": "",
        "hallucination_rate": "",
        "missing_entity_rate": "",
        "annotation_gap_rate": "",
        "overall_extraction_score": "",
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": avg([log.get("runtime_seconds") for log in logs]),
        "notes": f"schema_invalid_wrong_json_shape; parseable JSON {parse_success}/{attempted}, canonical valid {valid}/{attempted}",
    }


def ocr_to_json_row_from_results(system: str, results_path: Path, lane: str) -> dict[str, Any]:
    rows = [row for row in read_csv(results_path) if row.get("lane") == lane]
    attempted = len(rows)
    valid = sum(1 for row in rows if row.get("schema_valid") == "1")
    return {
        "system": system,
        "task_type": "ocr_to_json",
        "records_attempted": attempted,
        "records_schema_valid": valid,
        "schema_parse_success": pct(valid, attempted),
        "scalar_exact_accuracy": "",
        "scalar_lenient_accuracy": "",
        "entity_exact_f1": "",
        "entity_lenient_f1": "",
        "hallucination_rate": "",
        "missing_entity_rate": "",
        "annotation_gap_rate": "",
        "overall_extraction_score": "",
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": avg([row.get("runtime_seconds") for row in rows]),
        "notes": "canonical smoke/full results; detailed metrics pending full structured evaluator" if valid else "schema_invalid_wrong_json_shape or json_parse_failed",
    }


def raw_ocr_row() -> dict[str, Any]:
    summary = load_json(REPORTS / "stage1b_raw_ocr_benchmark_glm_ocr" / "summary_metrics.json")
    return {
        "system": "glm-ocr:latest",
        "task_type": "raw_ocr",
        "records_attempted": summary.get("records", ""),
        "records_schema_valid": "",
        "schema_parse_success": "",
        "scalar_exact_accuracy": "",
        "scalar_lenient_accuracy": "",
        "entity_exact_f1": "",
        "entity_lenient_f1": "",
        "hallucination_rate": "",
        "missing_entity_rate": "",
        "annotation_gap_rate": "",
        "overall_extraction_score": "",
        "records_nonempty": summary.get("records", ""),
        "ocr_f1": summary.get("token_f1", ""),
        "text_similarity": summary.get("normalized_edit_similarity", ""),
        "runtime": summary.get("runtime_seconds", ""),
        "notes": "raw OCR only; CER/WER omitted from primary PPT table",
    }


def write_ppt_metrics(import_status: dict[str, Any]) -> None:
    rows = [
        summarize_structured_csv(
            REPORTS / "stage1b_server1_qwen3vl_metrics_final.csv",
            "qwen3-vl:8b-instruct",
            "preserved final result; no rerun",
        ),
        summarize_structured_csv(
            REPORTS / "stage1b_server1_llava_metrics.csv",
            "llava:13b",
            "diagnostic baseline",
        ),
        raw_ocr_row(),
        bad_qwen25_row(),
    ]
    result_specs = [
        ("glm-ocr + qwen3:8b", REPORTS / "stage1b_server1_glm_ocr_qwen3_canonical_smoke_results.csv", "ocr_glm_ocr_qwen3_8b"),
        ("glm-ocr + qwen2.5:14b canonical smoke", REPORTS / "stage1b_server1_glm_ocr_qwen25_canonical_smoke_results.csv", "ocr_glm_ocr_qwen25_14b_canonical"),
    ]
    for system, path, lane in result_specs:
        if path.exists():
            rows.append(ocr_to_json_row_from_results(system, path, lane))
    fields = [
        "system",
        "task_type",
        "records_attempted",
        "records_schema_valid",
        "schema_parse_success",
        "scalar_exact_accuracy",
        "scalar_lenient_accuracy",
        "entity_exact_f1",
        "entity_lenient_f1",
        "hallucination_rate",
        "missing_entity_rate",
        "annotation_gap_rate",
        "overall_extraction_score",
        "records_nonempty",
        "ocr_f1",
        "text_similarity",
        "runtime",
        "notes",
    ]
    write_csv(REPORTS / "stage1b_server1_ppt_aligned_metrics.csv", rows, fields)
    table_lines = ["| system | type | attempted | schema_valid | schema_parse_success | scalar_lenient | entity_lenient_f1 | hallucination | missing | annotation_gap | overall | ocr_f1 | text_similarity | notes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        table_lines.append(
            f"| {row['system']} | {row['task_type']} | {row['records_attempted']} | {row['records_schema_valid']} | {row['schema_parse_success']} | {row['scalar_lenient_accuracy']} | {row['entity_lenient_f1']} | {row['hallucination_rate']} | {row['missing_entity_rate']} | {row['annotation_gap_rate']} | {row['overall_extraction_score']} | {row['ocr_f1']} | {row['text_similarity']} | {row['notes']} |"
        )
    write_text(
        REPORTS / "stage1b_server1_ppt_aligned_summary.md",
        f"""# Stage 1B Server 1 PPT-Aligned Summary

Generated: {now()}

Primary structured metrics are schema parse success, overall extraction score, scalar exact/lenient accuracy, entity exact/lenient F1, hallucination rate, missing entity rate, and annotation-gap rate. Primary OCR-only metrics are OCR/token F1 and text similarity. CER/WER are intentionally excluded from this primary table.

Overall extraction score uses `configs/benchmark_defaults.yaml` weights:
`0.10*schema_parse + 0.20*scalar_lenient + 0.45*entity_lenient_f1 + 0.15*entity_exact_f1 + 0.10*(1-hallucination_rate)`.

Annotation-gap rate is left blank where the current Stage 1B compatibility metrics did not persist that field.

Server 2 handoff import: available={import_status.get('available')}, rows={import_status.get('rows')}, available_rows={import_status.get('available_rows')}, engines={import_status.get('engines')}.

{chr(10).join(table_lines)}
""",
    )


def write_qwen25_diagnosis() -> None:
    examples = []
    for doc_id in ["p1", "p4", "p20", "p25_1", "p38_1"]:
        path = BAD_QWEN25_ROOT / "repaired" / "ocr_glm_ocr_qwen25_14b" / f"{doc_id}.json"
        if not path.exists():
            continue
        obj = load_json(path)
        examples.append(
            f"- `{doc_id}` keys: {sorted(obj.keys())}; missing `schema_version`, `patient_information`, `encounter_information`, and canonical entity list fields."
        )
    write_text(
        REPORTS / "stage1b_server1_qwen25_ocr_to_json_failure_shape_diagnosis.md",
        f"""# Stage 1B qwen2.5 OCR-to-JSON Failure Shape Diagnosis

Generated: {now()}

The stopped `glm-ocr + qwen2.5:14b` lane produced parseable JSON but not canonical structured extraction JSON.

Observed failure type: `schema_invalid_wrong_json_shape`.

Problems found:
- Wrong field names such as `patient_root`, `hospital`, `raw_text`, `structured_data`, direct `medications`, and document-specific nested objects.
- Missing top-level `schema_version: raw_rx_v2`.
- Missing current `CanonicalRawDoc` objects `patient_information` and `encounter_information`.
- Missing canonical list fields such as `complaints_or_diagnosis`, `observations`, `procedures`, `advice`, `allergy_mentions`, `other_notes`, and `lab_observations` in many outputs.
- The model followed the document content rather than the canonical schema, producing per-document ad hoc wrappers.

Examples:
{chr(10).join(examples)}

Decision: do not count this lane as successful; exclude from final structured table except as a failed OCR-to-JSON attempt.
""",
    )


def main() -> None:
    import_status = import_server2_handoff()
    write_qwen25_diagnosis()
    write_ppt_metrics(import_status)
    print(json.dumps({"import_status": import_status, "metrics": "reports/stage1b_server1_ppt_aligned_metrics.csv"}, indent=2))


if __name__ == "__main__":
    main()
