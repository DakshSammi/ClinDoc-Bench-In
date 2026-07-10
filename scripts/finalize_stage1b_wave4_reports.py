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

"""Create final Stage 1B Wave 4 PPT-aligned and paper-facing reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(rows: list[dict[str, str]], key: str) -> float | str:
    values = [number for row in rows if (number := as_float(row.get(key))) is not None]
    return round(sum(values) / len(values), 4) if values else ""


def overall(schema: Any, scalar_lenient: Any, entity_lenient: Any, entity_exact: Any, hallucination: Any) -> float | str:
    values = [as_float(value) for value in [schema, scalar_lenient, entity_lenient, entity_exact, hallucination]]
    if any(value is None for value in values):
        return ""
    schema_v, scalar_v, lenient_v, exact_v, hallucination_v = values
    return round(0.10 * schema_v + 0.20 * scalar_v + 0.45 * lenient_v + 0.15 * exact_v + 0.10 * (1 - hallucination_v), 4)


def direct_row(system: str, metrics_path: Path, notes: str) -> dict[str, Any]:
    metrics = read_csv(metrics_path)
    valid = [row for row in metrics if row.get("schema_validity") == "1"]
    schema = round(len(valid) / max(1, len(metrics)), 4)
    scalar_exact = avg(valid, "scalar_accuracy_exact")
    scalar_lenient = avg(valid, "scalar_accuracy_lenient")
    entity_exact = avg(valid, "entity_exact_f1")
    entity_lenient = avg(valid, "entity_lenient_f1")
    hallucination = avg(valid, "hallucination_rate")
    return {
        "system": system,
        "task_type": "structured_extraction",
        "records_attempted": len(metrics),
        "records_schema_valid": len(valid),
        "schema_parse_success": schema,
        "scalar_exact_accuracy": scalar_exact,
        "scalar_lenient_accuracy": scalar_lenient,
        "entity_exact_f1": entity_exact,
        "entity_lenient_f1": entity_lenient,
        "hallucination_rate": hallucination,
        "missing_entity_rate": avg(valid, "missing_entity_rate"),
        "annotation_gap_rate": "",
        "overall_extraction_score": overall(schema, scalar_lenient, entity_lenient, entity_exact, hallucination),
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": avg(metrics, "runtime_seconds"),
        "status": "complete",
        "notes": notes + "; annotation-gap unavailable in frozen compatibility metrics",
    }


def raw_ocr_row(system: str, slug: str, notes: str = "") -> dict[str, Any]:
    summary = load_json(REPORTS / f"stage1b_raw_ocr_benchmark_{slug}" / "summary_metrics.json")
    records = summary.get("records", "")
    nonempty_rate = as_float(summary.get("non_empty_output_rate"))
    nonempty = round(records * nonempty_rate) if isinstance(records, int) and nonempty_rate is not None else ""
    return {
        "system": system,
        "task_type": "raw_ocr",
        "records_attempted": records,
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
        "records_nonempty": nonempty,
        "ocr_f1": summary.get("token_f1", ""),
        "text_similarity": summary.get("normalized_edit_similarity", ""),
        "runtime": summary.get("runtime_seconds", ""),
        "status": "complete" if summary else "unavailable",
        "notes": notes or "raw OCR only; CER/WER supplementary",
    }


def pipeline_row(system: str, slug: str) -> dict[str, Any]:
    summary = load_json(REPORTS / f"stage1b_server1_{slug}_structured_summary.json")
    return {
        "system": system,
        "task_type": "ocr_to_json",
        "records_attempted": summary.get("records_attempted", ""),
        "records_schema_valid": summary.get("records_schema_valid", ""),
        "schema_parse_success": summary.get("schema_parse_success", ""),
        "scalar_exact_accuracy": summary.get("scalar_exact_accuracy", ""),
        "scalar_lenient_accuracy": summary.get("scalar_lenient_accuracy", ""),
        "entity_exact_f1": summary.get("entity_exact_f1", ""),
        "entity_lenient_f1": summary.get("entity_lenient_f1", ""),
        "hallucination_rate": summary.get("hallucination_rate", ""),
        "missing_entity_rate": summary.get("missing_entity_rate", ""),
        "annotation_gap_rate": summary.get("annotation_gap_rate", ""),
        "overall_extraction_score": summary.get("overall_extraction_score", ""),
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": summary.get("runtime_seconds", ""),
        "status": "complete" if summary else "unavailable",
        "notes": "canonical raw_rx_v2 structured pipeline",
    }


def failed_qwen25_row() -> dict[str, Any]:
    return {
        "system": "GLM-OCR + qwen2.5:14b",
        "task_type": "ocr_to_json",
        "records_attempted": 41,
        "records_schema_valid": 0,
        "schema_parse_success": 0.0,
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
        "runtime": 7.7354,
        "status": "excluded",
        "notes": "schema_invalid_wrong_json_shape; excluded from structured success counts",
    }


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def main() -> None:
    rows = [
        direct_row("qwen3-vl:8b-instruct", REPORTS / "stage1b_server1_qwen3vl_metrics_final.csv", "strongest completed direct baseline; 48/53 valid"),
        direct_row("llava:13b", REPORTS / "stage1b_server1_llava_metrics.csv", "diagnostic direct baseline"),
        raw_ocr_row("glm-ocr:latest", "glm_ocr"),
        raw_ocr_row("docTR", "doctr"),
        raw_ocr_row("TrOCR", "trocr"),
        raw_ocr_row("Docling", "docling", "raw OCR only; 43 imported records"),
        raw_ocr_row("Surya", "surya", "raw OCR only; 25 imported records"),
        pipeline_row("GLM-OCR + qwen3:8b", "glm_ocr_qwen3"),
        pipeline_row("docTR + qwen3:8b", "doctr_qwen3"),
        pipeline_row("TrOCR + qwen3:8b", "trocr_qwen3"),
        failed_qwen25_row(),
    ]
    fields = [
        "system", "task_type", "records_attempted", "records_schema_valid", "schema_parse_success",
        "scalar_exact_accuracy", "scalar_lenient_accuracy", "entity_exact_f1", "entity_lenient_f1",
        "hallucination_rate", "missing_entity_rate", "annotation_gap_rate", "overall_extraction_score",
        "records_nonempty", "ocr_f1", "text_similarity", "runtime", "status", "notes",
    ]
    write_csv(REPORTS / "stage1b_server1_ppt_aligned_metrics.csv", rows, fields)

    main_columns = [
        ("system", "System"), ("task_type", "Type"), ("records_attempted", "N"),
        ("records_schema_valid", "Valid"), ("schema_parse_success", "Schema"),
        ("scalar_exact_accuracy", "Scalar exact"), ("scalar_lenient_accuracy", "Scalar lenient"),
        ("entity_exact_f1", "Entity exact F1"), ("entity_lenient_f1", "Entity lenient F1"),
        ("hallucination_rate", "Hallucination"), ("missing_entity_rate", "Missing"),
        ("annotation_gap_rate", "Annotation gap"), ("overall_extraction_score", "Overall"),
        ("ocr_f1", "OCR F1"), ("text_similarity", "Text similarity"), ("runtime", "Runtime s"),
        ("status", "Status"),
    ]
    write_text(
        REPORTS / "stage1b_server1_ppt_aligned_summary.md",
        f"# Stage 1B Server 1 PPT-Aligned Summary\n\nGenerated: {now()}\n\n"
        "OCR-only rows use OCR/token F1 and text similarity. Structured rows use schema parse success, scalar exact/lenient, entity exact/lenient F1, hallucination, missing entity, annotation gap when available, and the configured overall extraction score. CER/WER remain supplementary.\n\n"
        + table(rows, main_columns),
    )

    raw_rows = [row for row in rows if row["task_type"] == "raw_ocr"]
    direct_rows = [row for row in rows if row["task_type"] == "structured_extraction"]
    pipeline_rows = [row for row in rows if row["task_type"] == "ocr_to_json" and row["status"] != "excluded"]
    raw_columns = [("system", "System"), ("records_attempted", "N"), ("records_nonempty", "Non-empty"), ("ocr_f1", "OCR F1"), ("text_similarity", "Text similarity"), ("runtime", "Runtime s")]
    structured_columns = [("system", "System"), ("records_attempted", "N"), ("records_schema_valid", "Valid"), ("schema_parse_success", "Schema"), ("scalar_exact_accuracy", "Scalar exact"), ("scalar_lenient_accuracy", "Scalar lenient"), ("entity_exact_f1", "Entity exact F1"), ("entity_lenient_f1", "Entity lenient F1"), ("hallucination_rate", "Hallucination"), ("missing_entity_rate", "Missing"), ("annotation_gap_rate", "Annotation gap"), ("overall_extraction_score", "Overall"), ("runtime", "Runtime s")]
    write_text(
        REPORTS / "stage1b_final_paper_table_server1_side.md",
        f"# Stage 1B Final Paper Table: Server 1 Side\n\nGenerated: {now()}\n\n"
        "## Raw OCR Baselines\n\n" + table(raw_rows, raw_columns)
        + "\n\n## Direct VLM Structured Extraction\n\n" + table(direct_rows, structured_columns)
        + "\n\n## OCR-to-JSON Structured Pipelines\n\n" + table(pipeline_rows, structured_columns)
        + "\n\nExcluded engineering result: `GLM-OCR + qwen2.5:14b` produced parseable but non-canonical JSON (`schema_invalid_wrong_json_shape`) and is not counted as a structured success.\n",
    )

    write_text(
        REPORTS / "stage1b_server1_paper_results_snapshot.md",
        f"# Stage 1B Server 1 Paper Results Snapshot\n\nGenerated: {now()}\n\n"
        "- Strongest direct structured baseline: Qwen3-VL, 48/53 valid, overall 0.3549.\n"
        "- Strongest OCR-to-JSON pipeline: GLM-OCR + qwen3, 50/53 valid, overall 0.3628.\n"
        "- docTR + qwen3: 49/53 valid, overall 0.3296.\n"
        "- TrOCR + qwen3: 48/53 valid, overall 0.2777; three stall-like invalid requests exceeded 600 seconds.\n"
        "- Raw OCR leaders by OCR F1: GLM-OCR 0.2464, Surya 0.2395 (25 records), docTR 0.1980.\n"
        "- qwen2.5 OCR-to-JSON remains excluded for wrong JSON shape.\n",
    )

    write_text(
        REPORTS / "stage1b_server1_doctr_qwen3_progress.md",
        f"# Stage 1B Server 1 docTR + qwen3 Progress\n\nGenerated: {now()}\n\n- Completed: 53/53\n- Schema-valid: 49\n- Failed: 4\n- Status: complete and benchmarked\n",
    )
    write_text(
        REPORTS / "stage1b_server1_trocr_qwen3_progress.md",
        f"# Stage 1B Server 1 TrOCR + qwen3 Progress\n\nGenerated: {now()}\n\n- Completed: 53/53\n- Schema-valid: 48\n- Failed: 5\n- Status: complete and benchmarked\n",
    )
    write_text(
        REPORTS / "stage1b_server1_server2ocr_qwen3_status.md",
        f"# Stage 1B Server 1 Imported Server 2 OCR + qwen3 Status\n\nGenerated: {now()}\n\n- docTR + qwen3: complete, 49/53 schema-valid.\n- TrOCR + qwen3: complete, 48/53 schema-valid.\n- Completed output root: `/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_server2_doctr_trocr_qwen3_canonical_full_20260619_125803`\n",
    )
    write_text(
        REPORTS / "stage1b_server1_docling_surya_qwen3_smoke_status.md",
        f"# Stage 1B Server 1 Docling/Surya qwen3 Smoke Status\n\nGenerated: {now()}\n\n- Smoke launch attempted after docTR/TrOCR completion.\n- Managed environment blocked the local Ollama request before the runner wrote progress; the escalated command was terminated after exceeding its watchdog window.\n- Docling smoke result: not produced.\n- Surya has only 3/5 requested smoke documents in the imported handoff (`p4`, `p20`, `p25_1`); `p38_1` and `p42_1` are absent.\n- No Docling or Surya full OCR-to-JSON lane was launched.\n",
    )
    print("Wrote Wave 4 final reports")


if __name__ == "__main__":
    main()
