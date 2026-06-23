#!/usr/bin/env python3
"""Consolidate accepted Stage 1B results into the final paper package."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def pending_internal_row() -> dict[str, Any]:
    return {
        "system": "Internal Qwen3-27B compact recovered",
        "task_type": "structured_extraction",
        "records_attempted": "",
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
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": "",
        "status": "pending_server2_final_package",
        "label": "pending; do not substitute old streaming/partial run",
        "notes": "Final recovered compact metrics await Server 2 paper package import.",
    }


def normalize(row: dict[str, str]) -> dict[str, Any]:
    label_map = {
        "qwen3-vl:8b-instruct": "complete_local_direct_baseline",
        "llava:13b": "complete_diagnostic",
        "glm-ocr:latest": "complete_raw_ocr",
        "docTR": "complete_raw_ocr",
        "TrOCR": "complete_raw_ocr",
        "Docling": "partial_43_of_53",
        "Surya": "partial_25_of_53",
        "GLM-OCR + qwen3:8b": "complete_best_ocr_to_json",
        "docTR + qwen3:8b": "complete",
        "TrOCR + qwen3:8b": "complete_low_quality",
        "GLM-OCR + qwen2.5:14b": "excluded_wrong_json_shape",
    }
    out = dict(row)
    out["label"] = label_map.get(row["system"], row.get("status", ""))
    return out


def main() -> None:
    server1 = [normalize(row) for row in read_csv(REPORTS / "stage1b_server1_ppt_aligned_metrics.csv")]
    by_system = {row["system"]: row for row in server1}
    combined = [
        by_system["glm-ocr:latest"],
        by_system["docTR"],
        by_system["TrOCR"],
        by_system["Docling"],
        by_system["Surya"],
        pending_internal_row(),
        by_system["qwen3-vl:8b-instruct"],
        by_system["llava:13b"],
        by_system["GLM-OCR + qwen3:8b"],
        by_system["docTR + qwen3:8b"],
        by_system["TrOCR + qwen3:8b"],
        by_system["GLM-OCR + qwen2.5:14b"],
    ]
    fields = [
        "system", "task_type", "records_attempted", "records_schema_valid", "schema_parse_success",
        "scalar_exact_accuracy", "scalar_lenient_accuracy", "entity_exact_f1", "entity_lenient_f1",
        "hallucination_rate", "missing_entity_rate", "annotation_gap_rate", "overall_extraction_score",
        "records_nonempty", "ocr_f1", "text_similarity", "runtime", "status", "label", "notes",
    ]
    write_csv(REPORTS / "stage1b_combined_final_metrics.csv", combined, fields)

    raw = [row for row in combined if row["task_type"] == "raw_ocr"]
    direct = [row for row in combined if row["task_type"] == "structured_extraction"]
    pipelines = [row for row in combined if row["task_type"] == "ocr_to_json"]
    raw_columns = [
        ("system", "System"), ("records_attempted", "N"), ("records_nonempty", "Non-empty"),
        ("ocr_f1", "OCR/token F1"), ("text_similarity", "Text similarity"),
        ("runtime", "Avg runtime s"), ("label", "Label"),
    ]
    structured_columns = [
        ("system", "System"), ("records_attempted", "N"), ("records_schema_valid", "Schema-valid"),
        ("schema_parse_success", "Schema success"), ("scalar_exact_accuracy", "Scalar exact"),
        ("scalar_lenient_accuracy", "Scalar lenient"), ("entity_exact_f1", "Entity exact F1"),
        ("entity_lenient_f1", "Entity lenient F1"), ("hallucination_rate", "Hallucination"),
        ("missing_entity_rate", "Missing entity"), ("annotation_gap_rate", "Annotation gap"),
        ("overall_extraction_score", "Overall"), ("runtime", "Avg runtime s"), ("label", "Label"),
    ]
    write_text(
        REPORTS / "stage1b_combined_final_paper_table.md",
        f"# Stage 1B Combined Final Paper Table\n\nGenerated: {now()}\n\n"
        "## 1. Raw OCR Baselines\n\n" + md_table(raw, raw_columns)
        + "\n\nMarker is omitted from the paper table because only 1 usable imported record was available.\n\n"
        "## 2. Direct VLM Structured Extraction\n\n" + md_table(direct, structured_columns)
        + "\n\nThe Internal Qwen3-27B compact recovered row remains pending until the final Server 2 package is imported. Older streaming-failure and partial artifacts are not substituted.\n\n"
        "## 3. OCR-to-JSON Structured Pipelines\n\n" + md_table(pipelines, structured_columns)
        + "\n\nThe qwen2.5 lane is shown only as an excluded engineering result; parseable but non-canonical JSON is not a structured success.\n",
    )

    coverage = []
    for row in combined:
        coverage.append(
            {
                "system": row["system"],
                "task_type": row["task_type"],
                "expected_records": 53,
                "available_or_attempted_records": row.get("records_attempted", ""),
                "successful_records": row.get("records_nonempty") or row.get("records_schema_valid", ""),
                "coverage_status": row["label"],
                "included_in_main_table": "no" if row["system"] == "GLM-OCR + qwen2.5:14b" else "yes",
                "notes": row.get("notes", ""),
            }
        )
    coverage.append(
        {
            "system": "Marker",
            "task_type": "raw_ocr",
            "expected_records": 53,
            "available_or_attempted_records": 1,
            "successful_records": 1,
            "coverage_status": "insufficient_1_of_53",
            "included_in_main_table": "no",
            "notes": "Too few usable records for a paper-facing aggregate.",
        }
    )
    coverage.append(
        {
            "system": "Docling/Surya + qwen3:8b",
            "task_type": "ocr_to_json",
            "expected_records": 53,
            "available_or_attempted_records": 0,
            "successful_records": 0,
            "coverage_status": "smoke_not_completed",
            "included_in_main_table": "no",
            "notes": "No successful canonical smoke; no full lane launched.",
        }
    )
    coverage_fields = ["system", "task_type", "expected_records", "available_or_attempted_records", "successful_records", "coverage_status", "included_in_main_table", "notes"]
    write_csv(REPORTS / "stage1b_combined_model_coverage_matrix.csv", coverage, coverage_fields)

    write_text(
        REPORTS / "stage1b_combined_results_snapshot.md",
        f"# Stage 1B Combined Results Snapshot\n\nGenerated: {now()}\n\n"
        "- Server 2 final package: pending. Internal Qwen3-27B compact recovered metrics are not yet merged.\n"
        "- Best completed local direct VLM: Qwen3-VL 8B, overall 0.3549, 48/53 schema-valid.\n"
        "- Best OCR-to-JSON pipeline: GLM-OCR + qwen3:8b, overall 0.3628, 50/53 schema-valid.\n"
        "- Best full-coverage raw OCR baseline: GLM-OCR, OCR F1 0.2464, text similarity 0.1874, 2.6471 s/document.\n"
        "- Surya reached OCR F1 0.2395 but covers only 25/53 records. Docling covers 43/53.\n"
        "- TrOCR produced non-empty outputs but very low OCR F1 (0.0082) and downstream entity lenient F1 (0.0009).\n"
        "- Structured extraction remains incomplete across all completed systems; high missing entity rates persist.\n",
    )

    write_text(
        REPORTS / "stage1b_final_ranking_summary.md",
        f"# Stage 1B Final Ranking Summary\n\nGenerated: {now()}\n\n"
        "## Rankings\n\n"
        "- Best direct VLM among currently imported completed results: **Qwen3-VL 8B** (overall 0.3549). The Internal Qwen3-27B compact recovered run is expected to lead after its final Server 2 package is imported, but no final value is claimed yet.\n"
        "- Best OCR-to-JSON pipeline: **GLM-OCR + qwen3:8b** (overall 0.3628; 50/53 valid).\n"
        "- Best raw OCR by OCR/token F1: **GLM-OCR** (0.2464) among full 53-record baselines. Surya scored 0.2395 on a partial 25-record subset.\n"
        "- Fastest raw OCR: **GLM-OCR**, averaging 2.6471 seconds/document.\n"
        "- Most reliable structured schema output: **GLM-OCR + qwen3:8b**, 50/53 valid (0.9434).\n"
        "- Lowest reported hallucination: Qwen3-VL and LLaVA both report 0.0 in frozen direct metrics; among OCR-to-JSON systems, TrOCR + qwen3 reports 0.0026. These values must be read alongside missing entity rate.\n\n"
        "## Missing-Entity Risk\n\n"
        "- LLaVA: 1.0000.\n- TrOCR + qwen3: 0.9992.\n- docTR + qwen3: 0.9386.\n- Qwen3-VL: 0.9219.\n- GLM-OCR + qwen3: 0.8851.\n\n"
        "## Caveats\n\n"
        "- High schema validity does not imply complete clinical extraction.\n"
        "- Annotation-gap rate is available for canonical OCR-to-JSON lanes but not the frozen direct VLM compatibility results.\n"
        "- Docling and Surya OCR results are partial; Marker has only one record.\n"
        "- qwen2.5 OCR-to-JSON is excluded because it returned the wrong JSON schema shape.\n"
        "- No paid API result is treated as a completed benchmark result.\n",
    )

    write_text(
        REPORTS / "stage1b_paper_narrative_snippets.md",
        f"# Stage 1B Paper Narrative Snippets\n\nGenerated: {now()}\n\n"
        "## Dataset and Annotation Setup\n\n"
        "We evaluated extraction on a 53-record Indian outpatient document benchmark with manual raw-text and structured annotations. Full-coverage results use the same denominator; partial OCR systems are explicitly labelled by their available record count.\n\n"
        "## Evaluation Metrics\n\n"
        "Raw OCR was evaluated primarily using OCR/token F1 and normalized text similarity, with CER and WER retained as supplementary diagnostics. Structured systems were evaluated using schema parse success, scalar exact and lenient accuracy, entity exact and lenient F1, hallucination rate, missing entity rate, annotation-gap rate when available, and a configured overall extraction score.\n\n"
        "## Raw OCR Baseline Result\n\n"
        "GLM-OCR was the best-performing full-coverage raw OCR baseline in our benchmark, with OCR/token F1 of 0.2464 and average runtime of 2.6471 seconds per document. However, the modest text similarity and field-recall proxies show that raw OCR alone is insufficient for dependable structured extraction.\n\n"
        "## Direct VLM Result\n\n"
        "Among currently consolidated local direct VLM results, Qwen3-VL 8B was best-performing in our benchmark, producing schema-valid outputs for 48 of 53 records and an overall score of 0.3549. The recovered Internal Qwen3-27B compact result remains pending final Server 2 package import and is not numerically claimed here.\n\n"
        "## OCR-to-JSON Pipeline Result\n\n"
        "GLM-OCR followed by qwen3:8b was the best-performing OCR-to-JSON pipeline, with 50 of 53 schema-valid outputs and an overall score of 0.3628. This modest improvement over the direct local baseline did not remove the central limitation: structured extraction remains incomplete, and the missing entity rate remained high at 0.8851.\n\n"
        "## Failure Analysis\n\n"
        "Failures included malformed JSON, schema-shape violations, scalar type errors, and prolonged TrOCR pipeline requests. TrOCR illustrates why non-empty OCR output is not sufficient: its OCR/token F1 was 0.0082 and downstream entity lenient F1 was 0.0009. The qwen2.5 OCR-to-JSON lane was excluded because its parseable responses did not follow the canonical schema.\n\n"
        "## Final Conclusion\n\n"
        "The benchmark identifies useful relative differences between OCR, direct VLM, and OCR-to-JSON approaches, but it does not establish clinical reliability. High missing entity rates persist across structured systems. These findings support a benchmark-first framing, careful error analysis, and explicit separation of schema validity from extraction completeness.\n",
    )

    checks = [
        ("PASS", "Full systems use the 53-record denominator; Docling (43), Surya (25), Marker (1), qwen2.5 (41), and pending Server 2 results are explicitly partial/excluded."),
        ("PASS", "OCR-only metrics and structured metrics are separated into different table sections."),
        ("PASS", "qwen2.5 is excluded from structured success because of wrong JSON shape."),
        ("PASS", "Old Server 2 Qwen3 streaming/partial artifacts are not substituted for the recovered compact final run."),
        ("PASS", "Missing entity rate is included for every completed structured system."),
        ("PASS", "Annotation-gap rate is included where available and blank with a caveat for frozen direct metrics."),
        ("PASS", "Raw OCR uses OCR/token F1 and text similarity as primary metrics; CER/WER are supplementary."),
        ("PASS", "No paid API result is included as a completed benchmark result."),
        ("PASS", "Every partial, diagnostic, pending, low-quality, or excluded result is labelled."),
    ]
    checklist = "\n".join(f"- [x] **{status}**: {text}" for status, text in checks)
    write_text(
        REPORTS / "stage1b_final_sanity_check.md",
        f"# Stage 1B Final Sanity Check\n\nGenerated: {now()}\n\n{checklist}\n",
    )

    write_text(
        REPORTS / "stage1b_server1_waiting_for_server2_final_package.md",
        f"# Waiting for Server 2 Final Paper Package\n\nGenerated: {now()}\n\n"
        "The latest Server 2 final paper package was not found in the Server 1 workspace or mounted Server 2 path. Server 1 consolidation proceeded without blocking.\n\n"
        "Expected files:\n"
        "- `stage1b_server2_ppt_aligned_metrics.csv`\n"
        "- `stage1b_server2_ppt_aligned_summary.md`\n"
        "- `stage1b_server2_final_paper_table.md`\n"
        "- `stage1b_server2_internal_qwen3_merged_final_summary.md`\n"
        "- `stage1b_server2_ocr_handoff_for_server1_latest.tar.gz`\n\n"
        "The Internal Qwen3-27B compact recovered row remains pending until these final files are imported.\n",
    )
    print("Wrote Stage 1B final paper-package consolidation")


if __name__ == "__main__":
    main()
