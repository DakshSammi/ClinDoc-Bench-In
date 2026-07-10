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

"""Write Stage 1B Wave 3 paper-ready rollup reports.

This script intentionally uses only the Python standard library so it can run
in the current benchmark environment even when optional data packages are not
ABI-compatible.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BENCH_ROOT = Path("/Computational5/daksh/_gnn_/benchmark_outputs")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(rows: list[dict[str, str]], key: str, only_valid: bool = False) -> float | None:
    values: list[float] = []
    for row in rows:
        if only_valid and row.get("schema_validity") != "1":
            continue
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def rate(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get(key) == "1") / len(rows), 4)


def summarize_structured(rows: list[dict[str, str]]) -> dict[str, object]:
    total = len(rows)
    valid = sum(1 for row in rows if row.get("schema_validity") == "1")
    failed = total - valid
    return {
        "records": total,
        "schema_valid": valid,
        "failed": failed,
        "parse_success_rate": rate(rows, "json_parse_success"),
        "schema_validity_rate": rate(rows, "schema_validity"),
        "avg_output_completeness": avg(rows, "output_completeness", only_valid=True),
        "avg_field_coverage": avg(rows, "field_coverage", only_valid=True),
        "avg_scalar_exact": avg(rows, "scalar_accuracy_exact", only_valid=True),
        "avg_scalar_lenient": avg(rows, "scalar_accuracy_lenient", only_valid=True),
        "avg_entity_exact_f1": avg(rows, "entity_exact_f1", only_valid=True),
        "avg_entity_lenient_f1": avg(rows, "entity_lenient_f1", only_valid=True),
        "avg_hallucination_rate": avg(rows, "hallucination_rate", only_valid=True),
        "avg_missing_entity_rate": avg(rows, "missing_entity_rate", only_valid=True),
        "avg_runtime_seconds": avg(rows, "runtime_seconds"),
    }


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    return str(value)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def best_and_worst(rows: list[dict[str, str]], key: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    valid = [row for row in rows if row.get("schema_validity") == "1" and as_float(row.get(key)) is not None]
    valid.sort(key=lambda row: as_float(row.get(key)) or 0.0)
    return valid[-3:][::-1], valid[:3]


def write_qwen_direct_reports(generated: str) -> dict[str, object]:
    metrics_src = REPORTS / "stage1b_server1_qwen3vl_metrics_final.csv"
    dept_src = REPORTS / "stage1b_server1_qwen3vl_departmentwise_metrics_final.csv"
    metrics_dst = REPORTS / "stage1b_direct_vlm_qwen3vl_metrics.csv"
    dept_dst = REPORTS / "stage1b_direct_vlm_qwen3vl_departmentwise.csv"
    copy_if_exists(metrics_src, metrics_dst)
    copy_if_exists(dept_src, dept_dst)

    rows = read_csv(metrics_src)
    summary = summarize_structured(rows)
    failures = [row for row in rows if row.get("schema_validity") != "1"]
    output_root = BENCH_ROOT / "stage1b_server1_full_20260618_1919"

    write_text(
        REPORTS / "stage1b_direct_vlm_qwen3vl_summary.md",
        f"""# Stage 1B Direct VLM Qwen3-VL Summary

Generated: {generated}

- Output root: `{output_root}`
- Records evaluated: {summary["records"]}
- Schema-valid canonical JSON: {summary["schema_valid"]}
- Remaining failures: {summary["failed"]}
- Parse success rate: {summary["parse_success_rate"]}
- Schema validity rate: {summary["schema_validity_rate"]}
- Average scalar lenient accuracy: {fmt(summary["avg_scalar_lenient"])}
- Average entity lenient F1: {fmt(summary["avg_entity_lenient_f1"])}
- Average hallucination rate: {fmt(summary["avg_hallucination_rate"])}
- Average missing entity rate: {fmt(summary["avg_missing_entity_rate"])}
- Average runtime seconds: {fmt(summary["avg_runtime_seconds"])}

Decision: direct Qwen3-VL is usable for Stage 1B structured benchmarking but has low entity recall on this schema; keep it as a baseline rather than as the only production candidate.
""",
    )

    failure_lines = "\n".join(
        f"- `{row.get('document_id')}`: {row.get('notes') or 'JSON parse/schema failure'}"
        for row in failures
    ) or "- None"
    zero_entity = sum(
        1
        for row in rows
        if row.get("schema_validity") == "1" and as_float(row.get("entity_lenient_f1")) == 0.0
    )
    write_text(
        REPORTS / "stage1b_direct_vlm_qwen3vl_error_taxonomy.md",
        f"""# Stage 1B Direct VLM Qwen3-VL Error Taxonomy

Generated: {generated}

## Hard Failures

{failure_lines}

## Observed Error Classes

- JSON parse/schema failures: {len(failures)} of {len(rows)} records.
- Entity omission dominates valid outputs: {zero_entity} schema-valid records have entity lenient F1 of 0.0.
- Hallucination was not the limiting failure mode under the current matcher; missing entity rate is high.
- Multi-page and alternate-view records remain higher risk and should stay in the manual review queue for qualitative inspection.
""",
    )

    best, worst = best_and_worst(rows, "entity_lenient_f1")
    best_lines = "\n".join(
        f"- `{row['document_id']}`: entity_lenient_f1={row.get('entity_lenient_f1')}, scalar_lenient={row.get('scalar_accuracy_lenient')}, runtime={row.get('runtime_seconds')}s"
        for row in best
    ) or "- None"
    worst_lines = "\n".join(
        f"- `{row['document_id']}`: entity_lenient_f1={row.get('entity_lenient_f1')}, scalar_lenient={row.get('scalar_accuracy_lenient')}, runtime={row.get('runtime_seconds')}s"
        for row in worst
    ) or "- None"
    write_text(
        REPORTS / "stage1b_direct_vlm_qwen3vl_qualitative_examples.md",
        f"""# Stage 1B Direct VLM Qwen3-VL Qualitative Examples

Generated: {generated}

This file lists document IDs and metric signals only; source images and raw outputs remain in the benchmark output root.

## Stronger Valid Examples

{best_lines}

## Weak Valid Examples

{worst_lines}

## Remaining Parse/Schema Failures

{failure_lines}
""",
    )
    return summary


def write_llava_reports(generated: str) -> dict[str, object]:
    rows = read_csv(REPORTS / "stage1b_server1_llava_metrics.csv")
    summary = summarize_structured(rows)
    failures = [row for row in rows if row.get("schema_validity") != "1"]
    output_root = BENCH_ROOT / "stage1b_server1_llava_full_20260619_1129"
    failure_lines = "\n".join(
        f"- `{row.get('document_id')}`: {row.get('notes') or 'JSON parse/schema failure'}"
        for row in failures
    ) or "- None"

    write_text(
        REPORTS / "stage1b_server1_llava_final_status.md",
        f"""# Stage 1B Final release LLaVA Final Status

Generated: {generated}

- Output root: `{output_root}`
- Records evaluated: {summary["records"]}
- Schema-valid canonical JSON: {summary["schema_valid"]}
- Failed outputs: {summary["failed"]}
- Parse success rate: {summary["parse_success_rate"]}
- Schema validity rate: {summary["schema_validity_rate"]}
- Average entity lenient F1: {fmt(summary["avg_entity_lenient_f1"])}
- Average hallucination rate: {fmt(summary["avg_hallucination_rate"])}
- Average missing entity rate: {fmt(summary["avg_missing_entity_rate"])}
- Average runtime seconds: {fmt(summary["avg_runtime_seconds"])}

Decision: LLaVA is retained as a diagnostic baseline only; it is not recommended for Stage 1B paper-leading structured extraction.
""",
    )
    write_text(
        REPORTS / "stage1b_server1_llava_summary.md",
        f"""# Stage 1B Final release LLaVA Summary

Generated: {generated}

- Output directory: `{output_root}`
- Total records: {summary["records"]}
- Successful schema-valid outputs: {summary["schema_valid"]}
- Failed outputs: {summary["failed"]}
- JSON parse success rate: {summary["parse_success_rate"]}
- Schema validity rate: {summary["schema_validity_rate"]}
- Average entity lenient F1: {fmt(summary["avg_entity_lenient_f1"])}
- Average hallucination rate: {fmt(summary["avg_hallucination_rate"])}
- Average missing entity rate: {fmt(summary["avg_missing_entity_rate"])}
- Average runtime seconds: {fmt(summary["avg_runtime_seconds"])}

See `stage1b_server1_llava_metrics.csv`, `stage1b_server1_llava_failure_log.md`, and the output directory for raw responses and logs.
""",
    )
    write_text(
        REPORTS / "stage1b_server1_llava_failure_log.md",
        f"""# Stage 1B Final release LLaVA Failure Log

Generated: {generated}

- Remaining failures: {len(failures)}

{failure_lines}
""",
    )
    return summary


def write_coverage_and_snapshot(
    generated: str,
    qwen: dict[str, object],
    llava: dict[str, object],
    raw_ocr: dict[str, object],
    ocr_json_progress: dict[str, object],
) -> None:
    lanes = ocr_json_progress.get("lanes", {}) if isinstance(ocr_json_progress.get("lanes"), dict) else {}
    session_name = "stage1b_server1_ocrjson_glm_20260619_1149"
    session_running = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0

    def lane_attempted(name: str) -> int:
        lane = lanes.get(name, {}) if isinstance(lanes, dict) else {}
        return int(lane.get("completed", 0) or 0) + int(lane.get("failed", 0) or 0)

    def lane_status(name: str) -> str:
        lane = lanes.get(name, {}) if isinstance(lanes, dict) else {}
        completed = int(lane.get("completed", 0) or 0)
        failed = int(lane.get("failed", 0) or 0)
        total = int(lane.get("total", 53) or 53)
        if not session_running:
            if name == "ocr_glm_ocr_qwen25_14b" and failed and not completed:
                return "stopped_schema_invalid"
            if name == "ocr_glm_ocr_qwen3_8b" and not (completed or failed):
                return "not_started_due_to_qwen25_schema_invalid"
        if completed + failed >= total:
            return "complete"
        if completed or failed:
            return "running"
        return "queued"

    rows = [
        {
            "system": "direct_vlm_qwen3vl",
            "task_type": "structured_json",
            "model_or_engine": "qwen3-vl via Ollama/OpenAI-compatible local endpoint",
            "records_expected": 53,
            "records_attempted": qwen.get("records", 0),
            "records_successful": qwen.get("schema_valid", 0),
            "status": "complete",
            "primary_metric": "schema_validity_rate",
            "primary_value": qwen.get("schema_validity_rate"),
            "secondary_metric": "entity_lenient_f1",
            "secondary_value": qwen.get("avg_entity_lenient_f1"),
            "notes": "paper baseline; low entity recall",
        },
        {
            "system": "direct_vlm_llava_13b",
            "task_type": "structured_json",
            "model_or_engine": "llava:13b",
            "records_expected": 53,
            "records_attempted": llava.get("records", 0),
            "records_successful": llava.get("schema_valid", 0),
            "status": "complete_diagnostic",
            "primary_metric": "schema_validity_rate",
            "primary_value": llava.get("schema_validity_rate"),
            "secondary_metric": "entity_lenient_f1",
            "secondary_value": llava.get("avg_entity_lenient_f1"),
            "notes": "diagnostic baseline only",
        },
        {
            "system": "raw_ocr_glm_ocr",
            "task_type": "raw_ocr",
            "model_or_engine": "glm-ocr:latest",
            "records_expected": 53,
            "records_attempted": raw_ocr.get("records", 0),
            "records_successful": raw_ocr.get("records", 0),
            "status": "complete",
            "primary_metric": "token_f1",
            "primary_value": raw_ocr.get("token_f1"),
            "secondary_metric": "normalized_edit_similarity",
            "secondary_value": raw_ocr.get("normalized_edit_similarity"),
            "notes": "fast, non-empty, but low raw text similarity",
        },
        {
            "system": "ocr_glm_ocr_qwen25_14b",
            "task_type": "ocr_to_json",
            "model_or_engine": "glm-ocr:latest + qwen2.5:14b",
            "records_expected": 53,
            "records_attempted": lane_attempted("ocr_glm_ocr_qwen25_14b"),
            "records_successful": lanes.get("ocr_glm_ocr_qwen25_14b", {}).get("completed", 0) if isinstance(lanes, dict) else 0,
            "status": lane_status("ocr_glm_ocr_qwen25_14b"),
            "primary_metric": "schema_validity_rate",
            "primary_value": "pending",
            "secondary_metric": "entity_lenient_f1",
            "secondary_value": "pending",
            "notes": "stopped after parseable but non-canonical JSON outputs" if not session_running else "live local tmux run; early failures indicate prompt/schema mismatch risk",
        },
        {
            "system": "ocr_glm_ocr_qwen3_8b",
            "task_type": "ocr_to_json",
            "model_or_engine": "glm-ocr:latest + qwen3:8b",
            "records_expected": 53,
            "records_attempted": lane_attempted("ocr_glm_ocr_qwen3_8b"),
            "records_successful": lanes.get("ocr_glm_ocr_qwen3_8b", {}).get("completed", 0) if isinstance(lanes, dict) else 0,
            "status": lane_status("ocr_glm_ocr_qwen3_8b"),
            "primary_metric": "schema_validity_rate",
            "primary_value": "pending",
            "secondary_metric": "entity_lenient_f1",
            "secondary_value": "pending",
            "notes": "not started because qwen2.5 lane was stopped" if not session_running else "queued/running after qwen2.5 lane in single-worker mode",
        },
        {
            "system": "server2_ocr_handoffs",
            "task_type": "raw_ocr_and_ocr_to_json",
            "model_or_engine": "docTR / TrOCR / Surya / Marker / Docling / PaddleOCR / MinerU",
            "records_expected": 53,
            "records_attempted": 0,
            "records_successful": 0,
            "status": "waiting_for_transfer",
            "primary_metric": "not_available",
            "primary_value": "pending",
            "secondary_metric": "not_available",
            "secondary_value": "pending",
            "notes": "Final release handoff CSV and raw OCR files are not accessible from this host",
        },
    ]
    fieldnames = [
        "system",
        "task_type",
        "model_or_engine",
        "records_expected",
        "records_attempted",
        "records_successful",
        "status",
        "primary_metric",
        "primary_value",
        "secondary_metric",
        "secondary_value",
        "notes",
    ]
    write_csv(REPORTS / "stage1b_server1_model_coverage_matrix.csv", rows, fieldnames)

    current = ocr_json_progress.get("current", "")
    pid = ocr_json_progress.get("pid", "")
    output_root = ocr_json_progress.get("output_root", "")
    ocr_json_state = "live" if session_running else "stopped after schema-invalid qwen2.5 outputs"
    write_text(
        REPORTS / "stage1b_server1_paper_results_snapshot.md",
        f"""# Stage 1B Final release Paper Results Snapshot

Generated: {generated}

## Completed Results

- Direct Qwen3-VL structured baseline: {qwen.get("schema_valid")} / {qwen.get("records")} schema-valid, schema validity {qwen.get("schema_validity_rate")}, entity lenient F1 {fmt(qwen.get("avg_entity_lenient_f1"))}.
- LLaVA 13B diagnostic structured baseline: {llava.get("schema_valid")} / {llava.get("records")} schema-valid, schema validity {llava.get("schema_validity_rate")}, entity lenient F1 {fmt(llava.get("avg_entity_lenient_f1"))}.
- GLM-OCR raw OCR: {raw_ocr.get("records")} / {raw_ocr.get("records")} non-empty, token F1 {raw_ocr.get("token_f1")}, normalized edit similarity {raw_ocr.get("normalized_edit_similarity")}, average runtime {raw_ocr.get("runtime_seconds")} seconds/document.

## OCR-to-JSON / Pending

- GLM-OCR to JSON local pipeline state: {ocr_json_state}.
- Session name: `{session_name}`; PID recorded by runner: `{pid}`.
- Current OCR-to-JSON item: `{current}`.
- OCR-to-JSON output root: `{output_root}`.
- Final release OCR handoff files are still not accessible on this host; see `stage1b_server1_waiting_for_server2_handoff_transfer.md`.

## Interpretation

- GLM-OCR is fast and always non-empty, but raw text similarity is low enough that it should be treated as a candidate OCR baseline, not as a high-confidence extraction system by itself.
- Direct Qwen3-VL remains the strongest completed structured local baseline, but missing entity rate is high.
- LLaVA underperforms Qwen3-VL and should stay diagnostic only.
- OCR-to-JSON results should not be used in the paper table; the qwen2.5 lane produced parseable but schema-invalid JSON, and qwen3 did not start before the stop.

## Key Report Paths

- `reports/stage1b_benchmark_script_audit.md`
- `reports/stage1b_raw_ocr_benchmark_glm_ocr/summary_metrics.json`
- `reports/stage1b_server1_glm_ocr_benchmark_summary.md`
- `reports/stage1b_direct_vlm_qwen3vl_summary.md`
- `reports/stage1b_server1_llava_summary.md`
- `reports/stage1b_server1_model_coverage_matrix.csv`
""",
    )

    qwen25 = lanes.get("ocr_glm_ocr_qwen25_14b", {}) if isinstance(lanes, dict) else {}
    qwen3 = lanes.get("ocr_glm_ocr_qwen3_8b", {}) if isinstance(lanes, dict) else {}
    write_text(
        REPORTS / "stage1b_server1_ocr_to_json_glm_ocr_status.md",
        f"""# Stage 1B Final release GLM-OCR OCR-to-JSON Status

Generated: {generated}

- Output root: `{output_root}`
- Runner session: `{session_name}`
- Session running: {session_running}
- qwen2.5 lane: completed={qwen25.get("completed", 0)} failed={qwen25.get("failed", 0)} total={qwen25.get("total", 53)}
- qwen3 lane: completed={qwen3.get("completed", 0)} failed={qwen3.get("failed", 0)} total={qwen3.get("total", 53)}
- Stop reason: qwen2.5 produced parseable JSON but not the canonical `raw_rx_v2` / `document_metadata` + `raw_entities` schema, so the remaining single-worker queue was stopped before qwen3.

Decision: failed OCR-to-JSON attempt for this prompt/schema version. Do not include in paper metrics except as an engineering note.
""",
    )


def write_wave3_progress(
    generated: str,
    qwen: dict[str, object],
    llava: dict[str, object],
    raw_ocr: dict[str, object],
    ocr_json_progress: dict[str, object],
) -> None:
    session_name = "stage1b_server1_ocrjson_glm_20260619_1149"
    session_running = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    payload = {
        "generated": generated,
        "stage": "stage1b_server1_wave3",
        "completed": {
            "benchmark_script_audit": "reports/stage1b_benchmark_script_audit.md",
            "glm_ocr_raw_ocr_evaluation": "reports/stage1b_raw_ocr_benchmark_glm_ocr/summary_metrics.json",
            "qwen3vl_direct_reports": "reports/stage1b_direct_vlm_qwen3vl_summary.md",
            "llava_diagnostic_reports": "reports/stage1b_server1_llava_summary.md",
            "coverage_matrix": "reports/stage1b_server1_model_coverage_matrix.csv",
            "paper_snapshot": "reports/stage1b_server1_paper_results_snapshot.md",
        },
        "live": {
            "ocr_to_json_session": session_name,
            "ocr_to_json_session_running": session_running,
            "ocr_to_json_pid": ocr_json_progress.get("pid"),
            "ocr_to_json_current": ocr_json_progress.get("current"),
            "ocr_to_json_lanes": ocr_json_progress.get("lanes"),
        },
        "blocked_or_waiting": {
            "server2_ocr_handoff": "not accessible from this host",
        },
        "summaries": {
            "qwen3vl": qwen,
            "llava": llava,
            "glm_ocr_raw": raw_ocr,
        },
    }
    write_text(REPORTS / "stage1b_server1_wave3_progress.json", json.dumps(payload, indent=2))
    lanes = ocr_json_progress.get("lanes", {})
    lane_lines = []
    if isinstance(lanes, dict):
        for name, lane in lanes.items():
            lane_lines.append(
                f"- `{name}`: completed={lane.get('completed', 0)} failed={lane.get('failed', 0)} total={lane.get('total', 53)}"
            )
    write_text(
        REPORTS / "stage1b_server1_wave3_progress.md",
        f"""# Stage 1B Final release Wave 3 Progress

Generated: {generated}

## Completed

- Benchmark script audit written.
- GLM-OCR raw OCR benchmark evaluated.
- Direct Qwen3-VL paper-ready reports written.
- LLaVA diagnostic reports frozen.
- Model coverage matrix and paper snapshot written.

## OCR-to-JSON

- OCR-to-JSON session: `{session_name}`
- Session running: {session_running}
- PID: `{ocr_json_progress.get("pid", "")}`
- Current: `{ocr_json_progress.get("current", "")}`

{chr(10).join(lane_lines) if lane_lines else "- No lane status available."}

## Waiting

- Final release OCR handoff transfer is not accessible from this host.
""",
    )


def update_audit(generated: str) -> None:
    audit = REPORTS / "stage1b_benchmark_script_audit.md"
    if not audit.exists():
        return
    text = audit.read_text(encoding="utf-8")
    if "scripts/write_stage1b_wave3_reports.py" in text:
        return
    addition = f"""

## Wave 3 Reporting Helper

- `scripts/write_stage1b_wave3_reports.py`: stdlib-only rollup helper created after the evaluator audit to write paper-facing summaries, coverage matrix, and live progress snapshots without changing evaluator outputs.

Updated: {generated}
"""
    audit.write_text(text.rstrip() + addition, encoding="utf-8")


def main() -> None:
    generated = datetime.now().isoformat(timespec="seconds")
    qwen = write_qwen_direct_reports(generated)
    llava = write_llava_reports(generated)
    raw_ocr = load_json(REPORTS / "stage1b_raw_ocr_benchmark_glm_ocr" / "summary_metrics.json")
    ocr_json_progress = load_json(REPORTS / "stage1b_server1_ocr_to_json_glm_ocr_progress.json")
    write_coverage_and_snapshot(generated, qwen, llava, raw_ocr, ocr_json_progress)
    write_wave3_progress(generated, qwen, llava, raw_ocr, ocr_json_progress)
    update_audit(generated)
    print("Wrote Stage 1B Wave 3 reports")


if __name__ == "__main__":
    main()
