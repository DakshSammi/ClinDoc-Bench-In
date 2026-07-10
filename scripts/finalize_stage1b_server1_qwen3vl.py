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

"""Finalize/deduplicate Stage 1B Final release qwen3-vl outputs.

Performs local JSON repair on failed raw responses where possible, writes unique
status and final metrics/reports. No model calls are made by this script.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_BACKEND = "ollama_qwen3_vl_8b"
FAILED_DOCS = {"p20", "p25_1", "p28", "p36_1", "p42_1"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    candidates = [text.strip()]
    if "```json" in text:
        candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
    if "```" in text:
        candidates.append(text.split("```", 1)[1].split("```", 1)[0].strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    cleaned = []
    for c in candidates:
        cleaned.extend([
            c,
            re.sub(r",\s*([}\]])", r"\1", c),
            c.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'"),
        ])
    for c in cleaned:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def schema_valid(obj: Optional[Dict[str, Any]]) -> bool:
    return bool(obj and (("document_metadata" in obj and "raw_entities" in obj) or obj.get("schema_version") == "raw_rx_v2"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def status_rows(root: Path, manifest: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows = []
    for doc in manifest:
        doc_id = doc["document_id"]
        log_path = root / "logs" / OUTPUT_BACKEND / f"{doc_id}.json"
        parsed_path = root / "raw_structured" / OUTPUT_BACKEND / f"{doc_id}.json"
        repaired_path = root / "raw_structured_repaired" / OUTPUT_BACKEND / f"{doc_id}.json"
        status = "not_run"
        parse = schema = False
        runtime = ""
        if log_path.exists():
            log = json.loads(log_path.read_text(encoding="utf-8"))
            status = log.get("status", "unknown")
            parse = bool(log.get("parse_success"))
            schema = bool(log.get("schema_validation_success"))
            runtime = log.get("runtime_seconds", "")
        if repaired_path.exists():
            status = "repaired"
            parse = schema = True
        rows.append({
            "document_id": doc_id,
            "patient_id": doc.get("patient_id", ""),
            "department_inferred": doc.get("department_inferred", ""),
            "status": status,
            "parse_success": int(parse),
            "schema_validity": int(schema),
            "has_original_parsed_json": int(parsed_path.exists()),
            "has_repaired_json": int(repaired_path.exists()),
            "runtime_seconds": runtime,
            "log_path": rel(log_path) if log_path.exists() else "",
        })
    return rows


def repair_failed(root: Path) -> List[Dict[str, Any]]:
    repairs = []
    for doc_id in sorted(FAILED_DOCS):
        raw_path = root / "raw_responses" / OUTPUT_BACKEND / f"{doc_id}.txt"
        log_path = root / "logs" / OUTPUT_BACKEND / f"{doc_id}.json"
        out_path = root / "raw_structured_repaired" / OUTPUT_BACKEND / f"{doc_id}.json"
        repair_log_path = root / "logs_repaired" / OUTPUT_BACKEND / f"{doc_id}.json"
        if not raw_path.exists():
            repairs.append({"document_id": doc_id, "repair_status": "raw_response_missing", "schema_validity": 0})
            continue
        text = raw_path.read_text(encoding="utf-8", errors="ignore")
        obj = extract_json(text)
        valid = schema_valid(obj)
        if valid:
            write_json(out_path, obj)
        write_json(repair_log_path, {
            "document_id": doc_id,
            "repair_status": "repaired" if valid else "repair_failed",
            "schema_validity": valid,
            "source_raw_response": rel(raw_path),
            "source_log": rel(log_path),
            "repaired_output": rel(out_path) if valid else "",
        })
        repairs.append({"document_id": doc_id, "repair_status": "repaired" if valid else "repair_failed", "schema_validity": int(valid)})
    return repairs


def compute_metrics(root: Path, manifest: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.run_stage1b_server1_full import compute_all_metrics, aggregate_metrics

    metrics = compute_all_metrics(manifest, root)
    # Overlay repaired docs for parse/schema counts and metric calculation when possible.
    from scripts.run_full_benchmark_stage1 import compute_smoke_metrics

    for row in metrics:
        repaired = root / "raw_structured_repaired" / OUTPUT_BACKEND / f"{row['document_id']}.json"
        if repaired.exists():
            log = root / "logs_repaired" / OUTPUT_BACKEND / f"{row['document_id']}.json"
            metric = compute_smoke_metrics(next(d for d in manifest if d["document_id"] == row["document_id"]), OUTPUT_BACKEND, "raw_structured_repaired", repaired, log, None)
            row.update({
                "json_parse_success": 1,
                "schema_validity": 1,
                "status": "repaired",
                "output_completeness": metric.get("output_completeness", 0.0),
                "field_coverage": metric.get("field_coverage", 0.0),
                "scalar_accuracy_exact": metric.get("scalar_accuracy_exact", ""),
                "scalar_accuracy_lenient": metric.get("scalar_accuracy_lenient", ""),
                "entity_exact_f1": metric.get("entity_exact_f1", ""),
                "entity_lenient_f1": metric.get("entity_lenient_f1", ""),
                "hallucination_rate": metric.get("hallucination_rate", ""),
                "missing_entity_rate": metric.get("missing_entity_rate", ""),
                "notes": "local_raw_response_repair",
            })
    fields = [
        "document_id", "patient_id", "department_inferred", "is_multi_page", "is_same_page_multi_view",
        "json_parse_success", "schema_validity", "output_completeness", "field_coverage",
        "scalar_accuracy_exact", "scalar_accuracy_lenient", "entity_exact_f1", "entity_lenient_f1",
        "hallucination_rate", "missing_entity_rate", "runtime_seconds", "status", "notes",
    ]
    write_csv(REPORTS_DIR / "stage1b_server1_qwen3vl_metrics_final.csv", metrics, fields)
    write_csv(
        REPORTS_DIR / "stage1b_server1_qwen3vl_departmentwise_metrics_final.csv",
        aggregate_metrics(metrics, "department_inferred"),
        ["group", "records", "parse_success_rate", "schema_validity_rate", "avg_entity_lenient_f1", "avg_hallucination_rate", "avg_missing_entity_rate", "avg_runtime_seconds"],
    )
    return metrics


def avg(vals: List[Any]) -> Any:
    nums = []
    for v in vals:
        try:
            if v not in ("", None):
                nums.append(float(v))
        except Exception:
            pass
    return round(sum(nums) / len(nums), 4) if nums else ""


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    manifest = read_csv(PROJECT_ROOT / "data" / "full_benchmark_manifest.csv")
    repairs = repair_failed(root)
    statuses = status_rows(root, manifest)
    write_csv(REPORTS_DIR / "stage1b_server1_qwen3vl_unique_status.csv", statuses, ["document_id", "patient_id", "department_inferred", "status", "parse_success", "schema_validity", "has_original_parsed_json", "has_repaired_json", "runtime_seconds", "log_path"])
    metrics = compute_metrics(root, manifest)
    failed = [m for m in metrics if str(m.get("schema_validity")) != "1"]
    write_text(REPORTS_DIR / "stage1b_server1_qwen3vl_failure_log_final.md", "\n".join([
        "# Stage 1B Final release Qwen3-VL Final Failure Log",
        "",
        f"- Remaining failures: {len(failed)}",
        *[f"- `{m['document_id']}`: {m.get('notes') or m.get('status')}" for m in failed],
        "",
    ]))
    write_text(REPORTS_DIR / "stage1b_server1_qwen3vl_final_status.md", "\n".join([
        "# Stage 1B Final release Qwen3-VL Final Status",
        "",
        f"- Output root: `{root}`",
        f"- Unique records: {len(statuses)}",
        f"- Original success/repaired schema-valid records: {sum(1 for m in metrics if str(m.get('schema_validity')) == '1')}",
        f"- Remaining failures: {len(failed)}",
        f"- Repair attempts: {len(repairs)}",
        *[f"- Repair `{r['document_id']}`: {r['repair_status']}" for r in repairs],
        "",
    ]))
    write_text(REPORTS_DIR / "stage1b_server1_qwen3vl_summary_final.md", "\n".join([
        "# Stage 1B Final release Qwen3-VL Final Summary",
        "",
        f"- Output root: `{root}`",
        f"- Total records: {len(metrics)}",
        f"- Parse success rate: {avg([m['json_parse_success'] for m in metrics])}",
        f"- Schema validity rate: {avg([m['schema_validity'] for m in metrics])}",
        f"- Average entity lenient F1: {avg([m['entity_lenient_f1'] for m in metrics])}",
        f"- Average hallucination rate: {avg([m['hallucination_rate'] for m in metrics])}",
        f"- Average missing entity rate: {avg([m['missing_entity_rate'] for m in metrics])}",
        f"- Remaining failures: {len(failed)}",
        "",
    ]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: finalize_stage1b_server1_qwen3vl.py OUTPUT_ROOT")
    main()
