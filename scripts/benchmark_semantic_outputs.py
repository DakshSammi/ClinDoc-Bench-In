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

"""Benchmark Stage 1C outputs using evidence-consistency and coverage proxies."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas.semantic_extraction import SemanticExtractionDoc


RAW_LISTS = ["complaints_or_diagnosis", "observations", "medications", "procedures", "advice", "allergy_mentions", "other_notes", "lab_observations"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def raw_entities(raw: dict[str, Any]) -> dict[str, Any]:
    return raw.get("raw_entities", raw)


def raw_count(raw: dict[str, Any]) -> int:
    entities = raw_entities(raw)
    count = sum(len(as_list(entities.get(key))) for key in RAW_LISTS)
    patient = entities.get("patient_information", raw.get("patient_information", {}))
    encounter = entities.get("encounter_information", raw.get("encounter_information", {}))
    if isinstance(patient, dict):
        count += sum(1 for value in patient.values() if value not in (None, "", [], {}))
    if isinstance(encounter, dict):
        count += sum(1 for value in encounter.values() if value not in (None, "", [], {}))
    return count


def raw_medication_count(raw: dict[str, Any]) -> int:
    return len(as_list(raw_entities(raw).get("medications")))


def raw_frequency_count(raw: dict[str, Any]) -> int:
    count = 0
    for item in as_list(raw_entities(raw).get("medications")):
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get(key, "")) for key in ["raw_frequency", "raw_frequency_text", "raw_line_text", "raw_medication_text"])
        if item.get("raw_frequency") or item.get("raw_frequency_text") or re.search(r"\b(?:OD|BD|BID|TDS|TID|QID|HS|SOS)\b|\b\d\s*-\s*\d(?:\s*-\s*\d)?\b", text, re.I):
            count += 1
    return count


def raw_diagnosis_count(raw: dict[str, Any]) -> int:
    return len(as_list(raw_entities(raw).get("complaints_or_diagnosis")))


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def review_reasons(entity: dict[str, Any]) -> list[str]:
    reasons = []
    confidence = float(entity.get("confidence", 0) or 0)
    evidence = str(entity.get("raw_evidence_text", "")).strip()
    semantic_type = entity.get("semantic_type", "")
    method = str(entity.get("normalization_method", "")).lower()
    source = str(entity.get("source_raw_field", "")).lower()
    if confidence < 0.7:
        reasons.append("low confidence")
    if not evidence:
        reasons.append("no evidence quote")
    if semantic_type == "medication" and (not entity.get("evidence_supported") or not evidence):
        reasons.append("normalized medication not clearly supported")
    if semantic_type in {"frequency", "dosage"} and ("abbreviation" in method or re.search(r"\b(?:OD|BD|BID|TDS|TID|QID|HS|SOS)\b", evidence, re.I)):
        reasons.append("frequency/dose inferred from abbreviation")
    if semantic_type == "diagnosis" and ("medication" in source or "treatment" in source or "treatment" in method):
        reasons.append("diagnosis inferred from treatment only")
    if not entity.get("evidence_supported"):
        reasons.append("evidence_supported=false")
    return list(dict.fromkeys(reasons))


def score_output(output_path: Path, log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = load_json(output_path)
    log = load_json(log_path)
    validated = SemanticExtractionDoc(**output)
    entities = [item.model_dump() for item in validated.semantic_entities]
    unsupported = [item.model_dump() for item in validated.unsupported_inferences]
    raw_path = Path(validated.metadata.source_stage1b_path)
    raw = load_json(raw_path)
    raw_total = raw_count(raw)
    raw_medications = raw_medication_count(raw)
    raw_frequencies = raw_frequency_count(raw)
    raw_diagnoses = raw_diagnosis_count(raw)
    evidence_linked = sum(1 for item in entities if item.get("evidence_supported") and item.get("source_raw_field"))
    evidence_quotes = sum(1 for item in entities if str(item.get("raw_evidence_text", "")).strip())
    medications = sum(1 for item in entities if item.get("semantic_type") == "medication")
    frequencies = sum(1 for item in entities if item.get("semantic_type") == "frequency")
    diagnoses = sum(1 for item in entities if item.get("semantic_type") in {"diagnosis", "complaint"})
    unsupported_entities = sum(1 for item in entities if not item.get("evidence_supported"))
    review = []
    for item in entities:
        for reason in review_reasons(item):
            review.append(
                {
                    "document_id": validated.document_id,
                    "system": validated.source_system,
                    "raw_evidence_text": item.get("raw_evidence_text", ""),
                    "normalized_name": item.get("normalized_name", ""),
                    "semantic_type": item.get("semantic_type", ""),
                    "confidence": item.get("confidence", ""),
                    "reason_for_review": reason,
                }
            )
    for item in unsupported:
        review.append(
            {
                "document_id": validated.document_id,
                "system": validated.source_system,
                "raw_evidence_text": item.get("raw_evidence_text", ""),
                "normalized_name": item.get("inferred_claim", ""),
                "semantic_type": "unsupported_inference",
                "confidence": item.get("confidence", ""),
                "reason_for_review": item.get("reason", "unsupported inference"),
            }
        )
    inference_total = len(entities) + len(unsupported)
    row = {
        "document_id": validated.document_id,
        "system": validated.source_system,
        "semantic_json_parse_success": 1,
        "semantic_schema_validity": 1,
        "semantic_entity_count": len(entities),
        "raw_entity_count": raw_total,
        "semantic_coverage_over_raw_entities": round(min(1.0, evidence_linked / max(1, raw_total)), 4),
        "medication_normalization_coverage": round(min(1.0, medications / max(1, raw_medications)), 4) if raw_medications else 0.0,
        "frequency_normalization_coverage": round(min(1.0, frequencies / max(1, raw_frequencies)), 4) if raw_frequencies else 0.0,
        "diagnosis_complaint_normalization_coverage": round(min(1.0, diagnoses / max(1, raw_diagnoses)), 4) if raw_diagnoses else 0.0,
        "evidence_linkage_rate": ratio(evidence_linked, len(entities)),
        "evidence_quote_present_rate": ratio(evidence_quotes, len(entities)),
        "unsupported_inference_rate": ratio(len(unsupported), inference_total),
        "contradiction_unsupported_claim_rate": ratio(len(unsupported) + unsupported_entities, inference_total),
        "manual_review_required_count": len(review),
        "runtime_seconds": log.get("runtime_seconds", validated.metadata.processing_time_ms / 1000.0),
        "status": "success",
        "notes": "evidence-backed semantic consistency metrics; no semantic gold accuracy claim",
    }
    return row, review


def avg(rows: list[dict[str, Any]], key: str) -> float | str:
    values = []
    for row in rows:
        try:
            if row.get(key) not in (None, ""):
                values.append(float(row[key]))
        except Exception:
            continue
    return round(sum(values) / len(values), 4) if values else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-prefix", default="stage1c_semantic")
    args = parser.parse_args()
    root = Path(args.output_root)
    systems = sorted(path.name for path in (root / "semantic_outputs").glob("*") if path.is_dir())
    rows = []
    review_rows = []
    errors = []
    for system in systems:
        for output_path in sorted((root / "semantic_outputs" / system).glob("*.json")):
            log_path = root / "logs" / system / output_path.name
            try:
                row, review = score_output(output_path, log_path)
                rows.append(row)
                review_rows.extend(review)
            except Exception as exc:
                errors.append({"system": system, "document_id": output_path.stem, "error_type": "benchmark_validation_error", "error": f"{type(exc).__name__}: {exc}"})
    for failed_path in sorted((root / "failed_cases").glob("*/*.json")):
        failed = load_json(failed_path)
        errors.append({"system": failed_path.parent.name, "document_id": failed_path.stem, "error_type": "semantic_generation_failure", "error": failed.get("error", "")})
    for missing_path in sorted((root / "missing_inputs").glob("*/*.json")):
        missing = load_json(missing_path)
        errors.append({"system": missing_path.parent.name, "document_id": missing_path.stem, "error_type": "valid_stage1b_input_missing", "error": missing.get("reason", "")})

    fields = [
        "document_id", "system", "semantic_json_parse_success", "semantic_schema_validity", "semantic_entity_count",
        "raw_entity_count", "semantic_coverage_over_raw_entities", "medication_normalization_coverage",
        "frequency_normalization_coverage", "diagnosis_complaint_normalization_coverage", "evidence_linkage_rate",
        "evidence_quote_present_rate", "unsupported_inference_rate", "contradiction_unsupported_claim_rate",
        "manual_review_required_count", "runtime_seconds", "status", "notes",
    ]
    write_csv(ROOT / "reports" / f"{args.report_prefix}_metrics.csv", rows, fields)
    review_fields = ["document_id", "system", "raw_evidence_text", "normalized_name", "semantic_type", "confidence", "reason_for_review"]
    write_csv(ROOT / "reports" / "stage1c_semantic_manual_review_queue.csv", review_rows, review_fields)
    error_fields = ["system", "document_id", "error_type", "error"]
    write_csv(ROOT / "reports" / f"{args.report_prefix}_errors.csv", errors, error_fields)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["system"]].append(row)
    error_counts = Counter(item["system"] for item in errors)
    summary_rows = []
    metric_keys = fields[2:16]
    for system in sorted(set(grouped) | set(error_counts)):
        items = grouped.get(system, [])
        attempted = len(items) + error_counts.get(system, 0)
        summary = {"system": system, "records_attempted": attempted, "records_semantic_valid": len(items), "semantic_validity_rate": ratio(len(items), attempted)}
        for key in metric_keys:
            summary[key] = avg(items, key)
        summary["semantic_json_parse_success"] = ratio(len(items), attempted)
        summary["semantic_schema_validity"] = ratio(len(items), attempted)
        summary["manual_review_required_count"] = sum(int(item.get("manual_review_required_count", 0) or 0) for item in items)
        summary_rows.append(summary)
    summary_fields = ["system", "records_attempted", "records_semantic_valid", "semantic_validity_rate", *metric_keys]
    write_csv(ROOT / "reports" / f"{args.report_prefix}_per_system_summary.csv", summary_rows, summary_fields)

    lines = ["# Stage 1C Semantic Benchmark Summary", "", f"Generated: {now()}", "", "These are evidence-backed semantic consistency, semantic coverage, and manual-review proxy metrics. No gold semantic accuracy is claimed.", ""]
    for row in summary_rows:
        lines.append(f"- `{row['system']}`: valid={row['records_semantic_valid']}/{row['records_attempted']}, semantic coverage={row['semantic_coverage_over_raw_entities']}, evidence linkage={row['evidence_linkage_rate']}, unsupported rate={row['unsupported_inference_rate']}, review count={row['manual_review_required_count']}")
    write_text(ROOT / "reports" / f"{args.report_prefix}_benchmark_summary.md", "\n".join(lines))
    taxonomy = Counter(item["error_type"] for item in errors)
    taxonomy_lines = ["# Stage 1C Semantic Error Taxonomy", "", f"Generated: {now()}", ""] + [f"- `{key}`: {value}" for key, value in sorted(taxonomy.items())]
    write_text(ROOT / "reports" / f"{args.report_prefix}_error_taxonomy.md", "\n".join(taxonomy_lines))

    paper_columns = [
        ("system", "System"), ("records_attempted", "N"), ("records_semantic_valid", "Valid"),
        ("semantic_validity_rate", "Schema validity"), ("semantic_coverage_over_raw_entities", "Semantic coverage"),
        ("medication_normalization_coverage", "Medication coverage"), ("frequency_normalization_coverage", "Frequency coverage"),
        ("diagnosis_complaint_normalization_coverage", "Diagnosis/complaint coverage"), ("evidence_linkage_rate", "Evidence linkage"),
        ("evidence_quote_present_rate", "Evidence quote"), ("unsupported_inference_rate", "Unsupported inference"),
        ("contradiction_unsupported_claim_rate", "Unsupported claim proxy"), ("manual_review_required_count", "Review count"),
        ("runtime_seconds", "Runtime s"),
    ]
    header = "| " + " | ".join(label for _, label in paper_columns) + " |"
    divider = "|" + "|".join("---" for _ in paper_columns) + "|"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in paper_columns) + " |" for row in summary_rows]
    write_text(ROOT / "reports" / f"{args.report_prefix}_paper_table.md", "\n".join(["# Stage 1C Semantic Paper Table", "", "Evidence-backed semantic consistency metrics; no gold semantic accuracy claim.", "", header, divider, *body]))
    print(json.dumps({"systems": systems, "valid_outputs": len(rows), "errors": len(errors), "review_rows": len(review_rows)}, indent=2))


if __name__ == "__main__":
    main()
