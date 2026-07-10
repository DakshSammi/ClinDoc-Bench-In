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

"""Benchmark a completed Stage 1B canonical structured-output lane."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.gt_adapter import GTAdapter
from src.benchmark.aggregation import MetricAggregator
from src.benchmark.entity_match import EntityMatcher
from src.benchmark.hallucination import HallucinationDetector
from src.benchmark.scalar_match import ScalarMatcher
from src.schemas.benchmark import DocumentBenchmarkResult
from src.schemas.raw_extraction import CanonicalRawDoc


CATEGORIES = [
    "complaints_or_diagnosis",
    "observations",
    "medications",
    "procedures",
    "advice",
    "allergy_mentions",
    "other_notes",
    "lab_observations",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def avg(rows: list[dict[str, Any]], key: str) -> float | str:
    values = []
    for row in rows:
        try:
            if row.get(key) not in (None, ""):
                values.append(float(row[key]))
        except (TypeError, ValueError):
            continue
    return round(sum(values) / len(values), 4) if values else ""


def evaluate_prediction(doc: dict[str, str], prediction_path: Path, runtime: Any) -> dict[str, Any]:
    gt_doc = GTAdapter.from_file(PROJECT_ROOT / doc["ground_truth_json"])
    pred_data = json.loads(prediction_path.read_text(encoding="utf-8"))
    pred_doc = CanonicalRawDoc(**pred_data)

    scalar_matches = ScalarMatcher(lenient_threshold=80.0).match_docs(gt_doc, pred_doc)
    scalar_exact = sum(1 for match in scalar_matches if match.exact_match) / max(1, len(scalar_matches))
    scalar_lenient = sum(1 for match in scalar_matches if match.lenient_match) / max(1, len(scalar_matches))

    entity_matcher = EntityMatcher(exact_threshold=95.0, lenient_threshold=80.0, review_threshold=65.0)
    all_alignments = []
    metrics_by_category = {}
    total_gt = 0
    total_pred = 0
    for category in CATEGORIES:
        gt_items = getattr(gt_doc, category, [])
        pred_items = getattr(pred_doc, category, [])
        total_gt += len(gt_items)
        total_pred += len(pred_items)
        alignments = entity_matcher.align_entities(gt_items, pred_items, category)
        all_alignments.extend(alignments)
        metrics_by_category[category] = entity_matcher.compute_category_metrics(alignments)

    unmatched = HallucinationDetector(hallucination_threshold=60.0, gap_threshold=80.0).detect_hallucinations(
        gt_doc, pred_doc, all_alignments
    )
    result = DocumentBenchmarkResult(
        document_id=doc["document_id"],
        document_type=doc.get("source_type") or "unknown",
        schema_parse_success=1,
        scalar_accuracy_exact=scalar_exact,
        scalar_accuracy_lenient=scalar_lenient,
        metrics_by_category=metrics_by_category,
        likely_hallucination_count=sum(1 for item in unmatched if item.classification == "likely_hallucination"),
        annotation_gap_candidate_count=sum(1 for item in unmatched if item.classification == "annotation_gap_candidate"),
        manual_review_required_count=sum(1 for item in unmatched if item.classification == "manual_review_required"),
        scalars=scalar_matches,
        entity_alignments=all_alignments,
        unmatched_predictions=unmatched,
        model_name=pred_doc.metadata.model_name if pred_doc.metadata else "unknown",
        backend_name=pred_doc.metadata.backend_name if pred_doc.metadata else "unknown",
        latency_ms=float(runtime or 0) * 1000.0,
    )
    aggregator = MetricAggregator()
    aggregator.calculate_rates(result, total_gt, total_pred)
    aggregator.compute_experimental_score(result)
    entity_exact = sum(item.f1_exact for item in metrics_by_category.values()) / len(metrics_by_category)
    entity_lenient = sum(item.f1_lenient for item in metrics_by_category.values()) / len(metrics_by_category)
    return {
        "scalar_exact_accuracy": round(scalar_exact, 4),
        "scalar_lenient_accuracy": round(scalar_lenient, 4),
        "entity_exact_f1": round(entity_exact, 4),
        "entity_lenient_f1": round(entity_lenient, 4),
        "hallucination_rate": round(result.hallucination_rate, 4),
        "missing_entity_rate": round(result.missing_entity_rate, 4),
        "annotation_gap_rate": round(result.annotation_gap_rate, 4),
        "overall_extraction_score": round(result.experimental_overall_score, 4),
        "likely_hallucination_count": result.likely_hallucination_count,
        "annotation_gap_candidate_count": result.annotation_gap_candidate_count,
        "manual_review_required_count": result.manual_review_required_count,
    }


def dataset_overall(summary: dict[str, Any]) -> float:
    return round(
        0.10 * float(summary["schema_parse_success"])
        + 0.20 * float(summary["scalar_lenient_accuracy"] or 0)
        + 0.45 * float(summary["entity_lenient_f1"] or 0)
        + 0.15 * float(summary["entity_exact_f1"] or 0)
        + 0.10 * (1.0 - float(summary["hallucination_rate"] or 0)),
        4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--failure-log", required=True)
    parser.add_argument("--error-taxonomy", required=True)
    parser.add_argument("--status-md", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest = read_csv(Path(args.manifest))
    rows = []
    failures = []
    for doc in manifest:
        doc_id = doc["document_id"]
        log = load_json(output_root / "logs" / args.lane / f"{doc_id}.json")
        prediction = output_root / "raw_structured" / args.lane / f"{doc_id}.json"
        base = {
            "document_id": doc_id,
            "system": args.system,
            "department": doc.get("department_inferred", ""),
            "schema_parse_success": 0,
            "scalar_exact_accuracy": "",
            "scalar_lenient_accuracy": "",
            "entity_exact_f1": "",
            "entity_lenient_f1": "",
            "hallucination_rate": "",
            "missing_entity_rate": "",
            "annotation_gap_rate": "",
            "overall_extraction_score": "",
            "runtime_seconds": log.get("runtime_seconds", ""),
            "status": "failed",
            "failure_label": log.get("failure_label", "") or "missing_or_invalid_output",
            "notes": log.get("error", ""),
        }
        if prediction.exists():
            try:
                metrics = evaluate_prediction(doc, prediction, log.get("runtime_seconds", 0))
                base.update(metrics)
                base.update({"schema_parse_success": 1, "status": "success", "failure_label": "", "notes": ""})
            except Exception as exc:
                base["failure_label"] = "schema_invalid_wrong_json_shape"
                base["notes"] = f"{type(exc).__name__}: {exc}"
        if base["status"] != "success":
            failures.append(base)
        rows.append(base)

    valid = [row for row in rows if row["schema_parse_success"] == 1]
    summary = {
        "system": args.system,
        "records_attempted": len(rows),
        "records_schema_valid": len(valid),
        "schema_parse_success": round(len(valid) / max(1, len(rows)), 4),
        "scalar_exact_accuracy": avg(valid, "scalar_exact_accuracy"),
        "scalar_lenient_accuracy": avg(valid, "scalar_lenient_accuracy"),
        "entity_exact_f1": avg(valid, "entity_exact_f1"),
        "entity_lenient_f1": avg(valid, "entity_lenient_f1"),
        "hallucination_rate": avg(valid, "hallucination_rate"),
        "missing_entity_rate": avg(valid, "missing_entity_rate"),
        "annotation_gap_rate": avg(valid, "annotation_gap_rate"),
        "runtime_seconds": avg(rows, "runtime_seconds"),
    }
    summary["overall_extraction_score"] = dataset_overall(summary)

    fields = [
        "document_id", "system", "department", "schema_parse_success", "scalar_exact_accuracy",
        "scalar_lenient_accuracy", "entity_exact_f1", "entity_lenient_f1", "hallucination_rate",
        "missing_entity_rate", "annotation_gap_rate", "overall_extraction_score", "runtime_seconds",
        "status", "failure_label", "notes", "likely_hallucination_count",
        "annotation_gap_candidate_count", "manual_review_required_count",
    ]
    write_csv(Path(args.metrics_csv), rows, fields)
    write_json(Path(args.summary_json), summary)

    metrics_lines = [f"- {key.replace('_', ' ')}: {value}" for key, value in summary.items() if key != "system"]
    write_text(
        Path(args.summary_md),
        f"# {args.system} PPT-Aligned Summary\n\nGenerated: {now()}\n\n" + "\n".join(metrics_lines),
    )
    failure_lines = [
        f"- `{row['document_id']}`: `{row['failure_label']}`; runtime={row['runtime_seconds']}s; {row['notes']}"
        for row in failures
    ] or ["- None"]
    write_text(Path(args.failure_log), f"# {args.system} Failure Log\n\nGenerated: {now()}\n\n" + "\n".join(failure_lines))
    counts = Counter(row["failure_label"] for row in failures)
    stalled = [row for row in failures if float(row.get("runtime_seconds") or 0) >= 600]
    taxonomy = [f"- `{label}`: {count}" for label, count in sorted(counts.items())] or ["- No failures"]
    taxonomy.extend(
        [f"- Stall-like runtime: `{row['document_id']}` at {row['runtime_seconds']}s" for row in stalled]
    )
    write_text(Path(args.error_taxonomy), f"# {args.system} Error Taxonomy\n\nGenerated: {now()}\n\n" + "\n".join(taxonomy))
    write_text(
        Path(args.status_md),
        f"# {args.system} Full Status\n\nGenerated: {now()}\n\n"
        f"- Output root: `{output_root}`\n- Lane: `{args.lane}`\n"
        f"- Completed: {len(rows)}/{len(manifest)}\n- Schema-valid: {len(valid)}\n- Failed: {len(failures)}\n",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
