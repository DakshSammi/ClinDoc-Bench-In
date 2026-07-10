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

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.raw_extraction import CanonicalRawDoc


TRUTHY = {"1", "true", "yes", "y", "t"}
SUPPORTED_TRACKS = {"raw_ocr", "direct_vlm", "hybrid"}
REQUIRED_METADATA_FIELDS = {
    "submission_name",
    "track",
    "model_name",
    "model_version",
    "provider",
    "license",
    "hardware",
    "benchmark_version",
}


def is_truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in TRUTHY


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_benchmark_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    filtered: dict[str, dict[str, str]] = {}
    for row in rows:
        include = is_truthy(row.get("benchmark_include"), default=True)
        if include:
            filtered[str(row["document_id"])] = row
    return filtered


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("metadata.yaml must contain a YAML mapping at the top level.")
    return data


def validate_runtime_csv(path: Path, expected_doc_ids: list[str]) -> tuple[list[str], list[str], list[dict[str, str]], str | None]:
    issues: list[str] = []
    rows = read_csv_rows(path)
    if not rows:
        return ["runtime.csv is empty."], [], [], None

    runtime_field = None
    for candidate in ("runtime_seconds", "latency_ms", "processing_time_ms"):
        if candidate in rows[0]:
            runtime_field = candidate
            break
    if runtime_field is None:
        issues.append("runtime.csv must contain one of: runtime_seconds, latency_ms, processing_time_ms.")

    seen: set[str] = set()
    present: list[str] = []
    for row in rows:
        doc_id = str(row.get("document_id", "")).strip()
        if not doc_id:
            issues.append("runtime.csv contains a row without document_id.")
            continue
        if doc_id in seen:
            issues.append(f"runtime.csv contains duplicate document_id '{doc_id}'.")
            continue
        seen.add(doc_id)
        present.append(doc_id)
        if runtime_field and str(row.get(runtime_field, "")).strip() == "":
            issues.append(f"runtime.csv is missing {runtime_field} for document '{doc_id}'.")

    expected = set(expected_doc_ids)
    missing = sorted(expected - seen)
    if missing:
        issues.append(f"runtime.csv is missing {len(missing)} benchmark document(s).")

    unexpected = sorted(seen - expected)
    return issues, unexpected, rows, runtime_field


def validate_structured_predictions(predictions_dir: Path, expected_doc_ids: list[str]) -> tuple[list[str], int]:
    issues: list[str] = []
    valid_count = 0
    for doc_id in expected_doc_ids:
        path = predictions_dir / f"{doc_id}.json"
        if not path.exists():
            issues.append(f"Missing prediction JSON for '{doc_id}'.")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            doc = CanonicalRawDoc(**data)
            if str(doc.document_id) != doc_id:
                issues.append(f"Prediction '{path.name}' contains document_id '{doc.document_id}', expected '{doc_id}'.")
                continue
            valid_count += 1
        except Exception as exc:
            issues.append(f"Prediction '{path.name}' failed schema validation: {exc}")
    return issues, valid_count


def validate_raw_ocr_predictions(predictions_dir: Path, expected_doc_ids: list[str]) -> tuple[list[str], int]:
    issues: list[str] = []
    valid_count = 0
    for doc_id in expected_doc_ids:
        path = predictions_dir / f"{doc_id}.txt"
        if not path.exists():
            issues.append(f"Missing OCR text file for '{doc_id}'.")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            issues.append(f"OCR text file '{path.name}' is empty.")
            continue
        valid_count += 1
    return issues, valid_count


def write_structured_manifest(path: Path, benchmark_rows: dict[str, dict[str, str]], submission_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for doc_id, row in benchmark_rows.items():
        rows.append(
            {
                "document_id": doc_id,
                "image_path": row.get("image_path") or row.get("image_paths", ""),
                "gt_path": row.get("gt_path") or row.get("ground_truth_json") or row.get("ground_truth_path", ""),
                "prediction_path": project_relative(submission_dir / "predictions" / f"{doc_id}.json"),
                "patient_id": row.get("patient_id", ""),
                "prescription_id": row.get("prescription_id", doc_id),
                "page_number": row.get("page_number", "1"),
                "split": "community_submission",
            }
        )
    write_csv_rows(
        path,
        ["document_id", "image_path", "gt_path", "prediction_path", "patient_id", "prescription_id", "page_number", "split"],
        rows,
    )


def write_raw_ocr_handoff(path: Path, benchmark_rows: dict[str, dict[str, str]], submission_dir: Path, metadata: dict[str, Any], runtime_rows: list[dict[str, str]], runtime_field: str | None) -> None:
    runtime_map = {str(row["document_id"]).strip(): row for row in runtime_rows}
    rows: list[dict[str, Any]] = []
    for doc_id in benchmark_rows:
        runtime_value = ""
        if runtime_field:
            runtime_value = runtime_map.get(doc_id, {}).get(runtime_field, "")
        rows.append(
            {
                "document_id": doc_id,
                "ocr_engine": metadata.get("model_name", "unknown"),
                "ocr_text_path": project_relative(submission_dir / "predictions" / f"{doc_id}.txt"),
                "runtime": runtime_value,
                "status": "success",
            }
        )
    write_csv_rows(path, ["document_id", "ocr_engine", "ocr_text_path", "runtime", "status"], rows)


def validate_submission(submission_dir: Path, manifest_path: Path) -> dict[str, Any]:
    issues: list[str] = []
    notes: list[str] = []

    metadata_path = submission_dir / "metadata.yaml"
    predictions_dir = submission_dir / "predictions"
    runtime_path = submission_dir / "runtime.csv"
    readme_path = submission_dir / "README.md"

    if not submission_dir.exists():
        raise FileNotFoundError(f"Submission directory does not exist: {submission_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.yaml in {submission_dir}")

    metadata = load_metadata(metadata_path)
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - set(metadata.keys()))
    if missing_metadata:
        issues.append(f"metadata.yaml is missing required field(s): {', '.join(missing_metadata)}.")

    track = str(metadata.get("track", "")).strip()
    if track not in SUPPORTED_TRACKS:
        issues.append(f"metadata.yaml track must be one of: {', '.join(sorted(SUPPORTED_TRACKS))}.")

    if not predictions_dir.exists():
        issues.append("predictions/ directory is missing.")
    if not runtime_path.exists():
        issues.append("runtime.csv is missing.")
    if not readme_path.exists():
        issues.append("README.md is missing from the submission directory.")

    benchmark_rows = load_benchmark_rows(manifest_path)
    expected_doc_ids = sorted(benchmark_rows.keys())
    notes.append(f"Expected benchmark documents: {len(expected_doc_ids)}")

    runtime_rows: list[dict[str, str]] = []
    runtime_field: str | None = None
    if runtime_path.exists():
        runtime_issues, unexpected_runtime_docs, runtime_rows, runtime_field = validate_runtime_csv(runtime_path, expected_doc_ids)
        issues.extend(runtime_issues)
        if unexpected_runtime_docs:
            notes.append(f"runtime.csv contains {len(unexpected_runtime_docs)} extra document_id value(s).")

    valid_predictions = 0
    if predictions_dir.exists() and track in SUPPORTED_TRACKS:
        if track == "raw_ocr":
            pred_issues, valid_predictions = validate_raw_ocr_predictions(predictions_dir, expected_doc_ids)
        else:
            pred_issues, valid_predictions = validate_structured_predictions(predictions_dir, expected_doc_ids)
        issues.extend(pred_issues)

    return {
        "valid": not issues,
        "track": track,
        "metadata": metadata,
        "expected_documents": len(expected_doc_ids),
        "validated_predictions": valid_predictions,
        "issues": issues,
        "notes": notes,
        "benchmark_rows": benchmark_rows,
        "runtime_rows": runtime_rows,
        "runtime_field": runtime_field,
    }


def print_report(report: dict[str, Any]) -> None:
    print("Submission validation report")
    print("=" * 80)
    print(f"[OK] track: {report.get('track') or 'unknown'}")
    print(f"[OK] expected benchmark documents: {report['expected_documents']}")
    print(f"[OK] validated predictions: {report['validated_predictions']}")
    for note in report.get("notes", []):
        print(f"[OK] {note}")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"[FAIL] {issue}")
    else:
        print("[OK] schema valid")
        print("[OK] all benchmark documents present")
        print("[OK] runtime available")
        print("[OK] model metadata complete")
        print("[OK] benchmark ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a community benchmark submission against ClinDoc-Bench-IN conventions.")
    parser.add_argument("--submission-dir", required=True, help="Directory containing metadata.yaml, predictions/, runtime.csv, and README.md.")
    parser.add_argument(
        "--manifest",
        default="benchmark/data/benchmark_manifest.csv",
        help="Benchmark manifest used to determine the expected document set.",
    )
    parser.add_argument(
        "--write-benchmark-manifest",
        default=None,
        help="For direct_vlm or hybrid submissions, write a structured evaluation manifest to this CSV path.",
    )
    parser.add_argument(
        "--write-ocr-handoff",
        default=None,
        help="For raw_ocr submissions, write an OCR handoff CSV to this path.",
    )
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir)
    if not submission_dir.is_absolute():
        submission_dir = PROJECT_ROOT / submission_dir
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    report = validate_submission(submission_dir, manifest_path)
    print_report(report)

    if report["valid"] and args.write_benchmark_manifest:
        if report["track"] not in {"direct_vlm", "hybrid"}:
            raise SystemExit("--write-benchmark-manifest is only valid for direct_vlm or hybrid submissions.")
        out_path = Path(args.write_benchmark_manifest)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        write_structured_manifest(out_path, report["benchmark_rows"], submission_dir)
        print(f"[OK] wrote structured evaluation manifest: {out_path}")

    if report["valid"] and args.write_ocr_handoff:
        if report["track"] != "raw_ocr":
            raise SystemExit("--write-ocr-handoff is only valid for raw_ocr submissions.")
        out_path = Path(args.write_ocr_handoff)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        write_raw_ocr_handoff(out_path, report["benchmark_rows"], submission_dir, report["metadata"], report["runtime_rows"], report["runtime_field"])
        print(f"[OK] wrote OCR handoff CSV: {out_path}")

    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
