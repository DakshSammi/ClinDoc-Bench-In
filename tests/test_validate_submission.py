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

import csv
import json

import yaml

from scripts.validate_submission import validate_submission, write_raw_ocr_handoff, write_structured_manifest


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_structured_submission_and_write_manifest(tmp_path):
    manifest_path = tmp_path / "benchmark_manifest.csv"
    write_csv(
        manifest_path,
        ["document_id", "patient_id", "benchmark_include", "ground_truth_json", "image_paths"],
        [
            {"document_id": "doc1", "patient_id": "p1", "benchmark_include": "true", "ground_truth_json": "gt/doc1.json", "image_paths": "images/doc1.jpg"},
            {"document_id": "doc2", "patient_id": "p2", "benchmark_include": "true", "ground_truth_json": "gt/doc2.json", "image_paths": "images/doc2.jpg"},
        ],
    )

    submission = tmp_path / "submission"
    predictions = submission / "predictions"
    predictions.mkdir(parents=True)
    (submission / "README.md").write_text("Example structured submission\n", encoding="utf-8")
    (submission / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "submission_name": "demo_structured_lane",
                "track": "hybrid",
                "model_name": "demo_model",
                "model_version": "v0",
                "provider": "local",
                "license": "apache-2.0",
                "hardware": "cpu",
                "benchmark_version": "ClinDoc-Bench-IN v1.0",
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        submission / "runtime.csv",
        ["document_id", "runtime_seconds"],
        [{"document_id": "doc1", "runtime_seconds": "1.2"}, {"document_id": "doc2", "runtime_seconds": "1.4"}],
    )
    for doc_id in ("doc1", "doc2"):
        (predictions / f"{doc_id}.json").write_text(
            json.dumps({"schema_version": "raw_rx_v2", "document_id": doc_id, "medications": [], "observations": [], "complaints_or_diagnosis": [], "procedures": [], "advice": [], "allergy_mentions": [], "other_notes": [], "lab_observations": []}),
            encoding="utf-8",
        )

    report = validate_submission(submission, manifest_path)
    assert report["valid"] is True
    assert report["validated_predictions"] == 2

    eval_manifest = tmp_path / "eval_manifest.csv"
    write_structured_manifest(eval_manifest, report["benchmark_rows"], submission)
    rows = list(csv.DictReader(eval_manifest.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["gt_path"] == "gt/doc1.json"
    assert rows[0]["prediction_path"].endswith("submission/predictions/doc1.json")


def test_validate_raw_ocr_submission_and_write_handoff(tmp_path):
    manifest_path = tmp_path / "benchmark_manifest.csv"
    write_csv(
        manifest_path,
        ["document_id", "patient_id", "benchmark_include", "ground_truth_json", "image_paths"],
        [{"document_id": "doc1", "patient_id": "p1", "benchmark_include": "true", "ground_truth_json": "gt/doc1.json", "image_paths": "images/doc1.jpg"}],
    )

    submission = tmp_path / "raw_submission"
    predictions = submission / "predictions"
    predictions.mkdir(parents=True)
    (submission / "README.md").write_text("Example OCR submission\n", encoding="utf-8")
    (submission / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "submission_name": "demo_ocr_lane",
                "track": "raw_ocr",
                "model_name": "demo_ocr",
                "model_version": "v0",
                "provider": "local",
                "license": "apache-2.0",
                "hardware": "cpu",
                "benchmark_version": "ClinDoc-Bench-IN v1.0",
            }
        ),
        encoding="utf-8",
    )
    write_csv(submission / "runtime.csv", ["document_id", "runtime_seconds"], [{"document_id": "doc1", "runtime_seconds": "0.9"}])
    (predictions / "doc1.txt").write_text("sample OCR output", encoding="utf-8")

    report = validate_submission(submission, manifest_path)
    assert report["valid"] is True
    assert report["validated_predictions"] == 1

    handoff = tmp_path / "ocr_handoff.csv"
    write_raw_ocr_handoff(handoff, report["benchmark_rows"], submission, report["metadata"], report["runtime_rows"], report["runtime_field"])
    rows = list(csv.DictReader(handoff.open(encoding="utf-8")))
    assert rows[0]["ocr_engine"] == "demo_ocr"
    assert rows[0]["ocr_text_path"].endswith("raw_submission/predictions/doc1.txt")
