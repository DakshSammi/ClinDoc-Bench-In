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

"""Canonical Stage 1B OCR-to-JSON runner with smoke gating.

Input is an OCR handoff CSV. Output is strict raw_rx_v2 CanonicalRawDoc JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CANONICAL_KEYS = {
    "schema_version",
    "document_id",
    "patient_information",
    "encounter_information",
    "complaints_or_diagnosis",
    "observations",
    "medications",
    "procedures",
    "advice",
    "allergy_mentions",
    "other_notes",
    "lab_observations",
}


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def extract_json(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    for fence in ("```json", "```"):
        if fence in text:
            try:
                candidates.append(text.split(fence, 1)[1].split("```", 1)[0].strip())
            except Exception:
                pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in list(candidates):
        candidates.append(re.sub(r",\s*([}\]])", r"\1", candidate))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def validate_canonical(obj: dict[str, Any] | None, expected_doc_id: str) -> tuple[bool, str]:
    if not obj:
        return False, "json_parse_failed"
    if obj.get("schema_version") != "raw_rx_v2":
        return False, "schema_invalid_wrong_json_shape: missing schema_version=raw_rx_v2"
    missing = sorted(key for key in CANONICAL_KEYS if key not in obj)
    if missing:
        return False, "schema_invalid_wrong_json_shape: missing " + ",".join(missing)
    if str(obj.get("document_id", "")) != expected_doc_id:
        return False, f"schema_invalid_wrong_json_shape: document_id mismatch {obj.get('document_id')}"
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.schemas.raw_extraction import CanonicalRawDoc

        CanonicalRawDoc(**obj)
    except Exception as exc:
        return False, f"schema_invalid_wrong_json_shape: {type(exc).__name__}: {exc}"
    return True, ""


def ollama_chat(model: str, prompt: str, timeout: int) -> tuple[bool, str, dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        if response.status_code >= 400:
            return False, "", {}, f"HTTP {response.status_code}: {response.text[:500]}"
        data = response.json()
        return True, data.get("message", {}).get("content", ""), data, ""
    except Exception as exc:
        return False, "", {}, f"{type(exc).__name__}: {exc}"


def canonical_prompt(doc: dict[str, str], ocr_text: str, engine: str) -> str:
    skeleton = {
        "schema_version": "raw_rx_v2",
        "document_id": doc["document_id"],
        "patient_information": {
            "name": None,
            "age": None,
            "gender": None,
            "address": None,
            "phone": None,
            "patient_identifier": None,
            "abha_id": None,
        },
        "encounter_information": {
            "date": None,
            "department": None,
            "hospital_name": None,
            "doctor_name": None,
            "visit_type": None,
            "fees": None,
            "room_or_queue_no": None,
        },
        "complaints_or_diagnosis": [{"raw_text": "", "evidence_text": "", "page_number": 1, "section": ""}],
        "observations": [{"raw_text": "", "evidence_text": "", "page_number": 1, "section": ""}],
        "medications": [
            {
                "raw_line_text": "",
                "raw_name": None,
                "raw_dosage": None,
                "raw_route": None,
                "raw_frequency": None,
                "raw_duration": None,
                "raw_instruction": None,
                "raw_timing": None,
                "evidence_text": "",
                "page_number": 1,
                "section": "",
            }
        ],
        "procedures": [{"raw_text": "", "evidence_text": "", "page_number": 1, "section": ""}],
        "advice": [{"raw_text": "", "evidence_text": "", "page_number": 1, "section": ""}],
        "follow_up": {"raw_text": None, "date": None, "review_after": None},
        "allergy_mentions": [{"raw_text": "", "evidence_text": "", "page_number": 1, "section": ""}],
        "other_notes": [{"raw_text": "", "evidence_text": "", "page_number": 1, "section": ""}],
        "lab_observations": [
            {
                "raw_line_text": "",
                "test_name": None,
                "result": None,
                "unit": None,
                "reference_range": None,
                "evidence_text": "",
                "page_number": 1,
                "section": "",
            }
        ],
        "metadata": {
            "model_name": "MODEL_NAME",
            "backend_name": "ocr_to_json",
            "schema_version": "raw_rx_v2",
            "processing_time_ms": 0.0,
            "document_type": doc.get("source_type") or "unknown",
            "validation_warnings": [],
            "pages": [],
        },
    }
    return (
        "Return exactly one valid JSON object. No markdown, no commentary, no extra keys outside the schema.\n"
        "You are converting noisy OCR text into the CURRENT CanonicalRawDoc schema.\n"
        "Hard requirements:\n"
        "- Top-level schema_version MUST be raw_rx_v2.\n"
        "- Top-level document_id MUST match the provided document id.\n"
        "- Use patient_information and encounter_information objects exactly as shown.\n"
        "- Use list fields exactly as shown: complaints_or_diagnosis, observations, medications, procedures, advice, allergy_mentions, other_notes, lab_observations.\n"
        "- Do NOT use document_metadata, raw_entities, structured_data, hospital, patient_root, raw_text, warnings, or arbitrary wrapper fields.\n"
        "- Preserve raw OCR wording. Do not normalize, translate, expand abbreviations, or infer hidden content.\n"
        "- If a scalar is absent, use null. If a list has no visible entries, use [].\n"
        "- Every list item must include page_number when possible. Use 1 if page is unknown.\n\n"
        f"Document ID: {doc['document_id']}\n"
        f"Patient root: {doc.get('patient_id', '')}\n"
        f"Hospital hint: {doc.get('hospital_name', '')}\n"
        f"Department hint: {doc.get('department_inferred', '')}\n"
        f"OCR engine: {engine}\n\n"
        "Minimal skeleton to fill. Keep these exact top-level keys:\n"
        f"{json.dumps(skeleton, ensure_ascii=False, indent=2)}\n\n"
        "OCR TEXT START\n"
        f"{ocr_text[:12000]}\n"
        "OCR TEXT END\n"
    )


def parse_lane(spec: str) -> tuple[str, str, str]:
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError("--lane must be lane_name|ollama_model|ocr_engine")
    return parts[0], parts[1], parts[2]


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest = {row["document_id"]: row for row in read_csv(Path(args.manifest))}
        self.handoff = read_csv(Path(args.handoff))
        self.lanes = [parse_lane(spec) for spec in args.lane]
        self.smoke_docs = set(args.smoke_docs.split(",")) if args.smoke_docs else set()
        self.pid = os.getpid()
        self.current = ""

    def row_engine(self, row: dict[str, str]) -> str:
        return row.get("ocr_engine") or row.get("engine") or ""

    def row_status_ok(self, row: dict[str, str]) -> bool:
        return (row.get("status") or "").lower() in {"available", "ok", "success"}

    def row_path(self, row: dict[str, str]) -> Path:
        return Path(row.get("ocr_text_path") or row.get("raw_text_path") or "")

    def matching_rows(self, engine: str) -> list[dict[str, str]]:
        rows = []
        for row in self.handoff:
            if self.row_engine(row) != engine or not self.row_status_ok(row):
                continue
            if row.get("document_id") not in self.manifest:
                continue
            if self.smoke_docs and row.get("document_id") not in self.smoke_docs:
                continue
            rows.append(row)
        rows.sort(key=lambda row: row["document_id"])
        if self.smoke_docs:
            order = {doc_id: idx for idx, doc_id in enumerate(self.args.smoke_docs.split(","))}
            rows.sort(key=lambda row: order.get(row["document_id"], 999))
        return rows

    def paths(self, lane: str, doc_id: str) -> dict[str, Path]:
        return {
            "parsed": self.output_root / "raw_structured" / lane / f"{doc_id}.json",
            "raw": self.output_root / "raw_responses" / lane / f"{doc_id}.txt",
            "failed": self.output_root / "failed_cases" / lane / f"{doc_id}.json",
            "log": self.output_root / "logs" / lane / f"{doc_id}.json",
        }

    def run_one(self, lane: str, model: str, engine: str, row: dict[str, str]) -> None:
        doc_id = row["document_id"]
        self.current = f"{lane}:{doc_id}"
        p = self.paths(lane, doc_id)
        if self.args.resume and p["log"].exists():
            try:
                if json.loads(p["log"].read_text(encoding="utf-8")).get("schema_validation_success"):
                    return
            except Exception:
                pass
        ocr_path = self.row_path(row)
        ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
        prompt = canonical_prompt(self.manifest[doc_id], ocr_text, engine)
        self.write_progress()
        started = time.time()
        ok, content, raw, error = ollama_chat(model, prompt, self.args.timeout)
        runtime = round(time.time() - started, 3)
        write_text(p["raw"], content or json.dumps(raw, ensure_ascii=False))
        parsed = extract_json(content)
        valid, reason = validate_canonical(parsed, doc_id)
        if valid and parsed:
            if parsed.get("metadata"):
                parsed["metadata"]["model_name"] = model
                parsed["metadata"]["backend_name"] = lane
                parsed["metadata"]["processing_time_ms"] = runtime * 1000.0
            write_json(p["parsed"], parsed)
        else:
            write_json(
                p["failed"],
                {
                    "error": error or reason or "schema_invalid_wrong_json_shape",
                    "failure_label": "schema_invalid_wrong_json_shape" if parsed else "json_parse_failed",
                    "parsed_keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
                    "content": content,
                },
            )
            append_jsonl(
                self.output_root / "checkpoints" / "failures.jsonl",
                {"timestamp": now(), "lane": lane, "document_id": doc_id, "error": error or reason},
            )
        log = {
            "timestamp": now(),
            "lane": lane,
            "model": model,
            "ocr_engine": engine,
            "document_id": doc_id,
            "ocr_text_path": str(ocr_path),
            "ocr_text_exists": ocr_path.exists(),
            "runtime_seconds": runtime,
            "parse_success": parsed is not None,
            "schema_validation_success": valid,
            "failure_label": "" if valid else ("schema_invalid_wrong_json_shape" if parsed else "json_parse_failed"),
            "error": "" if valid else (error or reason),
            "raw_response_path": str(p["raw"]),
            "parsed_response_path": str(p["parsed"]) if valid else "",
            "usage": {"prompt_eval_count": raw.get("prompt_eval_count"), "eval_count": raw.get("eval_count")} if raw else {},
        }
        write_json(p["log"], log)
        append_jsonl(
            self.output_root / "checkpoints" / "ocr_to_json.jsonl",
            {"timestamp": now(), "lane": lane, "document_id": doc_id, "status": "success" if valid else "failed", "runtime_seconds": runtime},
        )
        self.write_progress()

    def counts(self) -> dict[str, dict[str, int]]:
        out = {}
        for lane, _, engine in self.lanes:
            logs = list((self.output_root / "logs" / lane).glob("*.json"))
            success = failed = 0
            for path in logs:
                try:
                    log = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                success += int(log.get("schema_validation_success") is True)
                failed += int(log.get("schema_validation_success") is False)
            out[lane] = {"completed": success, "failed": failed, "total": len(self.matching_rows(engine))}
        return out

    def write_progress(self) -> None:
        progress = {"generated": now(), "pid": self.pid, "output_root": str(self.output_root), "current": self.current, "lanes": self.counts()}
        write_json(REPORTS_DIR / f"{self.args.report_stem}_progress.json", progress)
        lines = [f"# {self.args.report_title} Progress", "", f"Generated: {progress['generated']}", "", f"- Output root: `{self.output_root}`", f"- PID: `{self.pid}`", f"- Current: `{self.current}`", ""]
        for lane, counts in progress["lanes"].items():
            lines.append(f"- `{lane}`: completed={counts['completed']} failed={counts['failed']} total={counts['total']}")
        write_text(REPORTS_DIR / f"{self.args.report_stem}_progress.md", "\n".join(lines) + "\n")

    def run(self) -> None:
        for lane, model, engine in self.lanes:
            for row in self.matching_rows(engine):
                self.run_one(lane, model, engine, row)
        self.current = ""
        self.write_progress()
        self.write_summary()

    def write_summary(self) -> None:
        rows = []
        for lane, model, engine in self.lanes:
            for row in self.matching_rows(engine):
                doc_id = row["document_id"]
                log_path = self.output_root / "logs" / lane / f"{doc_id}.json"
                log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
                rows.append(
                    {
                        "lane": lane,
                        "model": model,
                        "ocr_engine": engine,
                        "document_id": doc_id,
                        "parse_success": int(bool(log.get("parse_success"))),
                        "schema_valid": int(bool(log.get("schema_validation_success"))),
                        "failure_label": log.get("failure_label", ""),
                        "runtime_seconds": log.get("runtime_seconds", ""),
                        "ocr_text_path": log.get("ocr_text_path", self.row_path(row)),
                        "parsed_response_path": log.get("parsed_response_path", ""),
                    }
                )
        fields = ["lane", "model", "ocr_engine", "document_id", "parse_success", "schema_valid", "failure_label", "runtime_seconds", "ocr_text_path", "parsed_response_path"]
        write_csv(REPORTS_DIR / f"{self.args.report_stem}_results.csv", rows, fields)
        lines = [f"# {self.args.report_title} Summary", "", f"Generated: {now()}", "", f"- Output root: `{self.output_root}`", ""]
        for lane, _, _ in self.lanes:
            lane_rows = [row for row in rows if row["lane"] == lane]
            valid = sum(int(row["schema_valid"]) for row in lane_rows)
            attempted = len(lane_rows)
            label_counts: dict[str, int] = {}
            for row in lane_rows:
                label = row["failure_label"] or "schema_valid"
                label_counts[label] = label_counts.get(label, 0) + 1
            lines.append(f"- `{lane}`: schema-valid {valid}/{attempted}; labels={label_counts}")
        write_text(REPORTS_DIR / f"{self.args.report_stem}_summary.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical OCR-to-JSON from handoff CSV")
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lane", action="append", required=True, help="lane_name|ollama_model|ocr_engine")
    parser.add_argument("--smoke-docs", default="")
    parser.add_argument("--report-stem", required=True)
    parser.add_argument("--report-title", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    Runner(parse_args()).run()


if __name__ == "__main__":
    main()
