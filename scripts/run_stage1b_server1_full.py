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

"""Stage 1B Server 1 full local Ollama VLM runner.

Approved scope: full raw structured extraction using qwen3-vl:8b-instruct only.
This runner is resume-safe, single-worker, and writes progress/checkpoints as it
goes. External APIs and unapproved Ollama models are not used here.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAKSH_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DAKSH_ROOT.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
SERVER_NAME = "server1_4090_ollama"
APPROVED_BACKEND = "ollama_qwen3_vl_8b_raw_structured"
OUTPUT_BACKEND = "ollama_qwen3_vl_8b"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def sort_key(value: Any) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_prompt() -> Tuple[str, str]:
    path = PROMPTS_DIR / "raw_structured_extraction_prompt.txt"
    text = path.read_text(encoding="utf-8")
    return text, sha256_text(text)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("benchmark_include", "true").lower() == "true"]
    return sorted(rows, key=lambda r: sort_key(r["document_id"]))


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def gpu_memory() -> str:
    try:
        import subprocess

        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def compress_images(doc: Dict[str, str], run_dir: Path, max_dim: int, jpeg_quality: int) -> Tuple[List[Path], List[Dict[str, Any]]]:
    out_dir = run_dir / "compressed_images" / doc["document_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [PROJECT_ROOT / p for p in doc["source_images_ordered"].split(";") if p]
    roles = doc.get("image_roles", "").split(";") if doc.get("image_roles") else []
    out_paths: List[Path] = []
    logs: List[Dict[str, Any]] = []
    for idx, src in enumerate(source_paths):
        role = roles[idx] if idx < len(roles) else f"image_{idx + 1}"
        dst = out_dir / f"{idx + 1:02d}_{role}.jpg"
        with Image.open(src) as im:
            original_size = im.size
            im = im.convert("RGB")
            scale = min(1.0, float(max_dim) / max(im.width, im.height))
            if scale < 1.0:
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
            compressed_size = im.size
            im.save(dst, format="JPEG", quality=jpeg_quality, optimize=True)
        out_paths.append(dst)
        logs.append({
            "source": rel(src),
            "compressed": rel(dst),
            "role": role,
            "original_size": list(original_size),
            "compressed_size": list(compressed_size),
            "max_image_dim": max_dim,
            "jpeg_quality": jpeg_quality,
            "bytes": dst.stat().st_size,
        })
    return out_paths, logs


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates = [text.strip()]
    if "```json" in text:
        candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
    if "```" in text:
        candidates.append(text.split("```", 1)[1].split("```", 1)[0].strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def schema_valid(parsed: Optional[Dict[str, Any]]) -> bool:
    if not parsed:
        return False
    if "document_metadata" in parsed and "raw_entities" in parsed:
        return True
    if parsed.get("schema_version") == "raw_rx_v2":
        return True
    return False


def ollama_chat(model: str, prompt: str, images: List[Path], timeout: int = 900) -> Tuple[bool, str, Any, Dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [encode_image(p) for p in images],
        }],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        if resp.status_code >= 400:
            return False, "", resp.text, {}, f"HTTP {resp.status_code}: {resp.text[:800]}"
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        usage = {"prompt_eval_count": data.get("prompt_eval_count"), "eval_count": data.get("eval_count")}
        return True, content, data, usage, ""
    except Exception as exc:
        return False, "", None, {}, f"{type(exc).__name__}: {exc}"


class Runner:
    def __init__(self, args: argparse.Namespace):
        if args.backend != APPROVED_BACKEND:
            raise SystemExit(f"Only approved backend is {APPROVED_BACKEND}; got {args.backend}")
        if args.model != "qwen3-vl:8b-instruct":
            raise SystemExit(f"Only approved model is qwen3-vl:8b-instruct; got {args.model}")
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.docs = load_manifest(Path(args.manifest))
        self.prompt, self.prompt_hash = read_prompt()
        self.start_time = time.time()
        self.current_doc = ""
        self.skipped = 0
        self.pid = os.getpid()
        self.checkpoint_path = self.output_root / "checkpoints" / "stage1b_server1_qwen3vl.jsonl"
        self.failure_events: List[Dict[str, Any]] = []

    def log_path(self, doc_id: str) -> Path:
        return self.output_root / "logs" / OUTPUT_BACKEND / f"{doc_id}.json"

    def parsed_path(self, doc_id: str) -> Path:
        return self.output_root / "raw_structured" / OUTPUT_BACKEND / f"{doc_id}.json"

    def raw_path(self, doc_id: str) -> Path:
        return self.output_root / "raw_responses" / OUTPUT_BACKEND / f"{doc_id}.txt"

    def failed_path(self, doc_id: str) -> Path:
        return self.output_root / "failed_cases" / OUTPUT_BACKEND / f"{doc_id}.json"

    def should_skip(self, doc: Dict[str, str]) -> bool:
        if self.args.force:
            return False
        log_path = self.log_path(doc["document_id"])
        parsed_path = self.parsed_path(doc["document_id"])
        if not (log_path.exists() and parsed_path.exists()):
            return False
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return bool(log.get("schema_validation_success") and log.get("parse_success"))

    def begin_log(self, doc: Dict[str, str], images: List[Path], compression_log: List[Dict[str, Any]]) -> Tuple[Path, Dict[str, Any]]:
        log_path = self.log_path(doc["document_id"])
        in_progress = log_path.with_suffix(".in_progress.json")
        log = {
            "server_name": SERVER_NAME,
            "backend": self.args.backend,
            "output_backend": OUTPUT_BACKEND,
            "model": self.args.model,
            "track": self.args.track,
            "document_id": doc["document_id"],
            "patient_id": doc["patient_id"],
            "department_inferred": doc.get("department_inferred", ""),
            "is_multi_page": doc.get("is_multi_page", ""),
            "is_same_page_multi_view": doc.get("is_same_page_multi_view", ""),
            "prompt_file": "prompts/raw_structured_extraction_prompt.txt",
            "prompt_hash": self.prompt_hash,
            "input_image_paths": doc["source_images_ordered"].split(";"),
            "compressed_image_paths": [rel(p) for p in images],
            "compression_log": compression_log,
            "start_timestamp": now(),
            "status": "in_progress",
            "retry_count": 0,
        }
        write_json_atomic(in_progress, log)
        append_jsonl(self.checkpoint_path, {"timestamp": now(), "event": "start", "document_id": doc["document_id"]})
        return log_path, log

    def finish_log(self, log_path: Path, log: Dict[str, Any], parsed: Optional[Dict[str, Any]], content: str, raw: Any, usage: Dict[str, Any], error: str, start: float) -> None:
        doc_id = log["document_id"]
        raw_path = self.raw_path(doc_id)
        parsed_path = self.parsed_path(doc_id)
        failed_path = self.failed_path(doc_id)
        write_text(raw_path, content or json.dumps(raw, ensure_ascii=False))
        valid = schema_valid(parsed)
        if parsed:
            write_json_atomic(parsed_path, parsed)
        if not valid:
            reason = error or ("JSON parse failed" if not parsed else "schema validation failed")
            write_json_atomic(failed_path, {"error": reason, "content": content, "raw_response": raw})
            event = {"timestamp": now(), "document_id": doc_id, "error": reason}
            self.failure_events.append(event)
            append_jsonl(self.output_root / "checkpoints" / "failures.jsonl", event)
        log.update({
            "end_timestamp": now(),
            "runtime_seconds": round(time.time() - start, 3),
            "status": "success" if valid else "failed",
            "parse_success": parsed is not None,
            "schema_validation_success": valid,
            "token_usage": usage,
            "error_message": "" if valid else (error or ("JSON parse failed" if not parsed else "schema validation failed")),
            "raw_response_path": rel(raw_path),
            "parsed_response_path": rel(parsed_path) if parsed else "",
            "failed_case_path": rel(failed_path) if not valid else "",
            "gpu_memory_after": gpu_memory(),
        })
        in_progress = log_path.with_suffix(".in_progress.json")
        write_json_atomic(in_progress, log)
        in_progress.replace(log_path)
        append_jsonl(self.checkpoint_path, {"timestamp": now(), "event": log["status"], "document_id": doc_id, "runtime_seconds": log["runtime_seconds"]})

    def build_prompt(self, doc: Dict[str, str]) -> str:
        return (
            f"Document ID: {doc['document_id']}\n"
            f"Patient root: {doc['patient_id']}\n"
            f"Hospital: {doc.get('hospital_name', '')}\n"
            f"Department: {doc.get('department_inferred', '')}\n"
            f"Source type: {doc.get('source_type', '')}\n"
            f"Total pages in ground truth: {doc.get('total_pages_gt', '')}\n"
            f"Image bundle type: {doc.get('image_bundle_type', '')}\n"
            f"Image roles: {doc.get('image_roles', '')}\n\n"
            + self.prompt
        )

    def run_doc(self, doc: Dict[str, str]) -> None:
        self.current_doc = doc["document_id"]
        if self.should_skip(doc):
            self.skipped += 1
            append_jsonl(self.checkpoint_path, {"timestamp": now(), "event": "skipped", "document_id": doc["document_id"]})
            self.write_progress()
            return
        images, compression_log = compress_images(doc, self.output_root, self.args.max_image_dim, self.args.jpeg_quality)
        log_path, log = self.begin_log(doc, images, compression_log)
        log["gpu_memory_before"] = gpu_memory()
        write_json_atomic(log_path.with_suffix(".in_progress.json"), log)
        self.write_progress()
        prompt = self.build_prompt(doc)
        start = time.time()
        ok, content, raw, usage, error = ollama_chat(self.args.model, prompt, images)
        if not ok and self.args.resume:
            log["retry_count"] = 1
            time.sleep(5)
            ok, content, raw, usage, error = ollama_chat(self.args.model, prompt, images)
        parsed = extract_json(content)
        self.finish_log(log_path, log, parsed, content, raw, usage, error, start)
        self.write_progress()
        self.write_interim_summary()

    def status_counts(self) -> Dict[str, int]:
        completed = failed = running = 0
        for log_path in (self.output_root / "logs" / OUTPUT_BACKEND).glob("*.json"):
            if log_path.name.endswith(".in_progress.json"):
                continue
            try:
                log = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if log.get("status") == "success":
                completed += 1
            elif log.get("status") == "failed":
                failed += 1
        running = len(list((self.output_root / "logs" / OUTPUT_BACKEND).glob("*.in_progress.json")))
        return {"completed": completed, "failed": failed, "running": running}

    def eta(self, done: int) -> str:
        if done <= 0:
            return "unknown"
        elapsed = time.time() - self.start_time
        per_doc = elapsed / done
        remaining = max(0, len(self.docs) - done - self.skipped)
        seconds = int(per_doc * remaining)
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"

    def last_failures(self) -> List[Dict[str, Any]]:
        failures = []
        path = self.output_root / "checkpoints" / "failures.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        failures.append(json.loads(line))
                    except Exception:
                        pass
        return failures[-5:]

    def write_progress(self) -> None:
        counts = self.status_counts()
        done_for_eta = counts["completed"] + counts["failed"] + self.skipped
        progress = {
            "generated": now(),
            "server_name": SERVER_NAME,
            "backend": self.args.backend,
            "model": self.args.model,
            "total_records": len(self.docs),
            "completed": counts["completed"],
            "failed": counts["failed"],
            "skipped": self.skipped,
            "currently_running_document": self.current_doc if counts["running"] else "",
            "running_count": counts["running"],
            "eta": self.eta(done_for_eta),
            "last_5_failures": self.last_failures(),
            "output_directory": str(self.output_root),
            "pid": self.pid,
            "log_path": self.args.process_log_path or "",
        }
        write_json_atomic(REPORTS_DIR / "stage1b_server1_progress.json", progress)
        lines = [
            "# Stage 1B Server 1 Progress",
            "",
            f"Generated: {progress['generated']}",
            "",
            f"- Output directory: `{progress['output_directory']}`",
            f"- PID: `{progress['pid']}`",
            f"- Backend: `{self.args.backend}`",
            f"- Model: `{self.args.model}`",
            f"- Total records: {progress['total_records']}",
            f"- Completed: {progress['completed']}",
            f"- Failed: {progress['failed']}",
            f"- Skipped: {progress['skipped']}",
            f"- Currently running document: `{progress['currently_running_document']}`",
            f"- ETA: {progress['eta']}",
            f"- Process log: `{progress['log_path']}`",
            "",
            "## Last 5 Failures",
            "",
        ]
        if progress["last_5_failures"]:
            lines.extend([f"- `{f.get('document_id')}`: {f.get('error')}" for f in progress["last_5_failures"]])
        else:
            lines.append("No failures recorded yet.")
        write_text(REPORTS_DIR / "stage1b_server1_progress.md", "\n".join(lines) + "\n")

    def write_interim_summary(self) -> None:
        counts = self.status_counts()
        lines = [
            "# Stage 1B Server 1 12h Interim Summary",
            "",
            f"Generated: {now()}",
            "",
            f"- Job still running: {str(counts['completed'] + counts['failed'] + self.skipped < len(self.docs)).lower()}",
            f"- Documents processed: {counts['completed'] + counts['failed']} / {len(self.docs)}",
            f"- Completed: {counts['completed']}",
            f"- Failed: {counts['failed']}",
            f"- Skipped: {self.skipped}",
            f"- Output directory: `{self.output_root}`",
            "",
            "Preliminary metrics are limited to completed documents and will be finalized after the run.",
            "",
            "## Failures",
            "",
        ]
        failures = self.last_failures()
        lines.extend([f"- `{f.get('document_id')}`: {f.get('error')}" for f in failures] or ["No failures recorded yet."])
        lines.extend(["", "## Next Recommended Action", "", "Let the background run continue unless the progress file stops updating or repeated Ollama failures appear."])
        write_text(REPORTS_DIR / "stage1b_server1_12h_interim_summary.md", "\n".join(lines) + "\n")

    def run(self) -> None:
        self.write_progress()
        self.write_interim_summary()
        for doc in self.docs:
            self.run_doc(doc)
        self.current_doc = ""
        self.write_progress()
        self.write_interim_summary()
        self.write_final_reports()

    def write_final_reports(self) -> None:
        metrics = compute_all_metrics(self.docs, self.output_root)
        fields = [
            "document_id", "patient_id", "department_inferred", "is_multi_page", "is_same_page_multi_view",
            "json_parse_success", "schema_validity", "output_completeness", "field_coverage",
            "scalar_accuracy_exact", "scalar_accuracy_lenient", "entity_exact_f1", "entity_lenient_f1",
            "hallucination_rate", "missing_entity_rate", "runtime_seconds", "status", "notes",
        ]
        write_csv(REPORTS_DIR / "stage1b_server1_qwen3vl_metrics.csv", metrics, fields)
        write_csv(REPORTS_DIR / "stage1b_server1_qwen3vl_departmentwise_metrics.csv", aggregate_metrics(metrics, "department_inferred"), ["group", "records", "parse_success_rate", "schema_validity_rate", "avg_entity_lenient_f1", "avg_hallucination_rate", "avg_missing_entity_rate", "avg_runtime_seconds"])
        write_csv(REPORTS_DIR / "stage1b_server1_qwen3vl_page_type_metrics.csv", aggregate_page_metrics(metrics), ["group", "records", "parse_success_rate", "schema_validity_rate", "avg_entity_lenient_f1", "avg_runtime_seconds"])
        write_text(REPORTS_DIR / "stage1b_server1_qwen3vl_failure_log.md", render_failure_log(self.output_root))
        write_text(REPORTS_DIR / "stage1b_server1_qwen3vl_cost_runtime_report.md", render_runtime_report(self.output_root))
        write_text(REPORTS_DIR / "stage1b_server1_qwen3vl_summary.md", render_summary(metrics, self.output_root))


def compute_all_metrics(docs: List[Dict[str, str]], output_root: Path) -> List[Dict[str, Any]]:
    rows = []
    sys.path.insert(0, str(PROJECT_ROOT))
    for doc in docs:
        doc_id = doc["document_id"]
        log_path = output_root / "logs" / OUTPUT_BACKEND / f"{doc_id}.json"
        parsed_path = output_root / "raw_structured" / OUTPUT_BACKEND / f"{doc_id}.json"
        log = {}
        if log_path.exists():
            try:
                log = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                log = {}
        row = {
            "document_id": doc_id,
            "patient_id": doc.get("patient_id", ""),
            "department_inferred": doc.get("department_inferred", ""),
            "is_multi_page": doc.get("is_multi_page", ""),
            "is_same_page_multi_view": doc.get("is_same_page_multi_view", ""),
            "json_parse_success": int(bool(log.get("parse_success"))),
            "schema_validity": int(bool(log.get("schema_validation_success"))),
            "output_completeness": 0.0,
            "field_coverage": 0.0,
            "scalar_accuracy_exact": "",
            "scalar_accuracy_lenient": "",
            "entity_exact_f1": "",
            "entity_lenient_f1": "",
            "hallucination_rate": "",
            "missing_entity_rate": "",
            "runtime_seconds": log.get("runtime_seconds", ""),
            "status": log.get("status", "not_run"),
            "notes": "",
        }
        if parsed_path.exists():
            try:
                from scripts.run_full_benchmark_stage1 import compute_smoke_metrics

                metric = compute_smoke_metrics(doc, OUTPUT_BACKEND, "raw_structured", parsed_path, log_path, None)
                row.update({
                    "output_completeness": metric.get("output_completeness", 0.0),
                    "field_coverage": metric.get("field_coverage", 0.0),
                    "scalar_accuracy_exact": metric.get("scalar_accuracy_exact", ""),
                    "scalar_accuracy_lenient": metric.get("scalar_accuracy_lenient", ""),
                    "entity_exact_f1": metric.get("entity_exact_f1", ""),
                    "entity_lenient_f1": metric.get("entity_lenient_f1", ""),
                    "hallucination_rate": metric.get("hallucination_rate", ""),
                    "missing_entity_rate": metric.get("missing_entity_rate", ""),
                    "notes": metric.get("notes", ""),
                })
            except Exception as exc:
                row["notes"] = f"Metric calculation failed: {type(exc).__name__}: {exc}"
        elif log.get("error_message"):
            row["notes"] = log.get("error_message")
        rows.append(row)
    return rows


def numeric(values: List[Any]) -> List[float]:
    out = []
    for v in values:
        try:
            if v not in ("", None):
                out.append(float(v))
        except Exception:
            pass
    return out


def avg(values: List[Any]) -> str:
    nums = numeric(values)
    return round(sum(nums) / len(nums), 4) if nums else ""


def aggregate_metrics(metrics: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in metrics:
        groups.setdefault(row.get(key) or "Unknown", []).append(row)
    out = []
    for group, rows in sorted(groups.items()):
        out.append({
            "group": group,
            "records": len(rows),
            "parse_success_rate": avg([r["json_parse_success"] for r in rows]),
            "schema_validity_rate": avg([r["schema_validity"] for r in rows]),
            "avg_entity_lenient_f1": avg([r["entity_lenient_f1"] for r in rows]),
            "avg_hallucination_rate": avg([r["hallucination_rate"] for r in rows]),
            "avg_missing_entity_rate": avg([r["missing_entity_rate"] for r in rows]),
            "avg_runtime_seconds": avg([r["runtime_seconds"] for r in rows]),
        })
    return out


def aggregate_page_metrics(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rows = []
    for name, pred in [
        ("single_page", lambda r: r.get("is_multi_page") == "false" and r.get("is_same_page_multi_view") == "false"),
        ("multi_page", lambda r: r.get("is_multi_page") == "true"),
        ("same_page_multi_view", lambda r: r.get("is_same_page_multi_view") == "true"),
    ]:
        rows = [r for r in metrics if pred(r)]
        out_rows.append({
            "group": name,
            "records": len(rows),
            "parse_success_rate": avg([r["json_parse_success"] for r in rows]),
            "schema_validity_rate": avg([r["schema_validity"] for r in rows]),
            "avg_entity_lenient_f1": avg([r["entity_lenient_f1"] for r in rows]),
            "avg_runtime_seconds": avg([r["runtime_seconds"] for r in rows]),
        })
    return out_rows


def render_failure_log(output_root: Path) -> str:
    lines = ["# Stage 1B Server 1 Qwen3-VL Failure Log", "", f"Generated: {now()}", ""]
    failures = []
    failure_file = output_root / "checkpoints" / "failures.jsonl"
    if failure_file.exists():
        failures = [json.loads(line) for line in failure_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not failures:
        lines.append("No failures recorded.")
    else:
        lines.extend([f"- `{f.get('document_id')}`: {f.get('error')}" for f in failures])
    return "\n".join(lines) + "\n"


def render_runtime_report(output_root: Path) -> str:
    lines = ["# Stage 1B Server 1 Qwen3-VL Cost/Runtime Report", "", f"Generated: {now()}", "", "No paid external API calls were made.", ""]
    for log_path in sorted((output_root / "logs" / OUTPUT_BACKEND).glob("*.json"), key=sort_key):
        log = json.loads(log_path.read_text(encoding="utf-8"))
        lines.append(f"- `{log.get('document_id')}`: {log.get('runtime_seconds')}s, status={log.get('status')}, usage={json.dumps(log.get('token_usage'), ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def render_summary(metrics: List[Dict[str, Any]], output_root: Path) -> str:
    total = len(metrics)
    completed = sum(1 for r in metrics if r["status"] == "success")
    failed = sum(1 for r in metrics if r["status"] == "failed")
    lines = [
        "# Stage 1B Server 1 Qwen3-VL Summary",
        "",
        f"Generated: {now()}",
        "",
        f"- Output directory: `{output_root}`",
        f"- Total records: {total}",
        f"- Successful schema-valid outputs: {completed}",
        f"- Failed outputs: {failed}",
        f"- JSON parse success rate: {avg([r['json_parse_success'] for r in metrics])}",
        f"- Schema validity rate: {avg([r['schema_validity'] for r in metrics])}",
        f"- Average entity lenient F1: {avg([r['entity_lenient_f1'] for r in metrics])}",
        f"- Average hallucination rate: {avg([r['hallucination_rate'] for r in metrics])}",
        f"- Average missing entity rate: {avg([r['missing_entity_rate'] for r in metrics])}",
        f"- Average runtime seconds: {avg([r['runtime_seconds'] for r in metrics])}",
        "",
        "See `stage1b_server1_qwen3vl_metrics.csv`, departmentwise metrics, failure log, and runtime report for details.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1B Server 1 full Qwen3-VL runner")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--track", default="raw_structured")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-image-dim", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--single-worker", action="store_true")
    parser.add_argument("--process-log-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Runner(args).run()


if __name__ == "__main__":
    main()
