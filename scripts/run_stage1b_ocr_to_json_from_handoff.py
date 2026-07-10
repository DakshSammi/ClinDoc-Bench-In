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

"""Stage 1B OCR-to-JSON runner from OCR handoff CSV.

Runs local Ollama text models over OCR text and writes canonical JSON outputs
with one local JSON repair pass. No paid APIs are used.
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
from typing import Any, Dict, List, Optional, Tuple

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
PROMPT_PATH = PROJECT_ROOT / "prompts" / "ocr_to_json_structuring_prompt.txt"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    candidates = [text.strip()]
    if "```json" in text:
        candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
    if "```" in text:
        candidates.append(text.split("```", 1)[1].split("```", 1)[0].strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    repaired = []
    for c in candidates:
        repaired.extend([c, re.sub(r",\s*([}\]])", r"\1", c)])
    for c in repaired:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def schema_valid(obj: Optional[Dict[str, Any]]) -> bool:
    if not obj:
        return False
    if "document_metadata" in obj and "raw_entities" in obj:
        return True
    if obj.get("schema_version") == "raw_rx_v2":
        return True
    return False


def ollama_chat(model: str, prompt: str, timeout: int = 900) -> Tuple[bool, str, Any, Dict[str, Any], str]:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.0}}
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


def build_prompt(doc: Dict[str, str], ocr_text: str) -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "STRICT OUTPUT: Return exactly one valid JSON object. No Markdown fences, no commentary, no trailing commas.\n"
        "Use the canonical manual annotation schema shown in the prompt. Preserve raw OCR wording; do not medically normalize.\n\n"
        f"Document ID: {doc['document_id']}\n"
        f"Patient root: {doc.get('patient_id', '')}\n"
        f"Hospital: {doc.get('hospital_name', '')}\n"
        f"Department: {doc.get('department_inferred', '')}\n"
        f"Source type: {doc.get('source_type', '')}\n\n"
        f"{base}\n\nOCR TEXT START\n{ocr_text}\nOCR TEXT END\n"
    )


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.handoff = read_csv(Path(args.handoff))
        self.manifest = {r["document_id"]: r for r in read_csv(Path(args.manifest))}
        self.lanes = [tuple(item.split(":", 1)) for item in args.lanes]
        self.pid = os.getpid()
        self.current = ""
        self.start = time.time()

    def paths(self, lane: str, doc_id: str) -> Dict[str, Path]:
        return {
            "parsed": self.output_root / "raw_structured" / lane / f"{doc_id}.json",
            "raw": self.output_root / "raw_responses" / lane / f"{doc_id}.txt",
            "log": self.output_root / "logs" / lane / f"{doc_id}.json",
            "failed": self.output_root / "failed_cases" / lane / f"{doc_id}.json",
            "repaired": self.output_root / "repaired" / lane / f"{doc_id}.json",
        }

    def should_skip(self, lane: str, doc_id: str) -> bool:
        p = self.paths(lane, doc_id)
        if not (p["log"].exists() and p["parsed"].exists()):
            return False
        try:
            return bool(json.loads(p["log"].read_text()).get("schema_validation_success"))
        except Exception:
            return False

    def run_one(self, lane: str, model: str, hrow: Dict[str, str]) -> None:
        doc_id = hrow["document_id"]
        self.current = f"{lane}:{doc_id}"
        if self.args.resume and self.should_skip(lane, doc_id):
            return
        p = self.paths(lane, doc_id)
        doc = self.manifest[doc_id]
        ocr_path = Path(hrow["ocr_text_path"])
        ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
        prompt = build_prompt(doc, ocr_text)
        self.write_progress()
        start = time.time()
        ok, content, raw, usage, error = ollama_chat(model, prompt)
        runtime = round(time.time() - start, 3)
        write_text(p["raw"], content or json.dumps(raw, ensure_ascii=False))
        parsed = extract_json(content)
        valid = schema_valid(parsed)
        if parsed:
            write_json(p["repaired"], parsed)
        if valid:
            write_json(p["parsed"], parsed)
        else:
            write_json(p["failed"], {"error": error or "JSON parse/schema validation failed", "content": content})
            append_jsonl(self.output_root / "checkpoints" / "failures.jsonl", {"timestamp": now(), "lane": lane, "document_id": doc_id, "error": error or "JSON parse/schema validation failed"})
        log = {
            "timestamp": now(),
            "lane": lane,
            "model": model,
            "document_id": doc_id,
            "ocr_text_path": str(ocr_path),
            "runtime_seconds": runtime,
            "parse_success": parsed is not None,
            "schema_validation_success": valid,
            "status": "success" if valid else "failed",
            "error": "" if valid else (error or "JSON parse/schema validation failed"),
            "usage": usage,
            "raw_response_path": str(p["raw"]),
            "parsed_response_path": str(p["parsed"]) if valid else "",
            "repaired_response_path": str(p["repaired"]) if parsed else "",
        }
        write_json(p["log"], log)
        append_jsonl(self.output_root / "checkpoints" / "ocr_to_json.jsonl", {"timestamp": now(), "lane": lane, "document_id": doc_id, "status": log["status"], "runtime_seconds": runtime})
        self.write_progress()

    def counts(self) -> Dict[str, Any]:
        out = {}
        for lane, _ in self.lanes:
            logs = list((self.output_root / "logs" / lane).glob("*.json"))
            success = failed = 0
            for p in logs:
                try:
                    status = json.loads(p.read_text()).get("status")
                except Exception:
                    status = ""
                success += int(status == "success")
                failed += int(status == "failed")
            out[lane] = {"completed": success, "failed": failed, "total": len(self.handoff)}
        return out

    def write_progress(self) -> None:
        counts = self.counts()
        progress = {"generated": now(), "output_root": str(self.output_root), "pid": self.pid, "current": self.current, "lanes": counts}
        write_json(REPORTS_DIR / "stage1b_server1_ocr_to_json_glm_ocr_progress.json", progress)
        lines = ["# Stage 1B Final release OCR-to-JSON GLM-OCR Progress", "", f"Generated: {progress['generated']}", "", f"- Output root: `{self.output_root}`", f"- PID: `{self.pid}`", f"- Current: `{self.current}`", ""]
        for lane, c in counts.items():
            lines.append(f"- `{lane}`: completed={c['completed']} failed={c['failed']} total={c['total']}")
        write_text(REPORTS_DIR / "stage1b_server1_ocr_to_json_glm_ocr_progress.md", "\n".join(lines) + "\n")

    def run(self) -> None:
        for lane, model in self.lanes:
            for hrow in self.handoff:
                if hrow.get("status") == "available" and hrow["document_id"] in self.manifest:
                    self.run_one(lane, model, hrow)
        self.current = ""
        self.write_progress()
        self.write_reports()

    def write_reports(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.run_stage1b_server1_full import compute_all_metrics, aggregate_metrics

        all_rows = []
        failures = []
        for lane, _ in self.lanes:
            docs = []
            for h in self.handoff:
                doc = dict(self.manifest[h["document_id"]])
                parsed = self.output_root / "raw_structured" / lane / f"{h['document_id']}.json"
                if parsed.exists():
                    doc["prediction_path_override"] = str(parsed)
                docs.append(doc)
            metrics = []
            for doc in docs:
                doc_id = doc["document_id"]
                parsed = self.output_root / "raw_structured" / lane / f"{doc_id}.json"
                log = self.output_root / "logs" / lane / f"{doc_id}.json"
                base = {
                    "document_id": doc_id, "pipeline": lane, "patient_id": doc.get("patient_id", ""),
                    "department_inferred": doc.get("department_inferred", ""),
                    "schema_validity": 0, "json_parse_success": 0, "runtime_seconds": "", "status": "not_run",
                }
                if log.exists():
                    ldata = json.loads(log.read_text())
                    base.update({"schema_validity": int(ldata.get("schema_validation_success", False)), "json_parse_success": int(ldata.get("parse_success", False)), "runtime_seconds": ldata.get("runtime_seconds", ""), "status": ldata.get("status", "")})
                if parsed.exists():
                    from scripts.run_full_benchmark_stage1 import compute_smoke_metrics
                    m = compute_smoke_metrics(doc, lane, "ocr_to_json", parsed, log, None)
                    base.update(m)
                    base["pipeline"] = lane
                metrics.append(base)
                if base["status"] == "failed":
                    failures.append(base)
            all_rows.extend(metrics)
        fields = sorted({k for r in all_rows for k in r.keys()})
        write_csv(REPORTS_DIR / "stage1b_server1_ocr_to_json_glm_ocr_metrics.csv", all_rows, fields)
        write_text(REPORTS_DIR / "stage1b_server1_ocr_to_json_glm_ocr_failure_log.md", "\n".join(["# OCR-to-JSON GLM-OCR Failure Log", "", *[f"- `{f.get('pipeline')}` `{f.get('document_id')}`: {f.get('notes') or f.get('status')}" for f in failures]]) + "\n")
        summary_lines = ["# Stage 1B Final release OCR-to-JSON GLM-OCR Summary", "", f"- Output root: `{self.output_root}`", ""]
        for lane, _ in self.lanes:
            rows = [r for r in all_rows if r.get("pipeline") == lane]
            valid = sum(1 for r in rows if str(r.get("schema_validity")) == "1")
            summary_lines.append(f"- `{lane}`: schema-valid {valid}/{len(rows)}")
        write_text(REPORTS_DIR / "stage1b_server1_ocr_to_json_glm_ocr_summary.md", "\n".join(summary_lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--lanes", nargs="+", required=True, help="lane:model pairs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--single-worker", action="store_true")
    args = ap.parse_args()
    Runner(args).run()


if __name__ == "__main__":
    main()
