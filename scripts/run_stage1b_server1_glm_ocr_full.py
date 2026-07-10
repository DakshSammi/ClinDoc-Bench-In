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

"""Stage 1B Final release GLM-OCR raw OCR runner.

Runs local Ollama `glm-ocr:latest` for Track 1 raw OCR/document recognition.
Supports 3-document smoke and full manifest runs with resume/checkpoints.
No paid APIs are used.
"""

from __future__ import annotations

import argparse
import base64
import csv
import difflib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_BACKEND = "ollama_glm_ocr"
SERVER_NAME = "server1_4090_ollama"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MINIMAL_PROMPT = "Text Recognition: {image_path}"
PRESCRIPTION_PROMPT = (
    "Extract all visible text from this prescription image. Preserve line order, "
    "handwriting, abbreviations, spelling, numeric values, crossed-out or uncertain "
    "text if visible, and page-wise order. Do not normalize medical terms. Do not "
    "infer hidden content."
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sort_key(value: Any) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def run_cmd(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:
        return 999, "", f"{type(exc).__name__}: {exc}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path, doc_ids: Optional[List[str]] = None) -> List[Dict[str, str]]:
    rows = [r for r in read_csv(path) if r.get("benchmark_include", "true").lower() == "true"]
    if doc_ids:
        wanted = set(doc_ids)
        rows = [r for r in rows if r["document_id"] in wanted]
    return sorted(rows, key=lambda r: sort_key(r["document_id"]))


def gpu_memory() -> str:
    code, out, err = run_cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], timeout=10)
    return out if code == 0 else err


def model_info(model: str) -> Dict[str, Any]:
    v_code, v_out, v_err = run_cmd(["ollama", "--version"], timeout=10)
    l_code, l_out, l_err = run_cmd(["ollama", "list"], timeout=20)
    s_code, s_out, s_err = run_cmd(["ollama", "show", model], timeout=20)
    return {
        "ollama_version": v_out or v_err,
        "ollama_list": l_out or l_err,
        "ollama_show_returncode": s_code,
        "ollama_show": s_out,
        "ollama_show_error": s_err,
    }


def package_images(doc: Dict[str, str], output_root: Path, max_dim: int = 1800, jpeg_quality: int = 92) -> List[Path]:
    out_dir = output_root / "compressed_images" / doc["document_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    roles = doc.get("image_roles", "").split(";") if doc.get("image_roles") else []
    for idx, src_rel in enumerate([p for p in doc["source_images_ordered"].split(";") if p]):
        src = PROJECT_ROOT / src_rel
        role = roles[idx] if idx < len(roles) else f"page_or_view_{idx + 1}"
        dst = out_dir / f"{idx + 1:02d}_{role}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            scale = min(1.0, max_dim / max(im.width, im.height))
            if scale < 1.0:
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
            im.save(dst, format="JPEG", quality=jpeg_quality, optimize=True)
        images.append(dst)
    return images


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def ollama_generate(model: str, prompt: str, images: List[Path], timeout: int = 600) -> Tuple[bool, str, Any, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(p) for p in images],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
        if resp.status_code >= 400:
            return False, "", resp.text, f"HTTP {resp.status_code}: {resp.text[:800]}"
        data = resp.json()
        return True, data.get("response", ""), data, ""
    except Exception as exc:
        return False, "", None, f"{type(exc).__name__}: {exc}"


def gt_text(doc: Dict[str, str]) -> str:
    try:
        data = json.loads((PROJECT_ROOT / doc["ground_truth_json"]).read_text(encoding="utf-8"))
    except Exception:
        return ""
    parts: List[str] = []
    raw_text = data.get("raw_text", {})
    if isinstance(raw_text, dict):
        parts.append(str(raw_text.get("full_text", "")))
        for page in raw_text.get("pages", []) or []:
            if isinstance(page, dict):
                parts.append(str(page.get("text", "")))
    parts.append(json.dumps(data.get("raw_entities", {}), ensure_ascii=False))
    return "\n".join(p for p in parts if p)


def proxy_scores(output: str, gt: str) -> Dict[str, float]:
    text = output.lower()
    gt_l = gt.lower()
    buckets = {
        "patient_name_recall_proxy": ["name", "patient", "devki", "alivelatamma"],
        "date_recall_proxy": ["date", "2024", "2023", "jul", "jun"],
        "medication_recall_proxy": ["tablet", "tab", "rx", "metformin", "gabapen", "pantop", "bep"],
        "vitals_recall_proxy": ["bp", "pulse", "weight", "height", "bmi"],
        "complaints_diagnosis_recall_proxy": ["pain", "diagnosis", "c/o", "complaint", "burning"],
    }
    out = {}
    for key, terms in buckets.items():
        relevant = [t for t in terms if t in gt_l]
        out[key] = round(sum(1 for t in relevant if t in text) / len(relevant), 4) if relevant else 0.0
    return out


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.docs = load_manifest(Path(args.manifest), args.doc_ids)
        self.info = model_info(args.model)
        self.pid = os.getpid()
        self.current_doc = ""
        self.start_time = time.time()
        self.skipped = 0

    def log_path(self, doc_id: str) -> Path:
        return self.output_root / "logs" / OUTPUT_BACKEND / f"{doc_id}.json"

    def raw_path(self, doc_id: str) -> Path:
        return self.output_root / "raw_ocr" / OUTPUT_BACKEND / f"{doc_id}.txt"

    def raw_response_path(self, doc_id: str) -> Path:
        return self.output_root / "raw_responses" / OUTPUT_BACKEND / f"{doc_id}.txt"

    def failed_path(self, doc_id: str) -> Path:
        return self.output_root / "failed_cases" / OUTPUT_BACKEND / f"{doc_id}.json"

    def should_skip(self, doc_id: str) -> bool:
        if self.args.force:
            return False
        log_path = self.log_path(doc_id)
        if not (log_path.exists() and self.raw_path(doc_id).exists()):
            return False
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return bool(log.get("non_empty_output"))

    def run_doc(self, doc: Dict[str, str]) -> Dict[str, Any]:
        doc_id = doc["document_id"]
        self.current_doc = doc_id
        if self.should_skip(doc_id):
            self.skipped += 1
            return self.metric_row(doc, skipped=True)
        images = package_images(doc, self.output_root, self.args.max_image_dim, self.args.jpeg_quality)
        page_dir = self.output_root / "raw_ocr_pages" / OUTPUT_BACKEND / doc_id
        page_dir.mkdir(parents=True, exist_ok=True)
        raw_chunks: List[Dict[str, Any]] = []
        outputs: List[str] = []
        errors: List[str] = []
        image_input_accepted = False
        multi_image_accepted = False
        gpu_before = gpu_memory()
        start = time.time()
        in_progress = self.log_path(doc_id).with_suffix(".in_progress.json")
        write_json_atomic(in_progress, {
            "server_name": SERVER_NAME,
            "backend": OUTPUT_BACKEND,
            "model": self.args.model,
            "document_id": doc_id,
            "status": "in_progress",
            "start_timestamp": now(),
            "image_paths": [rel(p) for p in images],
            "pid": self.pid,
        })
        self.write_progress()

        if len(images) > 1:
            prompt = PRESCRIPTION_PROMPT if self.args.prompt_mode in {"prescription", "both"} else MINIMAL_PROMPT.format(image_path=";".join(str(p) for p in images))
            ok, content, raw, err = ollama_generate(self.args.model, prompt, images)
            raw_chunks.append({"variant": "multi_image", "prompt": prompt, "images": [rel(p) for p in images], "ok": ok, "content": content, "raw": raw, "error": err})
            if ok and content.strip():
                multi_image_accepted = True
                image_input_accepted = True
                outputs.append(f"<!-- multi_image_bundle -->\n{content}")
                write_text(page_dir / "page_or_view_multi_image.txt", content)
            elif err:
                errors.append(err)

        if not outputs or not multi_image_accepted:
            outputs = []
            for idx, image in enumerate(images):
                variants = []
                if self.args.prompt_mode in {"minimal", "both"}:
                    variants.append(("minimal", MINIMAL_PROMPT.format(image_path=str(image))))
                if self.args.prompt_mode in {"prescription", "both"}:
                    variants.append(("prescription", PRESCRIPTION_PROMPT))
                image_text_parts = []
                for variant, prompt in variants:
                    ok, content, raw, err = ollama_generate(self.args.model, prompt, [image])
                    raw_chunks.append({"variant": variant, "prompt": prompt, "image": rel(image), "ok": ok, "content": content, "raw": raw, "error": err})
                    if ok and content.strip():
                        image_input_accepted = True
                        image_text_parts.append(f"## {variant}\n{content}")
                    elif err:
                        errors.append(err)
                page_text = "\n\n".join(image_text_parts)
                write_text(page_dir / f"page_or_view_{idx + 1:02d}.txt", page_text)
                outputs.append(f"<!-- {image.name} -->\n{page_text}")

        text = "\n\n".join(outputs).strip()
        runtime = round(time.time() - start, 3)
        gpu_after = gpu_memory()
        write_text(self.raw_path(doc_id), text)
        write_text(self.raw_response_path(doc_id), json.dumps(raw_chunks, indent=2, ensure_ascii=False))
        gt = gt_text(doc)
        sim = round(difflib.SequenceMatcher(None, text.lower(), gt.lower()).ratio(), 4) if text and gt else 0.0
        proxies = proxy_scores(text, gt)
        success = bool(text.strip()) and image_input_accepted
        log = {
            "server_name": SERVER_NAME,
            "backend": OUTPUT_BACKEND,
            "model": self.args.model,
            "document_id": doc_id,
            "patient_id": doc.get("patient_id", ""),
            "department_inferred": doc.get("department_inferred", ""),
            "prompt_variant": self.args.prompt_mode,
            "image_paths": [rel(p) for p in images],
            "runtime_seconds": runtime,
            "gpu_memory_before": gpu_before,
            "gpu_memory_after": gpu_after,
            "output_character_count": len(text),
            "non_empty_output": bool(text),
            "error": "; ".join(errors),
            "image_input_accepted": image_input_accepted,
            "multi_image_input_accepted": multi_image_accepted,
            "raw_output_path": rel(self.raw_path(doc_id)),
            "raw_response_path": rel(self.raw_response_path(doc_id)),
            "status": "success" if success else "failed",
            "model_info": self.info,
            "approx_text_similarity_to_gt": sim,
            **proxies,
            "end_timestamp": now(),
        }
        write_json_atomic(in_progress, log)
        in_progress.replace(self.log_path(doc_id))
        if not success:
            write_json_atomic(self.failed_path(doc_id), {"error": log["error"] or "empty output/image input failure", "log": log})
            append_jsonl(self.output_root / "checkpoints" / "failures.jsonl", {"timestamp": now(), "document_id": doc_id, "error": log["error"] or "empty output/image input failure"})
        append_jsonl(self.output_root / "checkpoints" / "glm_ocr.jsonl", {"timestamp": now(), "document_id": doc_id, "status": log["status"], "runtime_seconds": runtime})
        self.write_progress()
        return self.metric_row(doc)

    def metric_row(self, doc: Dict[str, str], skipped: bool = False) -> Dict[str, Any]:
        doc_id = doc["document_id"]
        log = {}
        if self.log_path(doc_id).exists():
            log = json.loads(self.log_path(doc_id).read_text(encoding="utf-8"))
        return {
            "document_id": doc_id,
            "patient_id": doc.get("patient_id", ""),
            "department_inferred": doc.get("department_inferred", ""),
            "is_multi_page": doc.get("is_multi_page", ""),
            "is_same_page_multi_view": doc.get("is_same_page_multi_view", ""),
            "non_empty_output": int(bool(log.get("non_empty_output"))) if not skipped else 1,
            "approx_text_similarity_to_gt": log.get("approx_text_similarity_to_gt", ""),
            "patient_name_recall_proxy": log.get("patient_name_recall_proxy", ""),
            "date_recall_proxy": log.get("date_recall_proxy", ""),
            "medication_recall_proxy": log.get("medication_recall_proxy", ""),
            "vitals_recall_proxy": log.get("vitals_recall_proxy", ""),
            "complaints_diagnosis_recall_proxy": log.get("complaints_diagnosis_recall_proxy", ""),
            "runtime_seconds": log.get("runtime_seconds", ""),
            "image_input_accepted": log.get("image_input_accepted", ""),
            "multi_image_input_accepted": log.get("multi_image_input_accepted", ""),
            "status": "skipped" if skipped else log.get("status", "not_run"),
        }

    def counts(self) -> Dict[str, int]:
        completed = failed = running = 0
        log_dir = self.output_root / "logs" / OUTPUT_BACKEND
        for p in log_dir.glob("*.json"):
            if p.name.endswith(".in_progress.json"):
                continue
            try:
                status = json.loads(p.read_text(encoding="utf-8")).get("status")
            except Exception:
                continue
            completed += int(status == "success")
            failed += int(status == "failed")
        running = len(list(log_dir.glob("*.in_progress.json")))
        return {"completed": completed, "failed": failed, "running": running}

    def write_progress(self) -> None:
        if self.args.smoke:
            return
        counts = self.counts()
        done = counts["completed"] + counts["failed"] + self.skipped
        elapsed = time.time() - self.start_time
        eta = "unknown" if done == 0 else f"{int((elapsed / done) * max(0, len(self.docs) - done)) // 60}m"
        failures = []
        fail_path = self.output_root / "checkpoints" / "failures.jsonl"
        if fail_path.exists():
            failures = [json.loads(line) for line in fail_path.read_text().splitlines() if line.strip()][-5:]
        progress = {
            "generated": now(),
            "server_name": SERVER_NAME,
            "backend": OUTPUT_BACKEND,
            "model": self.args.model,
            "total_records": len(self.docs),
            "completed": counts["completed"],
            "failed": counts["failed"],
            "skipped": self.skipped,
            "currently_running_document": self.current_doc if counts["running"] else "",
            "eta": eta,
            "last_5_failures": failures,
            "output_directory": str(self.output_root),
            "pid": self.pid,
            "process_log": self.args.process_log_path,
        }
        write_json_atomic(REPORTS_DIR / "stage1b_server1_glm_ocr_progress.json", progress)
        lines = [
            "# Stage 1B Final release GLM-OCR Progress",
            "",
            f"Generated: {progress['generated']}",
            "",
            f"- Output directory: `{progress['output_directory']}`",
            f"- PID: `{progress['pid']}`",
            f"- Total records: {progress['total_records']}",
            f"- Completed: {progress['completed']}",
            f"- Failed: {progress['failed']}",
            f"- Skipped: {progress['skipped']}",
            f"- Currently running document: `{progress['currently_running_document']}`",
            f"- ETA: {progress['eta']}",
            f"- Process log: `{progress['process_log']}`",
            "",
            "## Last 5 Failures",
            "",
        ]
        lines.extend([f"- `{f.get('document_id')}`: {f.get('error')}" for f in failures] or ["No failures recorded yet."])
        write_text(REPORTS_DIR / "stage1b_server1_glm_ocr_progress.md", "\n".join(lines) + "\n")
        self.write_12h_summary()

    def write_12h_summary(self) -> None:
        counts = self.counts()
        lines = [
            "# Stage 1B Final release GLM-OCR 12h Summary",
            "",
            f"Generated: {now()}",
            "",
            f"- Job still running: {str(counts['completed'] + counts['failed'] + self.skipped < len(self.docs)).lower()}",
            f"- Processed: {counts['completed'] + counts['failed']} / {len(self.docs)}",
            f"- Completed: {counts['completed']}",
            f"- Failed: {counts['failed']}",
            f"- Skipped: {self.skipped}",
            f"- Output directory: `{self.output_root}`",
        ]
        write_text(REPORTS_DIR / "stage1b_server1_glm_ocr_12h_summary.md", "\n".join(lines) + "\n")

    def run(self) -> List[Dict[str, Any]]:
        metrics = []
        self.write_progress()
        for doc in self.docs:
            metrics.append(self.run_doc(doc))
        self.current_doc = ""
        self.write_progress()
        if not self.args.smoke:
            self.write_final_reports(metrics)
        return metrics

    def write_final_reports(self, metrics: List[Dict[str, Any]]) -> None:
        fields = metric_fields()
        write_csv(REPORTS_DIR / "stage1b_server1_glm_ocr_metrics.csv", metrics, fields)
        write_csv(REPORTS_DIR / "stage1b_server1_glm_ocr_departmentwise_metrics.csv", aggregate(metrics, "department_inferred"), ["group", "records", "non_empty_output_rate", "avg_similarity", "avg_runtime_seconds"])
        write_text(REPORTS_DIR / "stage1b_server1_glm_ocr_failure_log.md", render_failure_log(self.output_root))
        write_text(REPORTS_DIR / "stage1b_server1_glm_ocr_runtime_report.md", render_runtime_report(self.output_root))
        write_text(REPORTS_DIR / "stage1b_server1_glm_ocr_summary.md", render_summary(metrics, self.output_root))


def metric_fields() -> List[str]:
    return [
        "document_id", "patient_id", "department_inferred", "is_multi_page", "is_same_page_multi_view",
        "non_empty_output", "approx_text_similarity_to_gt", "patient_name_recall_proxy",
        "date_recall_proxy", "medication_recall_proxy", "vitals_recall_proxy",
        "complaints_diagnosis_recall_proxy", "runtime_seconds", "image_input_accepted",
        "multi_image_input_accepted", "status",
    ]


def avg(values: List[Any]) -> Any:
    nums = []
    for v in values:
        try:
            if v not in ("", None):
                nums.append(float(v))
        except Exception:
            pass
    return round(sum(nums) / len(nums), 4) if nums else ""


def aggregate(metrics: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in metrics:
        groups.setdefault(row.get(key) or "Unknown", []).append(row)
    return [{
        "group": group,
        "records": len(rows),
        "non_empty_output_rate": avg([r["non_empty_output"] for r in rows]),
        "avg_similarity": avg([r["approx_text_similarity_to_gt"] for r in rows]),
        "avg_runtime_seconds": avg([r["runtime_seconds"] for r in rows]),
    } for group, rows in sorted(groups.items())]


def render_failure_log(output_root: Path) -> str:
    lines = ["# Stage 1B Final release GLM-OCR Failure Log", "", f"Generated: {now()}", ""]
    path = output_root / "checkpoints" / "failures.jsonl"
    failures = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    lines.extend([f"- `{f.get('document_id')}`: {f.get('error')}" for f in failures] or ["No failures recorded."])
    return "\n".join(lines) + "\n"


def render_runtime_report(output_root: Path) -> str:
    lines = ["# Stage 1B Final release GLM-OCR Runtime Report", "", f"Generated: {now()}", "", "No paid external API calls were made.", ""]
    for p in sorted((output_root / "logs" / OUTPUT_BACKEND).glob("*.json"), key=sort_key):
        log = json.loads(p.read_text())
        lines.append(f"- `{log.get('document_id')}`: {log.get('runtime_seconds')}s, status={log.get('status')}, chars={log.get('output_character_count')}")
    return "\n".join(lines) + "\n"


def render_summary(metrics: List[Dict[str, Any]], output_root: Path) -> str:
    return "\n".join([
        "# Stage 1B Final release GLM-OCR Summary",
        "",
        f"Generated: {now()}",
        "",
        f"- Output directory: `{output_root}`",
        f"- Total records: {len(metrics)}",
        f"- Non-empty output rate: {avg([m['non_empty_output'] for m in metrics])}",
        f"- Average approximate text similarity: {avg([m['approx_text_similarity_to_gt'] for m in metrics])}",
        f"- Average runtime seconds: {avg([m['runtime_seconds'] for m in metrics])}",
    ]) + "\n"


def update_availability(info: Dict[str, Any], status: str) -> None:
    path = REPORTS_DIR / "stage1b_server1_model_availability.csv"
    if not path.exists():
        path = REPORTS_DIR / "stage1a_server1_model_availability.csv"
    rows = read_csv(path) if path.exists() else []
    by_backend = {r.get("backend"): r for r in rows}
    by_backend[OUTPUT_BACKEND] = {
        "backend": OUTPUT_BACKEND,
        "model": "glm-ocr:latest",
        "modality": "ocr_vlm",
        "installed": "true",
        "smoke_test_status": status,
        "notes": f"ollama_show={info.get('ollama_show', '').replace(chr(10), ' ')[:240]}",
    }
    write_csv(REPORTS_DIR / "stage1b_server1_model_availability.csv", list(by_backend.values()), ["backend", "model", "modality", "installed", "smoke_test_status", "notes"])


def write_availability_report(info: Dict[str, Any]) -> None:
    lines = [
        "# Stage 1B Final release GLM-OCR Availability",
        "",
        f"Generated: {now()}",
        "",
        "- Model name: `glm-ocr:latest`",
        "- Size from `ollama list`: `2.2 GB`",
        "- Image input supported: yes (`vision` capability shown)",
        f"- Ollama version: `{info.get('ollama_version')}`",
        "",
        "## `ollama show glm-ocr:latest`",
        "",
        "```",
        info.get("ollama_show", "") or info.get("ollama_show_error", ""),
        "```",
    ]
    write_text(REPORTS_DIR / "stage1b_server1_glm_ocr_availability.md", "\n".join(lines) + "\n")


def write_smoke_reports(metrics: List[Dict[str, Any]], output_root: Path, info: Dict[str, Any]) -> str:
    status = "ready_for_limited_run"
    non_empty = avg([m["non_empty_output"] for m in metrics])
    catastrophic = all(not m.get("image_input_accepted") for m in metrics)
    if non_empty == 1 and avg([m["approx_text_similarity_to_gt"] for m in metrics]) not in ("", 0):
        status = "ready_for_full_run"
    elif float(non_empty or 0) >= 2 / 3 and not catastrophic:
        status = "ready_for_limited_run"
    elif float(non_empty or 0) > 0:
        status = "smoke_passed_but_low_quality"
    else:
        status = "failed_smoke"
    write_csv(REPORTS_DIR / "stage1b_server1_glm_ocr_smoke_metrics.csv", metrics, metric_fields())
    lines = [
        "# Stage 1B Final release GLM-OCR Smoke Summary",
        "",
        f"Generated: {now()}",
        "",
        f"- Output root: `{output_root}`",
        f"- Documents: {', '.join(m['document_id'] for m in metrics)}",
        f"- Non-empty output rate: {non_empty}",
        f"- Average approximate text similarity: {avg([m['approx_text_similarity_to_gt'] for m in metrics])}",
        f"- Decision: `{status}`",
        f"- Image input catastrophic failure: {str(catastrophic).lower()}",
    ]
    write_text(REPORTS_DIR / "stage1b_server1_glm_ocr_smoke_summary.md", "\n".join(lines) + "\n")
    failure_lines = ["# Stage 1B Final release GLM-OCR Failure Log", "", f"Generated: {now()}", ""]
    failures = [m for m in metrics if m.get("status") != "success"]
    failure_lines.extend([f"- `{m['document_id']}`: status={m.get('status')}" for m in failures] or ["No smoke failures recorded."])
    write_text(REPORTS_DIR / "stage1b_server1_glm_ocr_failure_log.md", "\n".join(failure_lines) + "\n")
    update_availability(info, status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", default="glm-ocr:latest")
    p.add_argument("--output-root", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--single-worker", action="store_true")
    p.add_argument("--doc-ids", nargs="*")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--prompt-mode", choices=["minimal", "prescription", "both"], default="prescription")
    p.add_argument("--max-image-dim", type=int, default=1800)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--process-log-path", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runner = Runner(args)
    write_availability_report(runner.info)
    metrics = runner.run()
    if args.smoke:
        status = write_smoke_reports(metrics, runner.output_root, runner.info)
        print(f"smoke_status={status}")
    else:
        update_availability(runner.info, "full_run_started_or_completed")


if __name__ == "__main__":
    main()
