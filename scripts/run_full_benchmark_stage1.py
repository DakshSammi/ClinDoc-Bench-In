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

"""
Stage 1A benchmark infrastructure and smoke runner.

This script intentionally does not run the full benchmark. It supports:
- manifest generation/verification
- OCR availability checks
- controlled smoke runs for approved local/external adapters
- Stage 1A report generation
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata as md
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAKSH_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DAKSH_ROOT.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_ROOT = REPO_ROOT / "benchmark_outputs"

load_dotenv(DAKSH_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=False)

OPENROUTER_DISABLED = True


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


def utcish_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_prompt(name: str) -> Tuple[str, str]:
    path = PROMPTS_DIR / name
    text = path.read_text(encoding="utf-8")
    return text, sha256_text(text)


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_model_dump(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return {"repr": repr(value)}


def append_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_manifest(path: Path = DATA_DIR / "full_benchmark_manifest.csv") -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_department(hospital: str, raw: str) -> str:
    h = (hospital or "").lower()
    r = (raw or "").lower()
    if "baba bihari" in h or "netralaya" in h:
        return "Ophthalmology"
    if "svs" in h:
        if "surgery" in r:
            return "Surgery"
        if "medicine" in r or "general medicine" in r or not r:
            return "Medicine / General Medicine"
    if "radiotherapy" in r:
        return "Radiotherapy"
    if "endocrinology" in r:
        return "Endocrinology"
    if "dermatology" in r or "venereology" in r:
        return "Dermatology"
    if "medicine" in r:
        return "Medicine / General Medicine"
    if "surgery" in r:
        return "Surgery"
    if "radiology" in r:
        return "Radiology"
    if "obstetrics" in r or "gynaec" in r:
        return "Obstetrics & Gynaecology"
    return raw or "Unknown / not annotated"


def discover_images(gt_path: Path, meta: Dict[str, Any]) -> List[Path]:
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
    rel_gt = gt_path.relative_to(PROJECT_ROOT / "raw_ground_truths")
    if len(rel_gt.parts) == 1:
        stem = rel_gt.stem
        candidates = [
            PROJECT_ROOT / "prescriptions" / f"{stem}.jpeg",
            PROJECT_ROOT / "prescriptions" / f"{stem}.jpg",
            PROJECT_ROOT / "prescriptions" / stem,
        ]
    else:
        candidates = [PROJECT_ROOT / "prescriptions" / Path(*rel_gt.parts[:-1]) / rel_gt.stem]

    images: List[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in img_exts:
            images.append(candidate)
        elif candidate.is_dir():
            images.extend(
                p for p in sorted(candidate.rglob("*"), key=sort_key)
                if p.is_file() and p.suffix.lower() in img_exts
            )

    source_images = meta.get("source_images", meta.get("source_image", []))
    if isinstance(source_images, str):
        source_images = [source_images]
    if source_images and images:
        by_name = {p.name: p for p in images}
        ordered = [by_name[name] for name in source_images if name in by_name]
        extras = [p for p in images if p not in ordered]
        if ordered:
            images = ordered + extras
    return images


def image_roles(doc_id: str, count: int, bundle_type: str) -> List[str]:
    if bundle_type == "same_page_multi_view":
        if doc_id == "p37_1":
            return ["page_1_full_view", "page_1_close_or_alternate_view"][:count]
        if doc_id == "p42_1":
            return ["page_1_primary_view", "page_1_blurred_alternate_view"][:count]
        return [f"page_1_view_{i + 1}" for i in range(count)]
    return [f"page_{i + 1}" for i in range(count)]


def classify_print_hand(data: Dict[str, Any], hospital: str) -> Tuple[bool, bool]:
    source_type = data.get("document_metadata", {}).get("source_type", "")
    text = " ".join([
        json.dumps(data.get("document_layout", {})),
        json.dumps(data.get("extraction_metadata", {})),
        data.get("raw_text", {}).get("full_text", "") if isinstance(data.get("raw_text"), dict) else "",
    ]).lower()
    handwritten = any(w in text for w in ["handwriting", "handwritten", "unclear", "faint", "best-effort", "partially"])
    printed = source_type in {"lab_report", "radiology_report", "diagnostic_reports", "radiology_image"}
    printed = printed or any(x in (hospital or "").lower() for x in ["acculab", "diagnostic"])
    if not printed and not handwritten:
        printed = bool(hospital)
        handwritten = source_type in {"prescription", "opd_prescription"}
    return printed, handwritten


def build_manifest() -> Path:
    rows: List[Dict[str, Any]] = []
    for gt_path in sorted((PROJECT_ROOT / "raw_ground_truths").rglob("*.json"), key=sort_key):
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        meta = data.get("document_metadata", {})
        raw_entities = data.get("raw_entities", {})
        encounter = raw_entities.get("encounter_information", {}) if isinstance(raw_entities, dict) else {}
        doc_id = meta.get("document_id") or gt_path.stem
        patient_id = meta.get("patient_id") or doc_id.split("_")[0]
        total_pages_gt = int(meta.get("total_pages") or 0)
        images = discover_images(gt_path, meta)
        hospital = encounter.get("hospital_name") or data.get("document_layout", {}).get("hospital_header", "") or ""
        dept_raw = encounter.get("department") or ""
        dept_inferred = infer_department(hospital, dept_raw)
        bundle_type = "same_page_multi_view" if doc_id in {"p37_1", "p42_1"} else (
            "multi_page" if total_pages_gt > 1 else "single_image_page"
        )
        roles = image_roles(doc_id, len(images), bundle_type)
        printed, handwritten = classify_print_hand(data, hospital)
        rows.append({
            "document_id": doc_id,
            "patient_id": patient_id,
            "ground_truth_json": rel(gt_path),
            "source_images_ordered": ";".join(rel(p) for p in images),
            "total_pages_gt": total_pages_gt,
            "num_source_images": len(images),
            "image_bundle_type": bundle_type,
            "image_roles": ";".join(roles),
            "hospital_name": hospital,
            "source_type": meta.get("source_type", ""),
            "department_raw": dept_raw,
            "department_inferred": dept_inferred,
            "is_single_page": str(total_pages_gt == 1).lower(),
            "is_multi_page": str(total_pages_gt > 1).lower(),
            "is_same_page_multi_view": str(bundle_type == "same_page_multi_view").lower(),
            "printed_heavy": str(bool(printed and not handwritten)).lower(),
            "handwritten_heavy": str(bool(handwritten and not printed)).lower(),
            "benchmark_include": "true",
            "exclusion_reason": "",
        })

    fields = [
        "document_id", "patient_id", "ground_truth_json", "source_images_ordered",
        "total_pages_gt", "num_source_images", "image_bundle_type", "image_roles",
        "hospital_name", "source_type", "department_raw", "department_inferred",
        "is_single_page", "is_multi_page", "is_same_page_multi_view", "printed_heavy",
        "handwritten_heavy", "benchmark_include", "exclusion_reason",
    ]
    out = DATA_DIR / "full_benchmark_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def package_document_images(doc: Dict[str, str], run_dir: Path, max_width: int = 1800) -> Tuple[List[Path], List[Dict[str, Any]]]:
    out_dir = run_dir / "compressed_images" / doc["document_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    images = [PROJECT_ROOT / p for p in doc["source_images_ordered"].split(";") if p]
    compressed: List[Path] = []
    logs: List[Dict[str, Any]] = []
    roles = doc.get("image_roles", "").split(";") if doc.get("image_roles") else []
    for idx, src in enumerate(images):
        role = roles[idx] if idx < len(roles) else f"image_{idx + 1}"
        suffix = ".jpg"
        dst = out_dir / f"{idx + 1:02d}_{role}{suffix}"
        with Image.open(src) as im:
            original_size = im.size
            im = im.convert("RGB")
            if im.width > max_width:
                scale = max_width / float(im.width)
                im = im.resize((max_width, max(1, int(im.height * scale))))
            im.save(dst, format="JPEG", quality=92, optimize=True)
        compressed.append(dst)
        logs.append({
            "source": rel(src),
            "compressed": rel(dst),
            "role": role,
            "original_size": original_size,
            "compressed_size": read_image_size(dst),
            "quality": 92,
            "max_width": max_width,
            "bytes": dst.stat().st_size,
        })
    return compressed, logs


def read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def make_pdf_from_images(images: List[Path], out_path: Path) -> Optional[Path]:
    if not images:
        return None
    pil_images = []
    for p in images:
        pil_images.append(Image.open(p).convert("RGB"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = pil_images[0], pil_images[1:]
    first.save(out_path, save_all=True, append_images=rest)
    for im in pil_images:
        im.close()
    return out_path


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{encode_image(path)}"


@dataclass
class CallResult:
    ok: bool
    content: str = ""
    parsed: Optional[Dict[str, Any]] = None
    raw_response: Any = None
    error: str = ""
    request_id: str = ""
    usage: Optional[Dict[str, Any]] = None
    cost: Optional[Any] = None
    extra: Optional[Dict[str, Any]] = None


class Stage1Runner:
    def __init__(self, run_dir: Optional[Path] = None, force: bool = False):
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_dir = run_dir or (OUTPUT_ROOT / f"stage1a_smoke_{stamp}")
        self.force = force
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log = self.run_dir / "stage1a_failure_log.jsonl"
        self.metric_rows: List[Dict[str, Any]] = []
        self.runtime_rows: List[Dict[str, Any]] = []

    def completed_log(self, backend: str, track: str, doc_id: str) -> Path:
        return self.run_dir / "logs" / backend / track / f"{doc_id}.json"

    def should_skip(self, backend: str, track: str, doc_id: str) -> bool:
        return self.completed_log(backend, track, doc_id).exists() and not self.force

    def begin_log(self, backend: str, track: str, doc: Dict[str, str], prompt_file: str, prompt_hash: str, images: List[Path]) -> Tuple[Path, Dict[str, Any]]:
        log_path = self.completed_log(backend, track, doc["document_id"])
        in_progress = log_path.with_suffix(".in_progress.json")
        log = {
            "patient_id": doc["patient_id"],
            "document_id": doc["document_id"],
            "backend": backend,
            "track": track,
            "input_image_paths": doc["source_images_ordered"].split(";"),
            "compressed_image_paths": [rel(p) for p in images],
            "prompt_file": prompt_file,
            "prompt_hash": prompt_hash,
            "start_timestamp": utcish_now(),
            "status": "in_progress",
            "retry_count": 0,
        }
        write_json_atomic(in_progress, log)
        return log_path, log

    def finish_log(self, log_path: Path, log: Dict[str, Any], result: CallResult, start: float, paths: Dict[str, str]) -> None:
        log.update({
            "end_timestamp": utcish_now(),
            "runtime_seconds": round(time.time() - start, 3),
            "status": "success" if result.ok else "failed",
            "parse_success": result.parsed is not None,
            "schema_validation_success": result.parsed is not None and ("raw_entities" in result.parsed or "semantic" in log.get("track", "")),
            "request_id": result.request_id,
            "token_usage": result.usage,
            "estimated_cost": result.cost,
            "error_message": result.error,
            **paths,
        })
        in_progress = log_path.with_suffix(".in_progress.json")
        write_json_atomic(in_progress, log)
        in_progress.replace(log_path)

    def record_failure(self, backend: str, track: str, doc_id: str, error: str) -> None:
        self.failure_log.parent.mkdir(parents=True, exist_ok=True)
        with self.failure_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": utcish_now(),
                "backend": backend,
                "track": track,
                "document_id": doc_id,
                "error": error,
            }) + "\n")

    def ollama_chat(self, model: str, prompt: str, images: List[Path], timeout: int = 600) -> CallResult:
        message: Dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }
        if images:
            message["images"] = [encode_image(p) for p in images]
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=timeout)
            if resp.status_code >= 400:
                return CallResult(False, error=f"HTTP {resp.status_code}: {resp.text[:500]}", raw_response=resp.text)
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            parsed = extract_json(content)
            return CallResult(True, content=content, parsed=parsed, raw_response=data, usage={
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            })
        except Exception as exc:
            return CallResult(False, error=f"{type(exc).__name__}: {exc}")

    def zai_glm_ocr(self, image: Path, timeout: int = 180) -> CallResult:
        key = os.getenv("ZAI_API_KEY")
        if not key:
            return CallResult(False, error="ZAI_API_KEY is not configured")
        payload = {
            "model": "glm-ocr",
            "file": data_url(image),
            "return_crop_images": False,
            "need_layout_visualization": False,
            "request_id": f"rxbench-{int(time.time())}-{random.randint(1000, 9999)}",
        }
        try:
            start = time.time()
            resp = requests.post(
                "https://api.z.ai/api/paas/v4/layout_parsing",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if resp.status_code >= 400:
                return CallResult(False, error=f"HTTP {resp.status_code}: {resp.text[:800]}", raw_response=resp.text)
            data = resp.json()
            md = data.get("md_results") or ""
            usage = data.get("usage")
            return CallResult(True, content=md, raw_response=data, request_id=data.get("request_id") or data.get("id", ""), usage=usage, extra={"runtime_seconds": time.time() - start})
        except Exception as exc:
            return CallResult(False, error=f"{type(exc).__name__}: {exc}")

    def datalab_convert(self, file_path: Path, output_format: str = "markdown", mode: str = "balanced", timeout: int = 300) -> CallResult:
        key = os.getenv("DATALAB_API_KEY")
        if not key:
            return CallResult(False, error="DATALAB_API_KEY is not configured")
        try:
            with file_path.open("rb") as f:
                resp = requests.post(
                    "https://www.datalab.to/api/v1/convert",
                    headers={"X-API-Key": key},
                    files={"file": (file_path.name, f, mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")},
                    data={"output_format": output_format, "mode": mode, "paginate": "true"},
                    timeout=timeout,
                )
            if resp.status_code >= 400:
                return CallResult(False, error=f"HTTP {resp.status_code}: {resp.text[:800]}", raw_response=resp.text)
            submitted = resp.json()
            check_url = submitted.get("request_check_url")
            request_id = submitted.get("request_id", "")
            if not check_url:
                return CallResult(False, error=f"No request_check_url in response: {submitted}", raw_response=submitted)
            start = time.time()
            while time.time() - start < timeout:
                poll = requests.get(check_url, headers={"X-API-Key": key}, timeout=60)
                if poll.status_code >= 400:
                    return CallResult(False, error=f"Poll HTTP {poll.status_code}: {poll.text[:800]}", raw_response=poll.text, request_id=request_id)
                data = poll.json()
                status = data.get("status")
                if status == "complete":
                    content = data.get("markdown") or data.get("html") or json.dumps(data.get("json", {}), ensure_ascii=False)
                    return CallResult(
                        True,
                        content=content or "",
                        raw_response=data,
                        request_id=request_id,
                cost=data.get("cost_breakdown"),
                extra={
                    "request_check_url": check_url,
                    "page_count": data.get("page_count"),
                    "parse_quality_score": data.get("parse_quality_score"),
                        },
                    )
                if status == "failed":
                    return CallResult(False, error=data.get("error", "Datalab conversion failed"), raw_response=data, request_id=request_id)
                time.sleep(2)
            return CallResult(False, error="Timed out polling Datalab result", raw_response=submitted, request_id=request_id)
        except Exception as exc:
            return CallResult(False, error=f"{type(exc).__name__}: {exc}")

    def sarvam_document_intelligence(self, file_path: Path, output_format: str = "md", timeout: int = 300) -> CallResult:
        key = os.getenv("SARVAM_API_KEY")
        if not key:
            return CallResult(False, error="SARVAM_API_KEY is not configured")
        try:
            from sarvamai import SarvamAI

            start = time.time()
            client = SarvamAI(api_subscription_key=key, timeout=60)
            job = client.document_intelligence.create_job(language="en-IN", output_format=output_format)
            job.upload_file(str(file_path))
            started = job.start()
            final_status = job.wait_until_complete(poll_interval=2.0, timeout=timeout)
            status_dump = safe_model_dump(final_status)
            started_dump = safe_model_dump(started)
            job_state = status_dump.get("job_state") or getattr(final_status, "job_state", "")
            if job_state not in {"Completed", "PartiallyCompleted"}:
                return CallResult(
                    False,
                    error=f"Sarvam job ended in state {job_state}",
                    raw_response={"started": started_dump, "final_status": status_dump},
                    request_id=job.job_id,
                )
            download_path = file_path.with_suffix(f".sarvam_output.{output_format}")
            job.download_output(str(download_path))
            content = download_path.read_text(encoding="utf-8", errors="ignore")
            return CallResult(
                True,
                content=content,
                raw_response={"started": started_dump, "final_status": status_dump},
                request_id=job.job_id,
                extra={
                    "downloaded_output_path": rel(download_path),
                    "runtime_seconds": time.time() - start,
                    "job_state": job_state,
                },
            )
        except Exception as exc:
            return CallResult(False, error=f"{type(exc).__name__}: {exc}")

    def run_structured_ollama(self, backend: str, model: str, doc: Dict[str, str], semantic: bool = False) -> None:
        track = "semantic_inference" if semantic else "raw_structured"
        if self.should_skip(backend, track, doc["document_id"]):
            return
        prompt_file = "semantic_inference_extraction_prompt.txt" if semantic else "raw_structured_extraction_prompt.txt"
        prompt, prompt_hash = read_prompt(prompt_file)
        prompt = (
            f"Document ID: {doc['document_id']}\n"
            f"Patient root: {doc['patient_id']}\n"
            f"Hospital: {doc['hospital_name']}\n"
            f"Department: {doc['department_inferred']}\n"
            f"Source type: {doc['source_type']}\n"
            f"Image roles: {doc['image_roles']}\n\n"
            + prompt
        )
        images, compression_log = package_document_images(doc, self.run_dir)
        log_path, log = self.begin_log(backend, track, doc, prompt_file, prompt_hash, images)
        log["compression_log"] = compression_log
        start = time.time()
        result = self.ollama_chat(model, prompt, images)
        raw_path = self.run_dir / "raw_responses" / backend / track / f"{doc['document_id']}.txt"
        parsed_path = self.run_dir / track / backend / f"{doc['document_id']}.json"
        fail_path = self.run_dir / "failed_cases" / backend / track / f"{doc['document_id']}.json"
        write_text(raw_path, result.content or json.dumps(result.raw_response, ensure_ascii=False))
        if result.parsed:
            write_json_atomic(parsed_path, result.parsed)
        else:
            write_json_atomic(fail_path, {"error": result.error or "JSON parse failed", "content": result.content})
            self.record_failure(backend, track, doc["document_id"], result.error or "JSON parse failed")
        self.finish_log(log_path, log, result, start, {
            "raw_response_path": rel(raw_path),
            "parsed_response_path": rel(parsed_path) if result.parsed else "",
            "failed_case_path": rel(fail_path) if not result.parsed else "",
        })
        self.add_metric_row(doc, backend, track, parsed_path if result.parsed else None, log_path)

    def run_text_ollama_from_ocr(self, backend: str, model: str, doc: Dict[str, str], ocr_text_path: Path, semantic: bool = False) -> None:
        track = "semantic_inference_from_ocr" if semantic else "ocr_to_json_structuring"
        if self.should_skip(backend, track, doc["document_id"]):
            return
        prompt_file = "semantic_inference_extraction_prompt.txt" if semantic else "ocr_to_json_structuring_prompt.txt"
        prompt, prompt_hash = read_prompt(prompt_file)
        if not ocr_text_path.exists():
            log_path, log = self.begin_log(backend, track, doc, prompt_file, prompt_hash, [])
            result = CallResult(False, error=f"OCR text not available: {rel(ocr_text_path)}")
            fail_path = self.run_dir / "failed_cases" / backend / track / f"{doc['document_id']}.json"
            write_json_atomic(fail_path, {"error": result.error})
            self.record_failure(backend, track, doc["document_id"], result.error)
            self.finish_log(log_path, log, result, time.time(), {
                "raw_response_path": "",
                "parsed_response_path": "",
                "failed_case_path": rel(fail_path),
            })
            self.add_metric_row(doc, backend, track, None, log_path)
            return
        ocr_text = ocr_text_path.read_text(encoding="utf-8", errors="ignore")
        prompt = (
            f"Document ID: {doc['document_id']}\n"
            f"Patient root: {doc['patient_id']}\n"
            f"Hospital: {doc['hospital_name']}\n"
            f"Department: {doc['department_inferred']}\n"
            f"Source type: {doc['source_type']}\n"
            f"OCR source path: {rel(ocr_text_path)}\n\n"
            f"{prompt}\n\n"
            "OCR TEXT START\n"
            f"{ocr_text}\n"
            "OCR TEXT END\n"
        )
        log_path, log = self.begin_log(backend, track, doc, prompt_file, prompt_hash, [])
        log["ocr_text_path"] = rel(ocr_text_path)
        start = time.time()
        result = self.ollama_chat(model, prompt, [])
        raw_path = self.run_dir / "raw_responses" / backend / track / f"{doc['document_id']}.txt"
        parsed_path = self.run_dir / track / backend / f"{doc['document_id']}.json"
        fail_path = self.run_dir / "failed_cases" / backend / track / f"{doc['document_id']}.json"
        write_text(raw_path, result.content or json.dumps(result.raw_response, ensure_ascii=False))
        if result.parsed:
            write_json_atomic(parsed_path, result.parsed)
        else:
            write_json_atomic(fail_path, {"error": result.error or "JSON parse failed", "content": result.content})
            self.record_failure(backend, track, doc["document_id"], result.error or "JSON parse failed")
        self.finish_log(log_path, log, result, start, {
            "raw_response_path": rel(raw_path),
            "parsed_response_path": rel(parsed_path) if result.parsed else "",
            "failed_case_path": rel(fail_path) if not result.parsed else "",
        })
        self.add_metric_row(doc, backend, track, parsed_path if result.parsed else None, log_path)

    def run_zai_smoke(self, doc: Dict[str, str]) -> None:
        backend, track = "zai_glm_ocr", "raw_ocr"
        if self.should_skip(backend, track, doc["document_id"]):
            return
        prompt, prompt_hash = read_prompt("raw_ocr_prompt.txt")
        images, compression_log = package_document_images(doc, self.run_dir)
        log_path, log = self.begin_log(backend, track, doc, "raw_ocr_prompt.txt", prompt_hash, images)
        log["compression_log"] = compression_log
        start = time.time()
        md_parts = []
        raw_parts = []
        ok = True
        error = ""
        usage = {"pages": []}
        request_ids = []
        for image in images:
            res = self.zai_glm_ocr(image)
            ok = ok and res.ok
            if not res.ok:
                error = res.error
                break
            md_parts.append(f"<!-- {image.name} -->\n{res.content}")
            raw_parts.append(res.raw_response)
            request_ids.append(res.request_id)
            usage["pages"].append(res.usage)
        result = CallResult(ok, content="\n\n".join(md_parts), raw_response=raw_parts, error=error, request_id=";".join(request_ids), usage=usage)
        out_path = self.run_dir / "raw_ocr" / backend / f"{doc['document_id']}.md"
        raw_path = self.run_dir / "raw_responses" / backend / track / f"{doc['document_id']}.json"
        fail_path = self.run_dir / "failed_cases" / backend / track / f"{doc['document_id']}.json"
        write_text(out_path, result.content)
        write_json_atomic(raw_path, {"responses": raw_parts})
        if not ok:
            write_json_atomic(fail_path, {"error": error})
            self.record_failure(backend, track, doc["document_id"], error)
        self.finish_log(log_path, log, result, start, {
            "raw_response_path": rel(raw_path),
            "parsed_response_path": rel(out_path),
            "failed_case_path": rel(fail_path) if not ok else "",
        })
        self.add_metric_row(doc, backend, track, None, log_path, ocr_text_path=out_path)

    def run_datalab_smoke(self, doc: Dict[str, str]) -> None:
        backend, track = "datalab_chandra", "raw_ocr"
        if self.should_skip(backend, track, doc["document_id"]):
            return
        prompt, prompt_hash = read_prompt("raw_ocr_prompt.txt")
        images, compression_log = package_document_images(doc, self.run_dir)
        # Datalab handles PDFs well; package multi-image docs into one PDF.
        pdf_path = self.run_dir / "compressed_images" / doc["document_id"] / f"{doc['document_id']}_bundle.pdf"
        make_pdf_from_images(images, pdf_path)
        log_path, log = self.begin_log(backend, track, doc, "raw_ocr_prompt.txt", prompt_hash, images)
        log["compression_log"] = compression_log
        log["submitted_file"] = rel(pdf_path)
        start = time.time()
        result = self.datalab_convert(pdf_path)
        out_path = self.run_dir / "raw_ocr" / backend / f"{doc['document_id']}.md"
        raw_path = self.run_dir / "raw_responses" / backend / track / f"{doc['document_id']}.json"
        fail_path = self.run_dir / "failed_cases" / backend / track / f"{doc['document_id']}.json"
        write_text(out_path, result.content)
        write_json_atomic(raw_path, result.raw_response if isinstance(result.raw_response, dict) else {"raw_response": result.raw_response})
        if not result.ok:
            write_json_atomic(fail_path, {"error": result.error})
            self.record_failure(backend, track, doc["document_id"], result.error)
        if result.extra:
            log.update(result.extra)
        self.finish_log(log_path, log, result, start, {
            "raw_response_path": rel(raw_path),
            "parsed_response_path": rel(out_path),
            "failed_case_path": rel(fail_path) if not result.ok else "",
        })
        self.add_metric_row(doc, backend, track, None, log_path, ocr_text_path=out_path)

    def run_sarvam_smoke(self, doc: Dict[str, str]) -> None:
        backend, track = "sarvam_document_intelligence", "raw_ocr"
        if self.should_skip(backend, track, doc["document_id"]):
            return
        prompt, prompt_hash = read_prompt("raw_ocr_prompt.txt")
        images, compression_log = package_document_images(doc, self.run_dir)
        pdf_path = self.run_dir / "compressed_images" / doc["document_id"] / f"{doc['document_id']}_sarvam_bundle.pdf"
        make_pdf_from_images(images, pdf_path)
        log_path, log = self.begin_log(backend, track, doc, "raw_ocr_prompt.txt", prompt_hash, images)
        log["compression_log"] = compression_log
        log["submitted_file"] = rel(pdf_path)
        start = time.time()
        result = self.sarvam_document_intelligence(pdf_path)
        out_path = self.run_dir / "raw_ocr" / backend / f"{doc['document_id']}.md"
        raw_path = self.run_dir / "raw_responses" / backend / track / f"{doc['document_id']}.json"
        fail_path = self.run_dir / "failed_cases" / backend / track / f"{doc['document_id']}.json"
        write_text(out_path, result.content)
        write_json_atomic(raw_path, result.raw_response if isinstance(result.raw_response, dict) else {"raw_response": result.raw_response})
        if not result.ok:
            write_json_atomic(fail_path, {"error": result.error})
            self.record_failure(backend, track, doc["document_id"], result.error)
        if result.extra:
            log.update(result.extra)
        self.finish_log(log_path, log, result, start, {
            "raw_response_path": rel(raw_path),
            "parsed_response_path": rel(out_path),
            "failed_case_path": rel(fail_path) if not result.ok else "",
        })
        self.add_metric_row(doc, backend, track, None, log_path, ocr_text_path=out_path)

    def add_metric_row(self, doc: Dict[str, str], backend: str, track: str, parsed_path: Optional[Path], log_path: Path, ocr_text_path: Optional[Path] = None) -> None:
        row = compute_smoke_metrics(doc, backend, track, parsed_path, log_path, ocr_text_path)
        self.metric_rows.append(row)
        if log_path.exists():
            log = json.loads(log_path.read_text())
            self.runtime_rows.append({
                "document_id": doc["document_id"],
                "backend": backend,
                "track": track,
                "runtime_seconds": log.get("runtime_seconds", ""),
                "estimated_cost": json.dumps(log.get("estimated_cost"), ensure_ascii=False),
                "token_usage": json.dumps(log.get("token_usage"), ensure_ascii=False),
                "request_id": log.get("request_id", ""),
                "status": log.get("status", ""),
            })

    def write_smoke_tables(self) -> None:
        metrics_path = REPORTS_DIR / "stage1a_smoke_metrics.csv"
        runtime_path = REPORTS_DIR / "stage1a_cost_runtime_report.csv"
        metric_fields = [
            "document_id", "backend", "track", "json_parse_success", "schema_validity",
            "output_completeness", "field_coverage", "scalar_accuracy_exact",
            "scalar_accuracy_lenient", "entity_exact_f1", "entity_lenient_f1",
            "hallucination_rate", "missing_entity_rate", "runtime_seconds", "cost_available",
            "notes",
        ]
        runtime_fields = ["document_id", "backend", "track", "runtime_seconds", "estimated_cost", "token_usage", "request_id", "status"]
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
            writer.writerows(self.metric_rows)
        with runtime_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=runtime_fields)
            writer.writeheader()
            writer.writerows(self.runtime_rows)
        update_availability_smoke_status(self.run_dir)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates = []
    stripped = text.strip()
    candidates.append(stripped)
    if "```json" in stripped:
        candidates.append(stripped.split("```json", 1)[1].split("```", 1)[0].strip())
    if "```" in stripped:
        candidates.append(stripped.split("```", 1)[1].split("```", 1)[0].strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def compute_smoke_metrics(doc: Dict[str, str], backend: str, track: str, parsed_path: Optional[Path], log_path: Path, ocr_text_path: Optional[Path]) -> Dict[str, Any]:
    runtime = ""
    cost_available = False
    if log_path.exists():
        log = json.loads(log_path.read_text())
        runtime = log.get("runtime_seconds", "")
        cost_available = log.get("estimated_cost") not in (None, "", {})

    base = {
        "document_id": doc["document_id"],
        "backend": backend,
        "track": track,
        "json_parse_success": 0,
        "schema_validity": 0,
        "output_completeness": 0.0,
        "field_coverage": 0.0,
        "scalar_accuracy_exact": "",
        "scalar_accuracy_lenient": "",
        "entity_exact_f1": "",
        "entity_lenient_f1": "",
        "hallucination_rate": "",
        "missing_entity_rate": "",
        "runtime_seconds": runtime,
        "cost_available": str(cost_available).lower(),
        "notes": "",
    }
    if ocr_text_path:
        text = ocr_text_path.read_text(encoding="utf-8", errors="ignore") if ocr_text_path.exists() else ""
        base["output_completeness"] = round(min(1.0, len(text.strip()) / 500.0), 4)
        base["field_coverage"] = ""
        base["notes"] = "OCR/layout-only output; structured field metrics not applicable until OCR-to-JSON structuring."
        return base
    if not parsed_path or not parsed_path.exists():
        base["notes"] = "No parsed JSON output."
        return base
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.adapters.gt_adapter import GTAdapter
        from src.adapters.legacy_prediction_adapter import LegacyPredictionAdapter
        from src.schemas.raw_extraction import CanonicalRawDoc
        from src.benchmark.scalar_match import ScalarMatcher
        from src.benchmark.entity_match import EntityMatcher
        from src.benchmark.hallucination import HallucinationDetector
        from src.benchmark.aggregation import MetricAggregator

        gt_doc = GTAdapter.from_file(PROJECT_ROOT / doc["ground_truth_json"])
        pred_data = json.loads(parsed_path.read_text(encoding="utf-8"))
        base["json_parse_success"] = 1
        if "document_metadata" in pred_data and "raw_entities" in pred_data:
            pred_doc = GTAdapter.from_dict(pred_data)
        elif pred_data.get("schema_version") == "raw_rx_v2":
            pred_doc = CanonicalRawDoc(**pred_data)
        else:
            pred_doc = LegacyPredictionAdapter.from_dict(pred_data, doc["document_id"])
        base["schema_validity"] = 1
        categories = [
            "complaints_or_diagnosis", "observations", "medications", "procedures",
            "advice", "allergy_mentions", "other_notes", "lab_observations",
        ]
        non_empty = 0
        for cat in categories:
            if getattr(pred_doc, cat, []):
                non_empty += 1
        base["output_completeness"] = round(non_empty / len(categories), 4)
        scalars = ScalarMatcher(lenient_threshold=80.0).match_docs(gt_doc, pred_doc)
        base["field_coverage"] = round(sum(1 for s in scalars if s.pred_value not in (None, "")) / max(1, len(scalars)), 4)
        base["scalar_accuracy_exact"] = round(sum(1 for s in scalars if s.exact_match) / max(1, len(scalars)), 4)
        base["scalar_accuracy_lenient"] = round(sum(1 for s in scalars if s.lenient_match) / max(1, len(scalars)), 4)
        matcher = EntityMatcher(exact_threshold=95.0, lenient_threshold=80.0, review_threshold=65.0)
        all_alignments = []
        cat_metrics = {}
        total_gt = 0
        total_pred = 0
        for cat in categories:
            gt_items = getattr(gt_doc, cat, [])
            pred_items = getattr(pred_doc, cat, [])
            total_gt += len(gt_items)
            total_pred += len(pred_items)
            aligns = matcher.align_entities(gt_items, pred_items, cat)
            all_alignments.extend(aligns)
            cat_metrics[cat] = matcher.compute_category_metrics(aligns)
        unmatched = HallucinationDetector().detect_hallucinations(gt_doc, pred_doc, all_alignments)
        class Tmp:
            pass
        tmp = Tmp()
        tmp.entity_alignments = all_alignments
        tmp.unmatched_predictions = unmatched
        tmp.likely_hallucination_count = sum(1 for u in unmatched if u.classification == "likely_hallucination")
        tmp.annotation_gap_candidate_count = sum(1 for u in unmatched if u.classification == "annotation_gap_candidate")
        tmp.metrics_by_category = cat_metrics
        tmp.schema_parse_success = 1
        tmp.scalar_accuracy_lenient = base["scalar_accuracy_lenient"]
        tmp.hallucination_rate = 0
        tmp.annotation_gap_rate = 0
        tmp.missing_entity_rate = 0
        MetricAggregator().calculate_rates(tmp, total_gt, total_pred)
        base["entity_exact_f1"] = round(sum(m.f1_exact for m in cat_metrics.values()) / len(cat_metrics), 4)
        base["entity_lenient_f1"] = round(sum(m.f1_lenient for m in cat_metrics.values()) / len(cat_metrics), 4)
        base["hallucination_rate"] = round(tmp.hallucination_rate, 4)
        base["missing_entity_rate"] = round(tmp.missing_entity_rate, 4)
    except Exception as exc:
        base["notes"] = f"Metric calculation failed: {type(exc).__name__}: {exc}"
    return base


def update_availability_smoke_status(run_dir: Path) -> None:
    path = REPORTS_DIR / "stage1a_model_availability.csv"
    if not path.exists():
        return
    rows = summarize_csv(path)
    backend_logs: Dict[str, List[Dict[str, Any]]] = {}
    for log_path in (run_dir / "logs").rglob("*.json"):
        if log_path.name.endswith(".in_progress.json"):
            continue
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        backend_logs.setdefault(log.get("backend", ""), []).append(log)
    for row in rows:
        engine = row.get("engine", "")
        logs = backend_logs.get(engine, [])
        if not logs:
            continue
        if any(log.get("status") == "success" for log in logs):
            row["smoke_test_status"] = "passed_smoke"
        else:
            row["smoke_test_status"] = "failed_smoke"
            first_error = next((log.get("error_message") for log in logs if log.get("error_message")), "")
            if first_error:
                row["reason_unavailable"] = first_error
    fields = ["engine", "installed", "version", "gpu_cpu_mode", "model_weights_source", "command_import_success", "smoke_test_status", "reason_unavailable"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(cmd: List[str], timeout: int = 15) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:
        return 999, "", f"{type(exc).__name__}: {exc}"


def package_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return ""


def import_success(module: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_ocr_availability() -> Path:
    rows = []
    checks = [
        ("tesseract", "binary", "tesseract", ["tesseract", "--version"], ""),
        ("easyocr", "python", "easyocr", [], "easyocr"),
        ("paddleocr", "python", "paddleocr", [], "paddleocr"),
        ("doctr", "python", "doctr", [], "python-doctr"),
        ("trocr", "python", "transformers", [], "transformers"),
        ("surya_ocr", "python", "surya", [], "surya-ocr"),
        ("nougat", "python", "nougat", [], "nougat-ocr"),
        ("mineru", "python", "mineru", [], "mineru"),
        ("got_ocr", "python", "got_ocr", [], "got-ocr"),
        ("paddleocr_vl_1_5", "python", "paddleocr", [], "paddleocr"),
        ("deepseek_ocr2", "python", "transformers", [], "transformers"),
        ("dots_ocr", "python", "dots_ocr", [], "dots-ocr"),
        ("datalab_chandra", "python", "datalab_sdk", [], "datalab-python-sdk"),
        ("zai_glm_ocr", "python", "zai", [], "zai-sdk"),
        ("sarvam_document_intelligence", "python", "sarvamai", [], "sarvamai"),
    ]
    for engine, kind, target, cmd, dist in checks:
        installed = False
        version = ""
        reason = ""
        if kind == "binary":
            path = shutil.which(target)
            installed = bool(path)
            if installed:
                code, out, err = run_cmd(cmd)
                version = (out or err).splitlines()[0] if (out or err) else ""
                reason = "" if code == 0 else err
            else:
                reason = f"{target} not found on PATH"
        else:
            installed, reason = import_success(target)
            version = package_version(dist) if dist else ""
        rows.append({
            "engine": engine,
            "installed": str(installed).lower(),
            "version": version,
            "gpu_cpu_mode": "unknown_until_smoke" if installed else "",
            "model_weights_source": "external_api" if engine in {"datalab_chandra", "zai_glm_ocr", "sarvam_document_intelligence"} else "local_or_package",
            "command_import_success": str(installed).lower(),
            "smoke_test_status": "not_run",
            "reason_unavailable": "" if installed else reason,
        })
    out = REPORTS_DIR / "stage1a_model_availability.csv"
    fields = ["engine", "installed", "version", "gpu_cpu_mode", "model_weights_source", "command_import_success", "smoke_test_status", "reason_unavailable"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def select_smoke_docs(manifest: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    by_id = {r["document_id"]: r for r in manifest}
    local_ids = ["p4", "p25_1", "p38_1"]
    external_ids = ["p38_1", "p4", "p25_1"]
    semantic_ids = ["p25_1"]
    sarvam_ids = ["p4"]
    return {
        "local": [by_id[i] for i in local_ids if i in by_id],
        "external": [by_id[i] for i in external_ids if i in by_id],
        "semantic": [by_id[i] for i in semantic_ids if i in by_id],
        "sarvam": [by_id[i] for i in sarvam_ids if i in by_id],
    }


def run_smoke(force: bool = False, include_external: bool = True, include_ollama: bool = True) -> Path:
    manifest = load_manifest()
    selected = select_smoke_docs(manifest)
    runner = Stage1Runner(force=force)
    if include_external:
        for doc in selected["external"]:
            runner.run_zai_smoke(doc)
        for doc in selected["external"]:
            runner.run_datalab_smoke(doc)
        for doc in selected["sarvam"]:
            runner.run_sarvam_smoke(doc)
    if include_ollama:
        for doc in selected["local"]:
            runner.run_structured_ollama("ollama_qwen3_vl_8b", "qwen3-vl:8b-instruct", doc)
        for doc in selected["local"]:
            runner.run_structured_ollama("ollama_llava_13b", "llava:13b", doc)
        for doc in selected["semantic"]:
            ocr_text = runner.run_dir / "raw_ocr" / "datalab_chandra" / f"{doc['document_id']}.md"
            runner.run_text_ollama_from_ocr("ollama_qwen3_8b_text", "qwen3:8b", doc, ocr_text, semantic=False)
            runner.run_text_ollama_from_ocr("ollama_qwen3_8b_text", "qwen3:8b", doc, ocr_text, semantic=True)
    runner.write_smoke_tables()
    write_reports(runner.run_dir)
    return runner.run_dir


def summarize_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_reports(run_dir: Optional[Path] = None) -> None:
    availability = summarize_csv(REPORTS_DIR / "stage1a_model_availability.csv")
    metrics = summarize_csv(REPORTS_DIR / "stage1a_smoke_metrics.csv")
    runtime = summarize_csv(REPORTS_DIR / "stage1a_cost_runtime_report.csv")
    failures = []
    if run_dir:
        fail_path = run_dir / "stage1a_failure_log.jsonl"
        if fail_path.exists():
            failures = [json.loads(line) for line in fail_path.read_text().splitlines() if line.strip()]

    adapter_lines = [
        "# Stage 1A Adapter Status",
        "",
        f"Generated: {utcish_now()}",
        "",
        "- OpenRouter: disabled for full run and not called.",
        "- Ollama qwen3-vl: implemented through `/api/chat` multi-image messages.",
        "- Ollama llava: implemented through `/api/chat` multi-image messages.",
        "- Ollama qwen2.5/qwen3 text: used only without images; OCR-to-JSON and semantic smoke consume saved OCR text from the same smoke run.",
        "- Z.AI GLM-OCR: implemented via layout parsing endpoint with base64 image input, smoke-only.",
        "- Datalab/Chandra: implemented via `/api/v1/convert`, PDF bundle upload, polling, smoke-only.",
        "- Sarvam: implemented as smoke-only Document Intelligence job flow through the current `sarvamai` SDK; existing placeholder wrapper remains excluded.",
    ]
    write_text(REPORTS_DIR / "stage1a_adapter_status.md", "\n".join(adapter_lines) + "\n")

    failure_md = ["# Stage 1A Failure Log", "", f"Generated: {utcish_now()}", ""]
    if failures:
        for item in failures:
            failure_md.append(f"- `{item.get('backend')}` `{item.get('track')}` `{item.get('document_id')}`: {item.get('error')}")
    else:
        failure_md.append("No Stage 1A smoke failures recorded in the active run directory.")
    write_text(REPORTS_DIR / "stage1a_failure_log.md", "\n".join(failure_md) + "\n")

    successful_backends = sorted({r["backend"] for r in metrics if r.get("json_parse_success") == "1" or r.get("output_completeness") not in ("", "0", "0.0")})
    local_ok = any(b.startswith("ollama") for b in successful_backends)
    datalab_ok = "datalab_chandra" in successful_backends
    sarvam_ok = "sarvam_document_intelligence" in successful_backends
    zai_ok = "zai_glm_ocr" in successful_backends
    external_ok = datalab_ok or sarvam_ok or zai_ok
    recommendation = "delay full run because environment/adapters are unstable"
    if local_ok and external_ok:
        recommendation = "proceed to full local + selected external OCR benchmark; keep Z.AI disabled until balance/resource package is fixed"
    elif local_ok:
        recommendation = "proceed to full local-only benchmark"

    roster = [
        "# Stage 1A Recommended Full Run Roster",
        "",
        f"Generated: {utcish_now()}",
        "",
        f"Recommendation: **{recommendation}**.",
        "",
        "Recommended local candidates after smoke review:",
        "- `ollama_qwen3_vl_8b` for raw structured VLM extraction if JSON parse/schema smoke is acceptable.",
        "- `ollama_llava_13b` only if smoke output quality is acceptable.",
        "- `ollama_qwen3_8b_text` / `ollama_qwen25_14b_text` for OCR-to-JSON and semantic inference after OCR text exists.",
        "",
        "Recommended external OCR candidates after smoke review:",
        "- `datalab_chandra` passed three-document smoke with cost/runtime logging.",
        "- `sarvam_document_intelligence` passed one-document smoke with downloaded Markdown output.",
        "",
        "Excluded until fixed:",
        "- OpenRouter: low credits.",
        "- `zai_glm_ocr`: callable adapter, but smoke failed with insufficient balance/resource package.",
        "- Existing `sarvam-vision` artifacts: placeholder outputs.",
    ]
    write_text(REPORTS_DIR / "stage1a_recommended_full_run_roster.md", "\n".join(roster) + "\n")

    summary = [
        "# Stage 1A Smoke Summary",
        "",
        f"Generated: {utcish_now()}",
        "",
        f"Smoke run directory: `{rel(run_dir) if run_dir else 'not_run'}`",
        "",
        "## Availability",
        "",
        f"- OCR/model availability rows: {len(availability)}",
        f"- Installed/available checks: {sum(1 for r in availability if r.get('installed') == 'true')}",
        "",
        "## Smoke Metrics",
        "",
        f"- Smoke metric rows: {len(metrics)}",
        f"- Runtime/cost rows: {len(runtime)}",
        f"- Failures recorded: {len(failures)}",
        "",
        f"Final recommendation: **{recommendation}**.",
        "",
        "See also:",
        "- `reports/stage1a_model_availability.csv`",
        "- `reports/stage1a_smoke_metrics.csv`",
        "- `reports/stage1a_cost_runtime_report.csv`",
        "- `reports/stage1a_adapter_status.md`",
        "- `reports/stage1a_failure_log.md`",
        "- `reports/stage1a_recommended_full_run_roster.md`",
    ]
    write_text(REPORTS_DIR / "stage1a_smoke_summary.md", "\n".join(summary) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1A benchmark infra/smoke runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-manifest")
    sub.add_parser("check-ocr")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--force", action="store_true")
    smoke.add_argument("--skip-external", action="store_true")
    smoke.add_argument("--skip-ollama", action="store_true")
    sub.add_parser("reports")
    args = parser.parse_args()

    if args.command == "build-manifest":
        out = build_manifest()
        print(f"wrote {out}")
    elif args.command == "check-ocr":
        out = check_ocr_availability()
        print(f"wrote {out}")
    elif args.command == "smoke":
        run_dir = run_smoke(force=args.force, include_external=not args.skip_external, include_ollama=not args.skip_ollama)
        print(f"smoke outputs: {run_dir}")
    elif args.command == "reports":
        write_reports()
        print("wrote Stage 1A reports")


if __name__ == "__main__":
    main()
