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

"""Stage 1B Extended Ollama image-to-text runner.

This runner is intentionally separate from frozen Stage 1B outputs. It writes
raw OCR text, raw model responses, per-document logs, failed cases, and a handoff
CSV suitable for scripts/benchmark_raw_ocr_outputs.py.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sort_key(value: Any) -> list[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


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


def image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def gpu_memory() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def compress_images(doc: dict[str, str], output_root: Path, max_dim: int, jpeg_quality: int) -> tuple[list[Path], list[dict[str, Any]]]:
    out_dir = output_root / "compressed_images" / doc["document_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = doc.get("image_roles", "").split(";") if doc.get("image_roles") else []
    out_paths: list[Path] = []
    logs: list[dict[str, Any]] = []
    for idx, raw in enumerate(doc["source_images_ordered"].split(";")):
        src = PROJECT_ROOT / raw
        role = roles[idx] if idx < len(roles) else f"image_{idx + 1}"
        dst = out_dir / f"{idx + 1:02d}_{role}.jpg"
        with Image.open(src) as image:
            original_size = image.size
            image = image.convert("RGB")
            scale = min(1.0, max_dim / max(image.width, image.height))
            if scale < 1.0:
                image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
            compressed_size = image.size
            image.save(dst, "JPEG", quality=jpeg_quality, optimize=True)
        out_paths.append(dst)
        logs.append(
            {
                "source": raw,
                "compressed": str(dst),
                "role": role,
                "original_size": list(original_size),
                "compressed_size": list(compressed_size),
                "bytes": dst.stat().st_size,
            }
        )
    return out_paths, logs


def ollama_chat(model: str, prompt: str, images: list[Path], timeout: int) -> tuple[bool, str, dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64(path) for path in images]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        if response.status_code >= 400:
            return False, "", {}, f"HTTP {response.status_code}: {response.text[:800]}"
        data = response.json()
        return True, data.get("message", {}).get("content", ""), data, ""
    except Exception as exc:
        return False, "", {}, f"{type(exc).__name__}: {exc}"


def prompt_text(template: str, doc: dict[str, str], images: list[Path]) -> str:
    image_list = "\n".join(str(path) for path in images)
    return template.format(
        document_id=doc["document_id"],
        patient_id=doc.get("patient_id", ""),
        hospital_name=doc.get("hospital_name", ""),
        department=doc.get("department_inferred", ""),
        image_path=image_list,
    )


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        docs = [row for row in read_csv(Path(args.manifest)) if row.get("benchmark_include", "true").lower() == "true"]
        if args.doc_ids:
            wanted = {item.strip() for item in args.doc_ids.split(",") if item.strip()}
            docs = [doc for doc in docs if doc["document_id"] in wanted]
            order = {doc_id: i for i, doc_id in enumerate(args.doc_ids.split(","))}
            docs.sort(key=lambda row: order.get(row["document_id"], 999))
        else:
            docs.sort(key=lambda row: sort_key(row["document_id"]))
        self.docs = docs
        self.handoff_rows: list[dict[str, Any]] = []

    def paths(self, doc_id: str) -> dict[str, Path]:
        return {
            "ocr": self.output_root / "raw_ocr" / self.args.lane / f"{doc_id}.txt",
            "raw": self.output_root / "raw_responses" / self.args.lane / f"{doc_id}.txt",
            "log": self.output_root / "logs" / self.args.lane / f"{doc_id}.json",
            "failed": self.output_root / "failed_cases" / self.args.lane / f"{doc_id}.json",
        }

    def should_skip(self, doc_id: str) -> bool:
        if not self.args.resume:
            return False
        log_path = self.paths(doc_id)["log"]
        if not log_path.exists():
            return False
        try:
            return json.loads(log_path.read_text(encoding="utf-8")).get("status") == "success"
        except Exception:
            return False

    def run_one(self, doc: dict[str, str]) -> None:
        doc_id = doc["document_id"]
        p = self.paths(doc_id)
        if self.should_skip(doc_id):
            self.handoff_rows.append(self.handoff_row(doc, "available", p["ocr"], p["log"]))
            return
        images, compression_log = compress_images(doc, self.output_root, self.args.max_image_dim, self.args.jpeg_quality)
        prompt = prompt_text(self.args.prompt, doc, images)
        started = time.time()
        gpu_before = gpu_memory()
        ok, content, raw, error = ollama_chat(self.args.model, prompt, images, self.args.timeout)
        runtime = round(time.time() - started, 3)
        text = content.strip()
        write_text(p["raw"], content or json.dumps(raw, ensure_ascii=False))
        if ok and text:
            write_text(p["ocr"], text + "\n")
            status = "success"
            failure_reason = ""
        else:
            write_text(p["ocr"], "")
            status = "failed"
            failure_reason = error or "empty_output"
            write_json(p["failed"], {"document_id": doc_id, "error": failure_reason, "raw_response": raw, "content": content})
        log = {
            "timestamp": now(),
            "model": self.args.model,
            "lane": self.args.lane,
            "document_id": doc_id,
            "prompt": prompt,
            "input_image_paths": doc["source_images_ordered"].split(";"),
            "compressed_images": compression_log,
            "runtime_seconds": runtime,
            "gpu_memory_before": gpu_before,
            "gpu_memory_after": gpu_memory(),
            "output_character_count": len(text),
            "non_empty_output": bool(text),
            "image_input_accepted": ok,
            "multi_image_input_count": len(images),
            "status": status,
            "error": failure_reason,
            "raw_output_path": str(p["ocr"]),
            "raw_response_path": str(p["raw"]),
            "usage": {"prompt_eval_count": raw.get("prompt_eval_count"), "eval_count": raw.get("eval_count")} if raw else {},
        }
        write_json(p["log"], log)
        self.handoff_rows.append(self.handoff_row(doc, "available" if status == "success" else "failed", p["ocr"], p["log"]))

    def handoff_row(self, doc: dict[str, str], status: str, ocr_path: Path, log_path: Path) -> dict[str, Any]:
        return {
            "document_id": doc["document_id"],
            "patient_id": doc.get("patient_id", ""),
            "ocr_engine": self.args.lane,
            "ocr_text_path": str(ocr_path),
            "status": status,
            "runtime": "",
            "env_name": "server1_ollama",
            "markdown_path": "",
            "layout_json_path": "",
            "pagewise_text_paths": "",
            "source_csv": "",
            "notes": str(log_path),
        }

    def run(self) -> None:
        for doc in self.docs:
            self.run_one(doc)
        fields = [
            "document_id",
            "patient_id",
            "ocr_engine",
            "ocr_text_path",
            "status",
            "runtime",
            "env_name",
            "markdown_path",
            "layout_json_path",
            "pagewise_text_paths",
            "source_csv",
            "notes",
        ]
        handoff = self.output_root / f"{self.args.lane}_handoff.csv"
        write_csv(handoff, self.handoff_rows, fields)
        nonempty = sum(1 for row in self.handoff_rows if row["status"] == "available")
        lines = [
            f"# {self.args.lane} OCR Run Summary",
            "",
            f"Generated: {now()}",
            "",
            f"- Model: `{self.args.model}`",
            f"- Output root: `{self.output_root}`",
            f"- Handoff CSV: `{handoff}`",
            f"- Records attempted: {len(self.handoff_rows)}",
            f"- Non-empty/available: {nonempty}",
            f"- Non-empty rate: {round(nonempty / max(1, len(self.handoff_rows)), 4)}",
        ]
        write_text(REPORTS_DIR / f"{self.args.report_stem}_summary.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--report-stem", required=True)
    parser.add_argument("--doc-ids", default="")
    parser.add_argument("--prompt", default="Text Recognition: {image_path}")
    parser.add_argument("--max-image-dim", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    Runner(parse_args()).run()


if __name__ == "__main__":
    main()
