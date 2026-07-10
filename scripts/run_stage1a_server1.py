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

"""Final release Stage 1A audit and local smoke runner.

This script is intentionally scoped to Final release:
- conda/environment audit
- manifest verification
- external API dry-run/key checks only
- local Ollama smoke tests

It does not run the full benchmark and does not make external paid API calls.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as md
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAKSH_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DAKSH_ROOT.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_ROOT = REPO_ROOT / "benchmark_outputs"
SERVER_NAME = "server1_4090_ollama"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

load_dotenv(DAKSH_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=False)


MODULE_CHECKS = {
    "torch": "torch",
    "transformers": "transformers",
    "qwen_vl_utils": "qwen_vl_utils",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "pillow": "PIL",
    "opencv": "cv2",
    "pandas": "pandas",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "openai": "openai",
    "requests": "requests",
    "ollama": "ollama",
    "datalab_sdk": "datalab_sdk",
    "zai": "zai",
    "sarvamai": "sarvamai",
    "mistral": "mistralai",
    "deepseek_client": "deepseek",
}


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_prompt(name: str) -> Tuple[str, str]:
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return text, sha256_text(text)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def run_cmd(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:
        return 999, "", f"{type(exc).__name__}: {exc}"


def sort_key(value: Any) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def load_manifest() -> List[Dict[str, str]]:
    path = DATA_DIR / "full_benchmark_manifest.csv"
    if not path.exists():
        code, out, err = run_cmd([sys.executable, str(PROJECT_ROOT / "scripts" / "run_full_benchmark_stage1.py"), "build-manifest"], timeout=60)
        if code != 0:
            raise RuntimeError(f"Manifest missing and build failed: {err or out}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_conda_envs() -> List[Dict[str, str]]:
    code, out, _ = run_cmd(["conda", "env", "list", "--json"], timeout=30)
    if code != 0:
        return []
    data = json.loads(out)
    envs = []
    for prefix in data.get("envs", []):
        envs.append({
            "env_name": "base" if Path(prefix) == Path(data.get("default_prefix", "")) else Path(prefix).name,
            "prefix": prefix,
        })
    return envs


def module_probe_script() -> str:
    return r"""
import importlib, json, sys
mods = {
    "torch": "torch",
    "transformers": "transformers",
    "qwen_vl_utils": "qwen_vl_utils",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "pillow": "PIL",
    "opencv": "cv2",
    "pandas": "pandas",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "openai": "openai",
    "requests": "requests",
    "ollama": "ollama",
    "datalab_sdk": "datalab_sdk",
    "zai": "zai",
    "sarvamai": "sarvamai",
    "mistral": "mistralai",
    "deepseek_client": "deepseek",
}
out = {"python": sys.executable, "python_version": sys.version.split()[0]}
for label, modname in mods.items():
    try:
        mod = importlib.import_module(modname)
        ver = getattr(mod, "__version__", "")
        if not ver:
            try:
                import importlib.metadata as md
                ver = md.version(modname)
            except Exception:
                ver = ""
        out[label + "_installed"] = True
        out[label + "_version"] = ver
    except Exception as exc:
        out[label + "_installed"] = False
        out[label + "_version"] = ""
        out[label + "_error"] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(out))
"""


def audit_environments() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    command_outputs: Dict[str, Any] = {}
    commands = {
        "conda_env_list": ["conda", "env", "list"],
        "which_python": ["which", "python"],
        "python_version": ["python", "--version"],
        "pip_list": ["python", "-m", "pip", "list"],
        "nvidia_smi": ["nvidia-smi"],
        "ollama_list": ["ollama", "list"],
        "disk_space": ["df", "-h", str(REPO_ROOT)],
    }
    for name, cmd in commands.items():
        code, out, err = run_cmd(cmd, timeout=90 if name in {"pip_list", "nvidia_smi"} else 30)
        command_outputs[name] = {"command": cmd, "returncode": code, "stdout": out, "stderr": err}
    write_text(REPORTS_DIR / "stage1a_server1_pip_list.txt", command_outputs["pip_list"]["stdout"] + "\n")

    env_rows: List[Dict[str, Any]] = []
    for env in get_conda_envs():
        python_bin = Path(env["prefix"]) / "bin" / "python"
        row: Dict[str, Any] = {"env_name": env["env_name"], "prefix": env["prefix"], "python": str(python_bin)}
        if python_bin.exists():
            code, out, err = run_cmd([str(python_bin), "-c", module_probe_script()], timeout=45)
            if code == 0:
                row.update(json.loads(out))
            else:
                row["probe_error"] = err or out
        else:
            row["probe_error"] = "python binary not found"
        env_rows.append(row)

    fields = ["env_name", "prefix", "python", "python_version", "probe_error"]
    for label in MODULE_CHECKS:
        fields.extend([f"{label}_installed", f"{label}_version", f"{label}_error"])
    write_csv(REPORTS_DIR / "stage1a_server1_conda_env_matrix.csv", env_rows, fields)

    keys = {
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
        "ZAI_API_KEY": bool(os.getenv("ZAI_API_KEY")),
        "DATALAB_API_KEY": bool(os.getenv("DATALAB_API_KEY")),
        "MISTRAL_API_KEY": bool(os.getenv("MISTRAL_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
        "SARVAM_API_KEY": bool(os.getenv("SARVAM_API_KEY")),
    }
    inventory = {
        "generated": now(),
        "server_name": SERVER_NAME,
        "repo_path": str(PROJECT_ROOT),
        "ollama_host": OLLAMA_HOST,
        "commands": command_outputs,
        "key_presence": keys,
        "conda_env_count": len(env_rows),
        "conda_envs": env_rows,
    }
    write_json(REPORTS_DIR / "stage1a_server1_environment_inventory.json", inventory)

    md = [
        "# Stage 1A Final release Environment Audit",
        "",
        f"Generated: {now()}",
        "",
        f"- Server role: `{SERVER_NAME}`",
        f"- Repo path: `{PROJECT_ROOT}`",
        f"- Ollama host: `{OLLAMA_HOST}`",
        f"- Conda environments discovered: {len(env_rows)}",
        f"- Current python: `{command_outputs['which_python']['stdout']}`",
        f"- Current python version: `{command_outputs['python_version']['stdout'] or command_outputs['python_version']['stderr']}`",
        "",
        "## Key Presence",
        "",
    ]
    md.extend([f"- `{k}`: {'present' if v else 'missing'}" for k, v in keys.items()])
    md.extend([
        "",
        "## Ollama Models",
        "",
        "```",
        command_outputs["ollama_list"]["stdout"] or command_outputs["ollama_list"]["stderr"],
        "```",
        "",
        "## GPU",
        "",
        "```",
        command_outputs["nvidia_smi"]["stdout"] or command_outputs["nvidia_smi"]["stderr"],
        "```",
        "",
        "## Disk",
        "",
        "```",
        command_outputs["disk_space"]["stdout"] or command_outputs["disk_space"]["stderr"],
        "```",
        "",
        "The full current-environment `pip list` is saved at `reports/stage1a_server1_pip_list.txt`.",
    ])
    write_text(REPORTS_DIR / "stage1a_server1_environment_audit.md", "\n".join(md) + "\n")


def manifest_check() -> None:
    rows = load_manifest()
    image_count = sum(len([p for p in r["source_images_ordered"].split(";") if p]) for r in rows)
    patients = {r["patient_id"] for r in rows}
    p37 = next((r for r in rows if r["document_id"] == "p37_1"), {})
    p42 = next((r for r in rows if r["document_id"] == "p42_1"), {})
    ordering_issues = [r["document_id"] for r in rows if int(r.get("num_source_images") or 0) != len([p for p in r["source_images_ordered"].split(";") if p])]
    md = [
        "# Stage 1A Final release Manifest Check",
        "",
        f"Generated: {now()}",
        "",
        f"- Manifest path: `{DATA_DIR / 'full_benchmark_manifest.csv'}`",
        f"- Annotated document records: {len(rows)}",
        f"- Patient roots: {len(patients)}",
        f"- Source images: {image_count}",
        f"- Image ordering count mismatches: {len(ordering_issues)}",
        "",
        "## Multi-View Bundle Checks",
        "",
        f"- `p37_1`: total_pages_gt={p37.get('total_pages_gt')}, num_source_images={p37.get('num_source_images')}, image_bundle_type={p37.get('image_bundle_type')}, roles={p37.get('image_roles')}",
        f"- `p42_1`: total_pages_gt={p42.get('total_pages_gt')}, num_source_images={p42.get('num_source_images')}, image_bundle_type={p42.get('image_bundle_type')}, roles={p42.get('image_roles')}",
    ]
    if ordering_issues:
        md.extend(["", "Ordering/count mismatches:", *[f"- `{x}`" for x in ordering_issues]])
    write_text(REPORTS_DIR / "stage1a_server1_manifest_check.md", "\n".join(md) + "\n")


def package_images(doc: Dict[str, str], run_dir: Path, max_width: int = 1800) -> Tuple[List[Path], List[Dict[str, Any]]]:
    out_dir = run_dir / "compressed_images" / doc["document_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [PROJECT_ROOT / p for p in doc["source_images_ordered"].split(";") if p]
    roles = doc.get("image_roles", "").split(";") if doc.get("image_roles") else []
    out_paths: List[Path] = []
    logs: List[Dict[str, Any]] = []
    for idx, src in enumerate(paths):
        role = roles[idx] if idx < len(roles) else f"page_{idx + 1}"
        dst = out_dir / f"{idx + 1:02d}_{role}.jpg"
        with Image.open(src) as im:
            original_size = im.size
            im = im.convert("RGB")
            if im.width > max_width:
                scale = max_width / im.width
                im = im.resize((max_width, max(1, int(im.height * scale))))
            im.save(dst, format="JPEG", quality=92, optimize=True)
            compressed_size = im.size
        out_paths.append(dst)
        logs.append({
            "source": rel(src),
            "compressed": rel(dst),
            "role": role,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "quality": 92,
            "max_width": max_width,
            "bytes": dst.stat().st_size,
        })
    return out_paths, logs


def create_contact_sheet(run_dir: Path, docs: List[Dict[str, str]]) -> None:
    thumbs = []
    for doc in docs:
        for rel_path in doc["source_images_ordered"].split(";"):
            if not rel_path:
                continue
            src = PROJECT_ROOT / rel_path
            with Image.open(src) as im:
                im.thumbnail((320, 240))
                thumb = Image.new("RGB", (340, 280), "white")
                thumb.paste(im.convert("RGB"), (10, 10))
                draw = ImageDraw.Draw(thumb)
                draw.text((10, 252), f"{doc['document_id']} {src.name[:32]}", fill="black")
                thumbs.append(thumb)
    if not thumbs:
        return
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 340, rows * 280), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 340, (idx // cols) * 280))
    out = run_dir / "qc_contact_sheets" / "server1_smoke_contact_sheet.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def gpu_memory() -> str:
    code, out, err = run_cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], timeout=10)
    return out if code == 0 else err


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
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def ollama_chat(model: str, prompt: str, images: List[Path], timeout: int = 600) -> Tuple[bool, str, Any, Dict[str, Any], str]:
    message: Dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        message["images"] = [encode_image(p) for p in images]
    payload = {"model": model, "messages": [message], "stream": False, "options": {"temperature": 0.0}}
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        if resp.status_code >= 400:
            return False, "", resp.text, {}, f"HTTP {resp.status_code}: {resp.text[:500]}"
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        usage = {"prompt_eval_count": data.get("prompt_eval_count"), "eval_count": data.get("eval_count")}
        return True, content, data, usage, ""
    except Exception as exc:
        return False, "", None, {}, f"{type(exc).__name__}: {exc}"


def find_ocr_text(doc_id: str) -> Optional[Path]:
    candidates = sorted(OUTPUT_ROOT.glob(f"stage1a_smoke_*/raw_ocr/datalab_chandra/{doc_id}.md"), reverse=True)
    return candidates[0] if candidates and candidates[0].exists() and candidates[0].stat().st_size > 0 else None


def select_docs(manifest: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_id = {r["document_id"]: r for r in manifest}
    return [by_id[i] for i in ["p4", "p25_1", "p38_1"] if i in by_id]


def run_one(
    run_dir: Path,
    backend: str,
    model: str,
    track: str,
    doc: Dict[str, str],
    prompt_file: str,
    prompt: str,
    images: List[Path],
    compression_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt_hash = sha256_text((PROMPTS_DIR / prompt_file).read_text(encoding="utf-8"))
    log_path = run_dir / "logs" / backend / f"{doc['document_id']}.json"
    raw_path = run_dir / "raw_responses" / backend / f"{doc['document_id']}.txt"
    parsed_dir = run_dir / ("semantic_inference" if track.startswith("semantic") else "raw_structured") / backend
    parsed_path = parsed_dir / f"{doc['document_id']}.json"
    failed_path = run_dir / "failed_cases" / backend / f"{doc['document_id']}.json"
    start_ts = now()
    gpu_before = gpu_memory()
    start = time.time()
    ok, content, raw, usage, error = ollama_chat(model, prompt, images)
    runtime = round(time.time() - start, 3)
    gpu_after = gpu_memory()
    parsed = extract_json(content)
    write_text(raw_path, content or json.dumps(raw, ensure_ascii=False))
    if parsed:
        write_json(parsed_path, parsed)
    else:
        write_json(failed_path, {"error": error or "JSON parse failed", "content": content})
    log = {
        "server_name": SERVER_NAME,
        "backend": backend,
        "model": model,
        "model_version": model,
        "track": track,
        "document_id": doc["document_id"],
        "patient_id": doc["patient_id"],
        "prompt_file": prompt_file,
        "prompt_hash": prompt_hash,
        "input_images": doc["source_images_ordered"].split(";"),
        "compressed_images": [rel(p) for p in images],
        "compression_settings": compression_log,
        "start_timestamp": start_ts,
        "end_timestamp": now(),
        "runtime_seconds": runtime,
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_after,
        "parse_success": parsed is not None,
        "schema_validation_success": parsed is not None and ("raw_entities" in parsed or parsed.get("schema_version") == "raw_rx_v2" or track.startswith("semantic")),
        "retry_count": 0,
        "error": error if not parsed else "",
        "usage": usage,
        "raw_response_path": rel(raw_path),
        "parsed_response_path": rel(parsed_path) if parsed else "",
        "failed_case_path": rel(failed_path) if not parsed else "",
    }
    write_json(log_path, log)
    return log


def compute_metric(doc: Dict[str, str], backend: str, track: str, log: Dict[str, Any], parsed_path: Path) -> Dict[str, Any]:
    row = {
        "document_id": doc["document_id"],
        "backend": backend,
        "track": track,
        "json_parse_success": int(bool(log.get("parse_success"))),
        "schema_validity": int(bool(log.get("schema_validation_success"))),
        "output_completeness": 0.0,
        "required_field_coverage": 0.0,
        "scalar_accuracy_exact": "",
        "scalar_accuracy_lenient": "",
        "entity_lenient_f1": "",
        "hallucination_rate": "",
        "missing_entity_rate": "",
        "runtime_seconds": log.get("runtime_seconds", ""),
        "notes": "",
    }
    if not parsed_path.exists():
        row["notes"] = log.get("error", "No parsed JSON")
        return row
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.run_full_benchmark_stage1 import compute_smoke_metrics

        source_row = dict(doc)
        source_row["ground_truth_json"] = doc["ground_truth_json"]
        nested_log = parsed_path.with_suffix(".metric_log.json")
        write_json(nested_log, {"runtime_seconds": log.get("runtime_seconds"), "estimated_cost": None})
        raw = compute_smoke_metrics(source_row, backend, track, parsed_path, nested_log, None)
        row.update({
            "output_completeness": raw.get("output_completeness", 0.0),
            "required_field_coverage": raw.get("field_coverage", 0.0),
            "scalar_accuracy_exact": raw.get("scalar_accuracy_exact", ""),
            "scalar_accuracy_lenient": raw.get("scalar_accuracy_lenient", ""),
            "entity_lenient_f1": raw.get("entity_lenient_f1", ""),
            "hallucination_rate": raw.get("hallucination_rate", ""),
            "missing_entity_rate": raw.get("missing_entity_rate", ""),
            "notes": raw.get("notes", ""),
        })
    except Exception as exc:
        row["notes"] = f"Metric calculation failed: {type(exc).__name__}: {exc}"
    return row


def run_smoke() -> Path:
    manifest = load_manifest()
    docs = select_docs(manifest)
    run_dir = OUTPUT_ROOT / f"stage1a_server1_smoke_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    create_contact_sheet(run_dir, docs)
    metrics: List[Dict[str, Any]] = []
    runtimes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for doc in docs:
        images, comp = package_images(doc, run_dir)
        prompt_base, _ = read_prompt("raw_structured_extraction_prompt.txt")
        prompt = f"Document ID: {doc['document_id']}\nImage roles: {doc['image_roles']}\n\n{prompt_base}"
        for backend, model in [
            ("ollama_qwen3_vl_8b_raw_structured", "qwen3-vl:8b-instruct"),
            ("ollama_llava_13b_raw_structured", "llava:13b"),
        ]:
            log = run_one(run_dir, backend, model, "raw_structured", doc, "raw_structured_extraction_prompt.txt", prompt, images, comp)
            parsed_path = run_dir / "raw_structured" / backend / f"{doc['document_id']}.json"
            metrics.append(compute_metric(doc, backend, "raw_structured", log, parsed_path))
            runtimes.append(runtime_row(log))
            if not log.get("parse_success"):
                failures.append(failure_row(log))

    for doc in docs:
        ocr_path = find_ocr_text(doc["document_id"])
        if not ocr_path:
            for backend in ["ollama_qwen25_14b_ocr_to_json", "ollama_qwen3_8b_ocr_to_json"]:
                failures.append({"backend": backend, "document_id": doc["document_id"], "error": "OCR text not available; skipped per Final release instruction"})
            continue
        ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore")
        prompt_base, _ = read_prompt("ocr_to_json_structuring_prompt.txt")
        prompt = f"Document ID: {doc['document_id']}\nOCR source: {rel(ocr_path)}\n\n{prompt_base}\n\nOCR TEXT START\n{ocr_text}\nOCR TEXT END"
        for backend, model in [
            ("ollama_qwen25_14b_ocr_to_json", "qwen2.5:14b"),
            ("ollama_qwen3_8b_ocr_to_json", "qwen3:8b"),
        ]:
            log = run_one(run_dir, backend, model, "ocr_to_json", doc, "ocr_to_json_structuring_prompt.txt", prompt, [], [])
            parsed_path = run_dir / "raw_structured" / backend / f"{doc['document_id']}.json"
            metrics.append(compute_metric(doc, backend, "ocr_to_json", log, parsed_path))
            runtimes.append(runtime_row(log))
            if not log.get("parse_success"):
                failures.append(failure_row(log))

    semantic_doc = next((d for d in docs if d["document_id"] == "p25_1"), docs[0])
    ocr_path = find_ocr_text(semantic_doc["document_id"])
    if ocr_path:
        ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore")
        prompt_base, _ = read_prompt("semantic_inference_extraction_prompt.txt")
        prompt = f"Document ID: {semantic_doc['document_id']}\nOCR source: {rel(ocr_path)}\n\n{prompt_base}\n\nOCR TEXT START\n{ocr_text}\nOCR TEXT END"
        for backend, model in [
            ("ollama_qwen25_14b_semantic_inference", "qwen2.5:14b"),
            ("ollama_qwen3_8b_semantic_inference", "qwen3:8b"),
        ]:
            log = run_one(run_dir, backend, model, "semantic_inference", semantic_doc, "semantic_inference_extraction_prompt.txt", prompt, [], [])
            parsed_path = run_dir / "semantic_inference" / backend / f"{semantic_doc['document_id']}.json"
            metrics.append(compute_metric(semantic_doc, backend, "semantic_inference", log, parsed_path))
            runtimes.append(runtime_row(log))
            if not log.get("parse_success"):
                failures.append(failure_row(log))

    metric_fields = ["document_id", "backend", "track", "json_parse_success", "schema_validity", "output_completeness", "required_field_coverage", "scalar_accuracy_exact", "scalar_accuracy_lenient", "entity_lenient_f1", "hallucination_rate", "missing_entity_rate", "runtime_seconds", "notes"]
    write_csv(REPORTS_DIR / "stage1a_server1_smoke_metrics.csv", metrics, metric_fields)
    write_text(REPORTS_DIR / "stage1a_server1_failure_log.md", render_failure_log(failures))
    write_text(REPORTS_DIR / "stage1a_server1_cost_runtime_report.md", render_runtime_report(runtimes))
    write_model_availability(metrics)
    write_adapter_status(metrics, failures)
    write_smoke_summary(run_dir, docs, metrics, failures)
    return run_dir


def runtime_row(log: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": log.get("document_id"),
        "backend": log.get("backend"),
        "model": log.get("model"),
        "track": log.get("track"),
        "runtime_seconds": log.get("runtime_seconds"),
        "status": "success" if log.get("parse_success") else "failed",
        "token_usage": json.dumps(log.get("usage"), ensure_ascii=False),
        "gpu_memory_before": log.get("gpu_memory_before"),
        "gpu_memory_after": log.get("gpu_memory_after"),
    }


def failure_row(log: Dict[str, Any]) -> Dict[str, Any]:
    return {"backend": log.get("backend"), "document_id": log.get("document_id"), "error": log.get("error") or "parse/schema failure"}


def render_failure_log(failures: List[Dict[str, Any]]) -> str:
    lines = ["# Stage 1A Final release Failure Log", "", f"Generated: {now()}", ""]
    if not failures:
        lines.append("No Final release smoke failures recorded.")
    else:
        lines.extend([f"- `{f.get('backend')}` `{f.get('document_id')}`: {f.get('error')}" for f in failures])
    return "\n".join(lines) + "\n"


def render_runtime_report(rows: List[Dict[str, Any]]) -> str:
    lines = ["# Stage 1A Final release Cost/Runtime Report", "", f"Generated: {now()}", "", "No external paid calls were made by this Final release run.", ""]
    for r in rows:
        lines.append(f"- `{r['backend']}` `{r['document_id']}` `{r['track']}`: {r['runtime_seconds']}s, {r['status']}, GPU before/after `{r['gpu_memory_before']}` -> `{r['gpu_memory_after']}`")
    return "\n".join(lines) + "\n"


def write_model_availability(metrics: List[Dict[str, Any]]) -> None:
    ollama_code, ollama_out, ollama_err = run_cmd(["ollama", "list"], timeout=20)
    present_models = ollama_out if ollama_code == 0 else ollama_err
    backends = [
        ("ollama_qwen3_vl_8b_raw_structured", "qwen3-vl:8b-instruct", "multimodal"),
        ("ollama_llava_13b_raw_structured", "llava:13b", "multimodal"),
        ("ollama_qwen25_14b_ocr_to_json", "qwen2.5:14b", "text"),
        ("ollama_qwen3_8b_ocr_to_json", "qwen3:8b", "text"),
        ("ollama_qwen25_14b_semantic_inference", "qwen2.5:14b", "text"),
        ("ollama_qwen3_8b_semantic_inference", "qwen3:8b", "text"),
    ]
    rows = []
    for backend, model, modality in backends:
        subset = [m for m in metrics if m["backend"] == backend]
        rows.append({
            "backend": backend,
            "model": model,
            "modality": modality,
            "installed": str(model in present_models).lower(),
            "smoke_test_status": "passed_smoke" if subset and all(m["json_parse_success"] for m in subset) else ("failed_or_partial" if subset else "not_run"),
            "notes": "",
        })
    write_csv(REPORTS_DIR / "stage1a_server1_model_availability.csv", rows, ["backend", "model", "modality", "installed", "smoke_test_status", "notes"])


def write_adapter_status(metrics: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    lines = ["# Stage 1A Final release Adapter Status", "", f"Generated: {now()}", ""]
    for backend in sorted({m["backend"] for m in metrics} | {f.get("backend", "") for f in failures}):
        if not backend:
            continue
        subset = [m for m in metrics if m["backend"] == backend]
        failed = [f for f in failures if f.get("backend") == backend]
        if subset:
            passed = sum(1 for m in subset if m["json_parse_success"])
            lines.append(f"- `{backend}`: {passed}/{len(subset)} smoke outputs parsed as JSON.")
        for f in failed:
            lines.append(f"- `{backend}` failure on `{f.get('document_id')}`: {f.get('error')}")
    lines.extend(["", "External API adapters are dry-run/key-check only in this Final release pass; no paid external calls were made."])
    write_text(REPORTS_DIR / "stage1a_server1_adapter_status.md", "\n".join(lines) + "\n")


def write_smoke_summary(run_dir: Path, docs: List[Dict[str, str]], metrics: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    lines = [
        "# Stage 1A Final release Smoke Summary",
        "",
        f"Generated: {now()}",
        "",
        f"- Server: `{SERVER_NAME}`",
        f"- Smoke output directory: `{run_dir}`",
        f"- Documents: {', '.join(d['document_id'] for d in docs)}",
        f"- Smoke metric rows: {len(metrics)}",
        f"- Failures recorded: {len(failures)}",
        "- External paid calls made by this Final release run: none.",
        "",
        "## Results",
        "",
    ]
    for backend in sorted({m["backend"] for m in metrics}):
        subset = [m for m in metrics if m["backend"] == backend]
        passed = sum(1 for m in subset if m["json_parse_success"])
        lines.append(f"- `{backend}`: {passed}/{len(subset)} JSON parse success.")
    write_text(REPORTS_DIR / "stage1a_server1_smoke_summary.md", "\n".join(lines) + "\n")


def external_api_dry_run_report() -> None:
    rows = []
    apis = [
        ("zai_glm_ocr", "ZAI_API_KEY", "zai", "implemented_dry_run", "exclude_until_balance_or_resource_package_fixed"),
        ("datalab_chandra", "DATALAB_API_KEY", "datalab_sdk", "implemented_dry_run", "include_for_approved_tiny_smoke_or_full_after_approval"),
        ("mistral_ocr", "MISTRAL_API_KEY", "mistralai", "not_implemented_in_repo", "exclude_until_key_and_adapter_added"),
        ("deepseek_api", "DEEPSEEK_API_KEY", "openai", "openai_compatible_dry_run_possible", "exclude_until_explicit_approval"),
        ("sarvam_document_intelligence", "SARVAM_API_KEY", "sarvamai", "implemented_dry_run", "include_for_approved_tiny_smoke_or_full_after_approval"),
        ("openrouter", "OPENROUTER_API_KEY", "openai", "disabled_by_policy", "exclude_low_credits"),
    ]
    for api, key, module, dry, decision in apis:
        try:
            __import__(module)
            installed = True
        except Exception:
            installed = False
        version = ""
        if installed:
            for dist in [module, {"PIL": "Pillow", "mistralai": "mistralai", "datalab_sdk": "datalab-python-sdk"}.get(module, module)]:
                try:
                    version = md.version(dist)
                    break
                except Exception:
                    continue
        if not installed:
            version = ""
        rows.append({
            "api": api,
            "key_present": str(bool(os.getenv(key))).lower(),
            "sdk_client_installed": str(installed).lower(),
            "sdk_client_version": version,
            "dry_run_adapter_implemented": dry,
            "paid_call_made": "false",
            "estimated_smoke_cost": "no paid call; estimate required from provider docs/account before approval",
            "recommended_decision": decision,
        })
    fields = ["api", "key_present", "sdk_client_installed", "sdk_client_version", "dry_run_adapter_implemented", "paid_call_made", "estimated_smoke_cost", "recommended_decision"]
    write_csv(REPORTS_DIR / "stage1a_server1_external_api_key_matrix.csv", rows, fields)
    md = ["# Stage 1A Final release External API Adapter Status", "", f"Generated: {now()}", "", "No paid external API calls were made by this Final release dry-run pass.", ""]
    for row in rows:
        md.append(f"- `{row['api']}`: key_present={row['key_present']}, sdk/client={row['sdk_client_installed']} {row['sdk_client_version']}, dry_run={row['dry_run_adapter_implemented']}, decision={row['recommended_decision']}.")
    write_text(REPORTS_DIR / "stage1a_server1_external_api_adapter_status.md", "\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1A Final release audit/smoke runner")
    parser.add_argument("command", choices=["audit", "manifest-check", "external-dry-run", "smoke", "all"])
    args = parser.parse_args()
    if args.command in {"audit", "all"}:
        audit_environments()
    if args.command in {"manifest-check", "all"}:
        manifest_check()
    if args.command in {"external-dry-run", "all"}:
        external_api_dry_run_report()
    if args.command in {"smoke", "all"}:
        run_dir = run_smoke()
        print(f"final smoke outputs: {run_dir}")


if __name__ == "__main__":
    main()
