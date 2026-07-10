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

"""Final release Stage 1A Ollama OCR addendum.

This addendum audits pullable Ollama OCR/document models without pulling them.
If glm-ocr is already installed it can run the approved three-document smoke;
otherwise it writes pull-approval-needed logs and stops.
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
DAKSH_ROOT = PROJECT_ROOT.parent
REPO_ROOT = DAKSH_ROOT.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "benchmark_outputs"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
SERVER_NAME = "server1_4090_ollama"
DOC_IDS = ["p4", "p25_1", "p38_1"]
GLM_TEXT_PROMPT_TEMPLATE = "Text Recognition: {image_path}"
GLM_STRUCTURED_PROMPT = "Extract all visible text from this prescription. Preserve line order, handwriting, abbreviations, and uncertainty. Do not normalize medical terms."


OLLAMA_LIBRARY_MODELS = [
    {
        "model": "glm-ocr:latest",
        "family": "glm-ocr",
        "expected_size": "2.2GB",
        "context": "128K",
        "input": "Text, Image",
        "multimodal_image_support": "yes",
        "gpu_vram_expectation": "Should fit RTX 4090 24GB; Ollama library package is 2.2GB and model is ~0.9B/1B params.",
        "license_source": "MIT on Hugging Face model card; Ollama library lists open-source SDK/toolchain.",
        "source_url": "https://ollama.com/library/glm-ocr",
        "decision_label": "available_needs_pull_approval",
    },
    {
        "model": "glm-ocr:q8_0",
        "family": "glm-ocr",
        "expected_size": "1.6GB",
        "context": "128K",
        "input": "Text, Image",
        "multimodal_image_support": "yes",
        "gpu_vram_expectation": "Should fit RTX 4090 24GB.",
        "license_source": "MIT on Hugging Face model card.",
        "source_url": "https://ollama.com/library/glm-ocr/tags",
        "decision_label": "available_needs_pull_approval",
    },
    {
        "model": "deepseek-ocr:latest",
        "family": "deepseek-ocr",
        "expected_size": "6.7GB",
        "context": "8K",
        "input": "Text, Image",
        "multimodal_image_support": "yes",
        "gpu_vram_expectation": "Likely fits RTX 4090 24GB, but larger than glm-ocr and requires Ollama v0.13.0+.",
        "license_source": "See Ollama/GitHub upstream before use.",
        "source_url": "https://ollama.com/library/deepseek-ocr",
        "decision_label": "available_needs_pull_approval",
    },
    {
        "model": "moondream:latest",
        "family": "moondream",
        "expected_size": "1.7GB",
        "context": "2K",
        "input": "Text, Image",
        "multimodal_image_support": "yes",
        "gpu_vram_expectation": "Should fit RTX 4090 24GB; small VLM, not OCR-specific.",
        "license_source": "See Ollama/GitHub/Hugging Face upstream before use.",
        "source_url": "https://ollama.com/library/moondream",
        "decision_label": "available_needs_pull_approval",
    },
]


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


def run_cmd(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:
        return 999, "", f"{type(exc).__name__}: {exc}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def load_manifest() -> List[Dict[str, str]]:
    path = DATA_DIR / "full_benchmark_manifest.csv"
    if not path.exists():
        code, out, err = run_cmd([sys.executable, str(PROJECT_ROOT / "scripts" / "run_full_benchmark_stage1.py"), "build-manifest"], timeout=60)
        if code != 0:
            raise RuntimeError(f"manifest build failed: {err or out}")
    return read_csv(path)


def selected_docs() -> List[Dict[str, str]]:
    by_id = {row["document_id"]: row for row in load_manifest()}
    return [by_id[doc_id] for doc_id in DOC_IDS if doc_id in by_id]


def installed_models() -> Tuple[str, List[str]]:
    code, out, err = run_cmd(["ollama", "list"], timeout=20)
    text = out if code == 0 else err
    names = []
    for line in text.splitlines()[1:]:
        if line.strip():
            names.append(line.split()[0])
    return text, names


def show_model(model: str) -> Dict[str, Any]:
    code, out, err = run_cmd(["ollama", "show", model], timeout=20)
    return {"model": model, "returncode": code, "stdout": out, "stderr": err}


def gpu_memory() -> str:
    code, out, err = run_cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], timeout=10)
    return out if code == 0 else err


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def package_images(doc: Dict[str, str], run_dir: Path) -> List[Path]:
    out_dir = run_dir / "compressed_images" / doc["document_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for idx, src_rel in enumerate([p for p in doc["source_images_ordered"].split(";") if p]):
        src = PROJECT_ROOT / src_rel
        dst = out_dir / f"{idx + 1:02d}_{Path(src_rel).stem}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            if im.width > 1800:
                scale = 1800 / im.width
                im = im.resize((1800, max(1, int(im.height * scale))))
            im.save(dst, format="JPEG", quality=92, optimize=True)
        out.append(dst)
    return out


def ollama_generate(model: str, prompt: str, image: Path) -> Tuple[bool, str, Any, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(image)],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=300)
        if resp.status_code >= 400:
            return False, "", resp.text, f"HTTP {resp.status_code}: {resp.text[:500]}"
        data = resp.json()
        return True, data.get("response", ""), data, ""
    except Exception as exc:
        return False, "", None, f"{type(exc).__name__}: {exc}"


def flatten_gt_text(doc: Dict[str, str]) -> str:
    gt_path = PROJECT_ROOT / doc["ground_truth_json"]
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
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


def key_field_proxy(output: str, gt: str) -> Dict[str, float]:
    text = output.lower()
    gt_text = gt.lower()
    buckets = {
        "patient_name": ["name", "patient", "devki", "alivelatamma"],
        "date": ["date", "2024", "2023", "jul", "jun"],
        "medication_names": ["tablet", "tab", "rx", "metformin", "gabapen", "pantop", "bep"],
        "vitals": ["bp", "pulse", "weight", "height", "bmi"],
        "complaints_diagnosis": ["pain", "diagnosis", "c/o", "complaint", "burning"],
    }
    scores = {}
    for key, terms in buckets.items():
        relevant = [t for t in terms if t in gt_text]
        if not relevant:
            scores[key] = 0.0
        else:
            scores[key] = round(sum(1 for t in relevant if t in text) / len(relevant), 4)
    return scores


def run_glm_smoke(run_dir: Path, docs: List[Dict[str, str]], model: str = "glm-ocr:latest") -> List[Dict[str, Any]]:
    rows = []
    for doc in docs:
        images = package_images(doc, run_dir)
        outputs = []
        raw_chunks = []
        errors = []
        start = time.time()
        gpu_before = gpu_memory()
        for image in images:
            prompt = GLM_TEXT_PROMPT_TEMPLATE.format(image_path=str(image))
            ok, content, raw, error = ollama_generate(model, prompt, image)
            if ok:
                outputs.append(content)
            else:
                errors.append(error)
            raw_chunks.append({"prompt": prompt, "image": rel(image), "ok": ok, "content": content, "raw": raw, "error": error})
        structured_ok, structured_content, structured_raw, structured_error = ollama_generate(model, GLM_STRUCTURED_PROMPT, images[0]) if images else (False, "", None, "no image")
        raw_chunks.append({"prompt": GLM_STRUCTURED_PROMPT, "image": rel(images[0]) if images else "", "ok": structured_ok, "content": structured_content, "raw": structured_raw, "error": structured_error})
        runtime = round(time.time() - start, 3)
        gpu_after = gpu_memory()
        content = "\n\n".join(outputs)
        raw_path = run_dir / "raw_responses" / "ollama_glm_ocr" / f"{doc['document_id']}.txt"
        out_path = run_dir / "raw_ocr" / "ollama_glm_ocr" / f"{doc['document_id']}.txt"
        log_path = run_dir / "logs" / "ollama_glm_ocr" / f"{doc['document_id']}.json"
        fail_path = run_dir / "failed_cases" / "ollama_glm_ocr" / f"{doc['document_id']}.json"
        write_text(raw_path, json.dumps(raw_chunks, indent=2, ensure_ascii=False))
        write_text(out_path, content)
        success = bool(content.strip()) and not errors
        gt = flatten_gt_text(doc)
        similarity = round(difflib.SequenceMatcher(None, content.lower(), gt.lower()).ratio(), 4) if content and gt else 0.0
        proxy = key_field_proxy(content, gt)
        log = {
            "server_name": SERVER_NAME,
            "command_used": f"POST {OLLAMA_HOST}/api/generate",
            "model_name": model,
            "model_size": "2.2GB expected from Ollama library",
            "prompts": [GLM_TEXT_PROMPT_TEMPLATE, GLM_STRUCTURED_PROMPT],
            "image_paths": [rel(p) for p in images],
            "runtime_seconds": runtime,
            "gpu_memory_before": gpu_before,
            "gpu_memory_after": gpu_after,
            "parse_output_success": success,
            "failure_reason": "; ".join(errors) if errors else "",
            "raw_response_path": rel(raw_path),
            "raw_ocr_path": rel(out_path),
            "non_empty_output": bool(content.strip()),
            "approx_text_similarity_to_gt": similarity,
            "key_field_recall_proxy": proxy,
        }
        write_json(log_path, log)
        if not success:
            write_json(fail_path, {"failure_reason": log["failure_reason"] or "empty output", "log": log})
        rows.append({"document_id": doc["document_id"], "non_empty_output": int(bool(content.strip())), "approx_text_similarity_to_gt": similarity, **proxy, "runtime_seconds": runtime, "decision_label": "ready_for_full_run" if success else "failed_smoke"})
    return rows


def write_not_installed_logs(run_dir: Path, docs: List[Dict[str, str]], show: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for doc in docs:
        image_paths = [PROJECT_ROOT / p for p in doc["source_images_ordered"].split(";") if p]
        log = {
            "server_name": SERVER_NAME,
            "command_used": "ollama show glm-ocr; no pull/run performed",
            "model_name": "glm-ocr:latest",
            "model_size": "2.2GB expected from Ollama library",
            "prompt": GLM_TEXT_PROMPT_TEMPLATE,
            "structured_prompt": GLM_STRUCTURED_PROMPT,
            "image_paths": [rel(p) for p in image_paths],
            "runtime_seconds": 0,
            "gpu_memory_before": gpu_memory(),
            "gpu_memory_after": gpu_memory(),
            "parse_output_success": False,
            "failure_reason": "glm-ocr is available in the Ollama library but is not installed locally; pull approval required before smoke.",
            "ollama_show": show,
            "decision_label": "available_needs_pull_approval",
        }
        write_text(run_dir / "raw_ocr" / "ollama_glm_ocr" / f"{doc['document_id']}.txt", "")
        write_text(run_dir / "raw_responses" / "ollama_glm_ocr" / f"{doc['document_id']}.txt", json.dumps(log, indent=2, ensure_ascii=False))
        write_json(run_dir / "logs" / "ollama_glm_ocr" / f"{doc['document_id']}.json", log)
        write_json(run_dir / "failed_cases" / "ollama_glm_ocr" / f"{doc['document_id']}.json", {"failure_reason": log["failure_reason"], "log": log})
        rows.append({
            "document_id": doc["document_id"],
            "non_empty_output": 0,
            "approx_text_similarity_to_gt": 0.0,
            "patient_name": 0.0,
            "date": 0.0,
            "medication_names": 0.0,
            "vitals": 0.0,
            "complaints_diagnosis": 0.0,
            "runtime_seconds": 0,
            "decision_label": "available_needs_pull_approval",
        })
    return rows


def update_model_availability(installed: List[str], show_results: Dict[str, Dict[str, Any]], glm_decision: str) -> None:
    path = REPORTS_DIR / "stage1a_server1_model_availability.csv"
    rows = read_csv(path)
    existing = {r.get("backend"): r for r in rows}
    additions = [
        ("ollama_glm_ocr", "glm-ocr:latest", "ocr_vlm", glm_decision),
        ("ollama_deepseek_ocr", "deepseek-ocr:latest", "ocr_vlm", "available_needs_pull_approval"),
        ("ollama_moondream", "moondream:latest", "vlm", "available_needs_pull_approval"),
        ("ollama_qwen3_vl_local", "qwen3-vl:8b-instruct", "vlm", "ready_for_full_run"),
        ("ollama_llava_local", "llava:13b", "vlm", "smoke_passed_but_low_quality"),
    ]
    for backend, model, modality, decision in additions:
        existing[backend] = {
            "backend": backend,
            "model": model,
            "modality": modality,
            "installed": str(model in installed or model.split(":")[0] in installed).lower(),
            "smoke_test_status": "not_run_pull_required" if decision == "available_needs_pull_approval" else decision,
            "notes": f"addendum_decision={decision}; ollama_show={show_results.get(model.split(':')[0], {}).get('stderr') or 'ok'}",
        }
    fields = ["backend", "model", "modality", "installed", "smoke_test_status", "notes"]
    write_csv(path, list(existing.values()), fields)


def append_smoke_metrics(rows: List[Dict[str, Any]]) -> None:
    path = REPORTS_DIR / "stage1a_server1_smoke_metrics.csv"
    existing = read_csv(path)
    fields = ["document_id", "backend", "track", "json_parse_success", "schema_validity", "output_completeness", "required_field_coverage", "scalar_accuracy_exact", "scalar_accuracy_lenient", "entity_lenient_f1", "hallucination_rate", "missing_entity_rate", "runtime_seconds", "notes"]
    existing = [r for r in existing if not (r.get("backend") == "ollama_glm_ocr" and r.get("track") == "raw_ocr_addendum")]
    for row in rows:
        existing.append({
            "document_id": row["document_id"],
            "backend": "ollama_glm_ocr",
            "track": "raw_ocr_addendum",
            "json_parse_success": "",
            "schema_validity": "",
            "output_completeness": row["non_empty_output"],
            "required_field_coverage": "",
            "runtime_seconds": row["runtime_seconds"],
            "notes": f"decision={row['decision_label']}; approx_text_similarity_to_gt={row['approx_text_similarity_to_gt']}; key_proxy patient={row['patient_name']} date={row['date']} meds={row['medication_names']} vitals={row['vitals']} complaints={row['complaints_diagnosis']}",
        })
    write_csv(path, existing, fields)


def write_roster(glm_decision: str) -> None:
    lines = [
        "# Stage 1A Final release Recommended Full Run Roster",
        "",
        f"Generated: {now()}",
        "",
        "Recommended for Stage 1B based on Final release smoke so far:",
        "- `ollama_qwen3_vl_8b_raw_structured`: ready_for_full_run.",
        "- `ollama_qwen25_14b_ocr_to_json`: smoke_passed_but_low_quality until schema prompt is tightened.",
        "- `ollama_qwen3_8b_semantic_inference`: ready_for semantic-only add-on smoke scope.",
        "",
        "Hold or exclude:",
        "- `ollama_llava_13b_raw_structured`: smoke_passed_but_low_quality; one malformed JSON case on `p38_1`.",
        "- `ollama_qwen3_8b_ocr_to_json`: failed_smoke for 2/3 OCR-to-JSON cases.",
        "- `ollama_qwen25_14b_semantic_inference`: failed_smoke.",
        f"- `ollama_glm_ocr`: {glm_decision}; do not include in Stage 1B until pulled and passed the 3-document OCR addendum smoke.",
        "- `ollama_deepseek_ocr`: available_needs_pull_approval; not tested.",
        "- `ollama_moondream`: available_needs_pull_approval; not OCR-specific and not tested.",
    ]
    write_text(REPORTS_DIR / "stage1a_server1_recommended_full_run_roster.md", "\n".join(lines) + "\n")


def write_report(run_dir: Path, ollama_list: str, show_results: Dict[str, Dict[str, Any]], metric_rows: List[Dict[str, Any]], glm_installed: bool, glm_decision: str) -> None:
    lines = [
        "# Stage 1A Final release Ollama OCR Addendum",
        "",
        f"Generated: {now()}",
        "",
        f"- Output directory: `{run_dir}`",
        f"- `glm-ocr` installed locally: {str(glm_installed).lower()}",
        f"- Final `ollama_glm_ocr` decision label: `{glm_decision}`",
        "- Full benchmark was not started.",
        "- No Ollama model was pulled by this addendum.",
        "",
        "## Local Ollama List",
        "",
        "```",
        ollama_list,
        "```",
        "",
        "## Non-Destructive Ollama Show Checks",
        "",
    ]
    for name, result in show_results.items():
        lines.extend([
            f"### `{name}`",
            "",
            f"- Return code: {result['returncode']}",
            f"- stderr: `{result['stderr']}`",
            "```",
            result["stdout"],
            "```",
            "",
        ])
    lines.extend([
        "## Pullable OCR/Vision Candidates",
        "",
        "| model | size | input | context | RTX 4090 fit | decision |",
        "|---|---:|---|---:|---|---|",
    ])
    for item in OLLAMA_LIBRARY_MODELS:
        lines.append(f"| `{item['model']}` | {item['expected_size']} | {item['input']} | {item['context']} | {item['gpu_vram_expectation']} | `{item['decision_label']}` |")
    lines.extend([
        "",
        "## GLM-OCR Smoke Metrics",
        "",
        "| document_id | non_empty_output | approx_text_similarity_to_gt | patient | date | meds | vitals | complaints | runtime | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in metric_rows:
        lines.append(f"| `{row['document_id']}` | {row['non_empty_output']} | {row['approx_text_similarity_to_gt']} | {row['patient_name']} | {row['date']} | {row['medication_names']} | {row['vitals']} | {row['complaints_diagnosis']} | {row['runtime_seconds']} | `{row['decision_label']}` |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"`ollama_glm_ocr` should **not** be included in Stage 1B yet. Current label: `{glm_decision}`.",
        "It is small enough and relevant enough to pull-test next, but requires approval before pulling.",
        "",
        "Sources checked:",
        "- Ollama GLM-OCR library: https://ollama.com/library/glm-ocr",
        "- Ollama GLM-OCR tags: https://ollama.com/library/glm-ocr/tags",
        "- Hugging Face GLM-OCR model card: https://huggingface.co/zai-org/GLM-OCR",
        "- Ollama DeepSeek-OCR library: https://ollama.com/library/deepseek-ocr",
        "- Ollama Moondream library: https://ollama.com/library/moondream",
    ])
    write_text(REPORTS_DIR / "stage1a_server1_ollama_ocr_addendum.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final release Ollama OCR addendum")
    parser.add_argument("--run-if-installed", action="store_true", help="Run glm-ocr smoke only if already installed")
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / f"stage1a_server1_ollama_ocr_addendum_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    docs = selected_docs()
    ollama_list_text, installed = installed_models()
    show_results = {
        "glm-ocr": show_model("glm-ocr"),
        "glm-ocr:latest": show_model("glm-ocr:latest"),
        "deepseek-ocr": show_model("deepseek-ocr"),
        "moondream": show_model("moondream"),
        "qwen3-vl:8b-instruct": show_model("qwen3-vl:8b-instruct"),
    }
    glm_installed = "glm-ocr" in installed or "glm-ocr:latest" in installed
    if glm_installed and args.run_if_installed:
        metric_rows = run_glm_smoke(run_dir, docs)
        glm_decision = "ready_for_full_run" if all(r["decision_label"] == "ready_for_full_run" for r in metric_rows) else "failed_smoke"
    else:
        metric_rows = write_not_installed_logs(run_dir, docs, show_results["glm-ocr"])
        glm_decision = "available_needs_pull_approval"

    update_model_availability(installed, show_results, glm_decision)
    append_smoke_metrics(metric_rows)
    write_roster(glm_decision)
    write_report(run_dir, ollama_list_text, show_results, metric_rows, glm_installed, glm_decision)
    print(f"wrote addendum outputs: {run_dir}")


if __name__ == "__main__":
    main()
