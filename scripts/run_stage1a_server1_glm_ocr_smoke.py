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

"""Approved Server 1 GLM-OCR smoke recorder/runner.

This runner is safe to execute after `ollama pull glm-ocr:latest` approval.
If the model is not installed, it records a failed-smoke artifact with the
pull/show failure so Stage 1B cannot accidentally include it.
"""

from __future__ import annotations

import argparse
import base64
import csv
import difflib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
MODEL = "glm-ocr:latest"
BACKEND = "ollama_glm_ocr"
DOC_IDS = ["p4", "p25_1", "p38_1"]
MINIMAL_PROMPT_TEMPLATE = "Text Recognition: {image_path}"
PRESCRIPTION_PROMPT = (
    "Extract all visible text from this prescription image. Preserve line order, "
    "handwriting, abbreviations, spelling, numeric values, crossed-out/uncertain "
    "text if visible, and page-wise order. Do not normalize medical terms. Do not "
    "infer hidden content."
)


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
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


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


def gpu_memory() -> str:
    code, out, err = run_cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], timeout=10)
    return out if code == 0 else err


def model_installed() -> bool:
    code, out, _ = run_cmd(["ollama", "list"], timeout=20)
    if code != 0:
        return False
    return any(line.split()[0] == MODEL for line in out.splitlines()[1:] if line.strip())


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


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def ollama_generate(prompt: str, images: List[Path]) -> Tuple[bool, str, Any, str]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [encode_image(p) for p in images],
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


def flatten_gt(doc: Dict[str, str]) -> str:
    try:
        data = json.loads((PROJECT_ROOT / doc["ground_truth_json"]).read_text(encoding="utf-8"))
    except Exception:
        return ""
    parts = []
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
    gt_text = gt.lower()
    buckets = {
        "patient_name_recall_proxy": ["name", "patient", "devki", "alivelatamma"],
        "date_recall_proxy": ["date", "2024", "2023", "jul", "jun"],
        "medication_recall_proxy": ["tablet", "tab", "rx", "metformin", "gabapen", "pantop", "bep"],
        "vitals_recall_proxy": ["bp", "pulse", "weight", "height", "bmi"],
        "complaints_diagnosis_recall_proxy": ["pain", "diagnosis", "c/o", "complaint", "burning"],
    }
    scores = {}
    for key, terms in buckets.items():
        relevant = [t for t in terms if t in gt_text]
        scores[key] = round(sum(1 for t in relevant if t in text) / len(relevant), 4) if relevant else 0.0
    return scores


def failure_artifacts(run_dir: Path, docs: List[Dict[str, str]], pull: Dict[str, Any], show: Dict[str, Any], ollama_version: str, ollama_list: str) -> List[Dict[str, Any]]:
    rows = []
    for doc in docs:
        image_paths = [PROJECT_ROOT / p for p in doc["source_images_ordered"].split(";") if p]
        raw_path = run_dir / "raw_responses" / BACKEND / f"{doc['document_id']}.txt"
        out_path = run_dir / "raw_ocr" / BACKEND / f"{doc['document_id']}.txt"
        log_path = run_dir / "logs" / BACKEND / f"{doc['document_id']}.json"
        fail_path = run_dir / "failed_cases" / BACKEND / f"{doc['document_id']}.json"
        reason = "ollama pull glm-ocr:latest failed before smoke; model is not installed"
        log = {
            "server_name": SERVER_NAME,
            "model_name": MODEL,
            "model_size": "2.2GB expected from Ollama library; not installed locally",
            "architecture_capabilities": "unknown locally because pull failed; library says Text+Image OCR VLM",
            "context_length": "128K expected from Ollama library",
            "quantization": "not shown locally because pull failed",
            "license": "MIT per Hugging Face GLM-OCR model card; not shown locally because pull failed",
            "pull_success": False,
            "pull_command": "ollama pull glm-ocr:latest",
            "pull_result": pull,
            "ollama_version": ollama_version,
            "ollama_list_after_pull": ollama_list,
            "ollama_show_result": show,
            "prompt_variants": [MINIMAL_PROMPT_TEMPLATE, PRESCRIPTION_PROMPT],
            "image_paths": [rel(p) for p in image_paths],
            "runtime_seconds": 0,
            "gpu_memory_before": gpu_memory(),
            "gpu_memory_after": gpu_memory(),
            "output_character_count": 0,
            "non_empty_output": False,
            "error": reason,
            "image_input_accepted": False,
            "multi_image_input_accepted": False,
            "raw_output_path": rel(out_path),
            "decision_label": "failed_smoke",
        }
        write_text(out_path, "")
        write_text(raw_path, json.dumps(log, indent=2, ensure_ascii=False))
        write_json(log_path, log)
        write_json(fail_path, {"failure_reason": reason, "log": log})
        rows.append({
            "document_id": doc["document_id"],
            "non_empty_output_rate": 0,
            "approx_text_similarity_to_gt": 0.0,
            "patient_name_recall_proxy": 0.0,
            "date_recall_proxy": 0.0,
            "medication_recall_proxy": 0.0,
            "vitals_recall_proxy": 0.0,
            "complaints_diagnosis_recall_proxy": 0.0,
            "runtime_seconds": 0,
            "decision_label": "failed_smoke",
        })
    return rows


def smoke_artifacts(run_dir: Path, docs: List[Dict[str, str]], show: Dict[str, Any], ollama_version: str, ollama_list: str) -> List[Dict[str, Any]]:
    rows = []
    for doc in docs:
        images = package_images(doc, run_dir)
        raw_chunks = []
        outputs = []
        errors = []
        start = time.time()
        gpu_before = gpu_memory()
        multi_ok = False
        if len(images) > 1:
            ok, content, raw, error = ollama_generate(MINIMAL_PROMPT_TEMPLATE.format(image_path=";".join(str(p) for p in images)), images)
            multi_ok = ok and bool(content.strip())
            raw_chunks.append({"variant": "minimal_multi_image", "ok": ok, "content": content, "raw": raw, "error": error})
            if multi_ok:
                outputs.append(content)
            else:
                errors.append(error or "multi-image returned empty")
        if not outputs:
            for image in images:
                prompt = MINIMAL_PROMPT_TEMPLATE.format(image_path=str(image))
                ok, content, raw, error = ollama_generate(prompt, [image])
                raw_chunks.append({"variant": "minimal_single_image", "prompt": prompt, "image": rel(image), "ok": ok, "content": content, "raw": raw, "error": error})
                if ok:
                    outputs.append(f"<!-- {image.name} -->\n{content}")
                else:
                    errors.append(error)
        if images:
            ok, content, raw, error = ollama_generate(PRESCRIPTION_PROMPT, images if len(images) == 1 else [images[0]])
            raw_chunks.append({"variant": "prescription_aware", "prompt": PRESCRIPTION_PROMPT, "image": rel(images[0]), "ok": ok, "content": content, "raw": raw, "error": error})
        runtime = round(time.time() - start, 3)
        gpu_after = gpu_memory()
        text = "\n\n".join(outputs)
        raw_path = run_dir / "raw_responses" / BACKEND / f"{doc['document_id']}.txt"
        out_path = run_dir / "raw_ocr" / BACKEND / f"{doc['document_id']}.txt"
        log_path = run_dir / "logs" / BACKEND / f"{doc['document_id']}.json"
        fail_path = run_dir / "failed_cases" / BACKEND / f"{doc['document_id']}.json"
        write_text(raw_path, json.dumps(raw_chunks, indent=2, ensure_ascii=False))
        write_text(out_path, text)
        gt = flatten_gt(doc)
        similarity = round(difflib.SequenceMatcher(None, text.lower(), gt.lower()).ratio(), 4) if text and gt else 0.0
        proxies = proxy_scores(text, gt)
        success = bool(text.strip()) and not errors
        log = {
            "server_name": SERVER_NAME,
            "model_name": MODEL,
            "model_size": "see ollama show/list",
            "architecture_capabilities": show.get("stdout", ""),
            "pull_success": True,
            "ollama_version": ollama_version,
            "ollama_list_after_pull": ollama_list,
            "prompt_variants": [MINIMAL_PROMPT_TEMPLATE, PRESCRIPTION_PROMPT],
            "image_paths": [rel(p) for p in images],
            "runtime_seconds": runtime,
            "gpu_memory_before": gpu_before,
            "gpu_memory_after": gpu_after,
            "output_character_count": len(text),
            "non_empty_output": bool(text.strip()),
            "error": "; ".join(e for e in errors if e),
            "image_input_accepted": bool(text.strip()),
            "multi_image_input_accepted": multi_ok,
            "raw_output_path": rel(out_path),
            "approx_text_similarity_to_gt": similarity,
            **proxies,
        }
        write_json(log_path, log)
        if not success:
            write_json(fail_path, {"failure_reason": log["error"] or "empty output", "log": log})
        rows.append({
            "document_id": doc["document_id"],
            "non_empty_output_rate": int(bool(text.strip())),
            "approx_text_similarity_to_gt": similarity,
            **proxies,
            "runtime_seconds": runtime,
            "decision_label": "ready_for_full_run" if success and similarity >= 0.1 else ("smoke_passed_but_low_quality" if success else "failed_smoke"),
        })
    return rows


def final_decision(rows: List[Dict[str, Any]], pull_ok: bool) -> str:
    if not pull_ok:
        return "failed_smoke"
    non_empty = sum(1 for r in rows if r["non_empty_output_rate"]) / max(1, len(rows))
    avg_similarity = sum(float(r["approx_text_similarity_to_gt"]) for r in rows) / max(1, len(rows))
    if non_empty == 1 and avg_similarity >= 0.18:
        return "ready_for_full_run"
    if non_empty >= 2 / 3 and avg_similarity >= 0.08:
        return "ready_for_limited_run"
    if non_empty > 0:
        return "smoke_passed_but_low_quality"
    return "failed_smoke"


def update_availability(decision: str, installed: bool, show: Dict[str, Any], pull: Dict[str, Any]) -> None:
    path = REPORTS_DIR / "stage1a_server1_model_availability.csv"
    rows = read_csv(path)
    by_backend = {r.get("backend"): r for r in rows}
    by_backend[BACKEND] = {
        "backend": BACKEND,
        "model": MODEL,
        "modality": "ocr_vlm",
        "installed": str(installed).lower(),
        "smoke_test_status": decision,
        "notes": f"pull_success={pull['returncode'] == 0}; show_returncode={show['returncode']}; error={(pull.get('stderr') or show.get('stderr') or '')[:220]}",
    }
    fields = ["backend", "model", "modality", "installed", "smoke_test_status", "notes"]
    write_csv(path, list(by_backend.values()), fields)


def update_metrics(rows: List[Dict[str, Any]]) -> None:
    path = REPORTS_DIR / "stage1a_server1_smoke_metrics.csv"
    existing = read_csv(path)
    fields = ["document_id", "backend", "track", "json_parse_success", "schema_validity", "output_completeness", "required_field_coverage", "scalar_accuracy_exact", "scalar_accuracy_lenient", "entity_lenient_f1", "hallucination_rate", "missing_entity_rate", "runtime_seconds", "notes"]
    existing = [r for r in existing if not (r.get("backend") == BACKEND and r.get("track") in {"raw_ocr_addendum", "glm_ocr_smoke"})]
    for row in rows:
        existing.append({
            "document_id": row["document_id"],
            "backend": BACKEND,
            "track": "glm_ocr_smoke",
            "json_parse_success": "",
            "schema_validity": "",
            "output_completeness": row["non_empty_output_rate"],
            "required_field_coverage": "",
            "runtime_seconds": row["runtime_seconds"],
            "notes": (
                f"decision={row['decision_label']}; approx_text_similarity_to_gt={row['approx_text_similarity_to_gt']}; "
                f"patient={row['patient_name_recall_proxy']} date={row['date_recall_proxy']} "
                f"meds={row['medication_recall_proxy']} vitals={row['vitals_recall_proxy']} "
                f"complaints={row['complaints_diagnosis_recall_proxy']}"
            ),
        })
    write_csv(path, existing, fields)


def update_failure_log(decision: str, pull: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    lines = ["# Stage 1A Server 1 Failure Log", "", f"Generated: {now()}", ""]
    old_path = REPORTS_DIR / "stage1a_server1_failure_log.md"
    if old_path.exists():
        old = old_path.read_text(encoding="utf-8").splitlines()
        old_items = [line for line in old if line.startswith("- `") and BACKEND not in line]
        lines.extend(old_items)
    if decision == "failed_smoke":
        err = pull.get("stderr") or "GLM-OCR smoke failed"
        for row in rows:
            lines.append(f"- `{BACKEND}` `{row['document_id']}`: {err}")
    write_text(old_path, "\n".join(lines) + "\n")


def update_roster(decision: str) -> None:
    lines = [
        "# Stage 1A Server 1 Recommended Full Run Roster",
        "",
        f"Generated: {now()}",
        "",
        "Recommended for Stage 1B based on Server 1 smoke so far:",
        "- `ollama_qwen3_vl_8b_raw_structured`: ready_for_full_run.",
        "- `ollama_qwen25_14b_ocr_to_json`: smoke_passed_but_low_quality until schema prompt is tightened.",
        "- `ollama_qwen3_8b_semantic_inference`: ready for semantic-only add-on smoke scope.",
        "",
        "Hold or exclude:",
        "- `ollama_llava_13b_raw_structured`: smoke_passed_but_low_quality; one malformed JSON case on `p38_1`.",
        "- `ollama_qwen3_8b_ocr_to_json`: failed_smoke for 2/3 OCR-to-JSON cases.",
        "- `ollama_qwen25_14b_semantic_inference`: failed_smoke.",
        f"- `ollama_glm_ocr`: {decision}; do not include in Stage 1B unless it is successfully pulled and passes the 3-document OCR smoke.",
        "- `ollama_deepseek_ocr`: do not pull next until the GLM-OCR/Ollama version issue is resolved.",
        "- `ollama_moondream`: do not pull next; not OCR-specific and GLM-OCR has not yet been tested.",
    ]
    write_text(REPORTS_DIR / "stage1a_server1_recommended_full_run_roster.md", "\n".join(lines) + "\n")


def write_summary(run_dir: Path, rows: List[Dict[str, Any]], decision: str, pull: Dict[str, Any], show: Dict[str, Any], ollama_version: str, ollama_list: str) -> None:
    lines = [
        "# Stage 1A Server 1 GLM-OCR Smoke Summary",
        "",
        f"Generated: {now()}",
        "",
        f"- Output directory: `{run_dir}`",
        f"- Pull command: `ollama pull {MODEL}`",
        f"- Pull success: {str(pull['returncode'] == 0).lower()}",
        f"- Ollama version: `{ollama_version}`",
        f"- Final decision label: `{decision}`",
        "- Full Stage 1B benchmark was not started.",
        "- `deepseek-ocr:latest` and `moondream:latest` were not pulled.",
        "",
        "## Pull / Show",
        "",
        f"- Pull stderr: `{pull.get('stderr', '')}`",
        f"- Show return code: {show.get('returncode')}",
        f"- Show stderr: `{show.get('stderr', '')}`",
        "",
        "## Ollama List After Pull Attempt",
        "",
        "```",
        ollama_list,
        "```",
        "",
        "## Smoke Metrics",
        "",
        "| document_id | non_empty | similarity | patient | date | meds | vitals | complaints | runtime | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['document_id']}` | {r['non_empty_output_rate']} | {r['approx_text_similarity_to_gt']} | "
            f"{r['patient_name_recall_proxy']} | {r['date_recall_proxy']} | {r['medication_recall_proxy']} | "
            f"{r['vitals_recall_proxy']} | {r['complaints_diagnosis_recall_proxy']} | {r['runtime_seconds']} | `{r['decision_label']}` |"
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"`ollama_glm_ocr` should **not** be included in Stage 1B now. Current label: `{decision}`.",
        "Resolve the Ollama/model compatibility issue, rerun only this 3-document smoke, then reconsider inclusion.",
    ])
    write_text(REPORTS_DIR / "stage1a_server1_glm_ocr_smoke_summary.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/record approved GLM-OCR smoke")
    parser.add_argument("--pull-stdout", default="")
    parser.add_argument("--pull-stderr", default="")
    parser.add_argument("--pull-returncode", type=int, default=1)
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / f"stage1a_server1_glm_ocr_smoke_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    docs = selected_docs()
    pull = {"returncode": args.pull_returncode, "stdout": args.pull_stdout, "stderr": args.pull_stderr}
    code, version_out, version_err = run_cmd(["ollama", "--version"], timeout=10)
    ollama_version = version_out or version_err
    _, ollama_list, list_err = run_cmd(["ollama", "list"], timeout=20)
    ollama_list = ollama_list or list_err
    show_code, show_out, show_err = run_cmd(["ollama", "show", MODEL], timeout=20)
    show = {"returncode": show_code, "stdout": show_out, "stderr": show_err}
    installed = model_installed()

    if installed and pull["returncode"] == 0:
        rows = smoke_artifacts(run_dir, docs, show, ollama_version, ollama_list)
    else:
        rows = failure_artifacts(run_dir, docs, pull, show, ollama_version, ollama_list)
    decision = final_decision(rows, pull["returncode"] == 0 and installed)
    update_availability(decision, installed, show, pull)
    update_metrics(rows)
    update_failure_log(decision, pull, rows)
    update_roster(decision)
    write_summary(run_dir, rows, decision, pull, show, ollama_version, ollama_list)
    print(f"wrote GLM-OCR smoke outputs: {run_dir}")


if __name__ == "__main__":
    main()
