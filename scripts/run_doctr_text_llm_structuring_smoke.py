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

"""Phase D docTR OCR text -> text LLM structuring smoke.

Consumes OCR text files only. No images are sent to the model.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import dotenv
import requests
import yaml

dotenv.load_dotenv(PROJECT_ROOT / ".env")
dotenv.load_dotenv(PROJECT_ROOT.parent / ".env")

from src.cli.extract import clean_and_repair_json
from src.schemas.raw_extraction import CanonicalRawDoc, Metadata
from src.utils.rate_limiter import get_global_limiter, init_global_limiter

OCR_DIR = PROJECT_ROOT / "outputs/ocr_transcriptions/doctr_baseline"
OUTPUT_DIR = PROJECT_ROOT / "outputs/raw_extractions/doctr_text_llm_smoke"
RAW_DIR = PROJECT_ROOT / "outputs/raw_responses/doctr_text_llm_smoke"
USAGE_LOG = PROJECT_ROOT / "logs/doctr_text_llm_usage.csv"
MANIFEST_PATH = PROJECT_ROOT / "data/manifest_phase_d_doctr_text_llm_smoke.csv"
STATUS_REPORT = PROJECT_ROOT / "reports/phase_d_doctr_text_llm_smoke_status.md"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs/prompts_phase_d_ocr_text.yaml"

SMOKE_DOCS = ["p1", "p2", "p36_1"]

PROMPT_TEMPLATE = """Convert the following OCR/transcribed clinical document text into one valid CanonicalRawDoc JSON object using the raw_rx_v2 schema. Return only JSON. Do not include markdown, explanation, or reasoning. Preserve raw clinical wording where uncertain. Do not invent facts not present in the OCR text.

Required JSON object shape:
{{
  "schema_version": "raw_rx_v2",
  "document_id": "{document_id}",
  "patient_information": {{"name": null, "age": null, "gender": null, "address": null, "phone": null, "patient_identifier": null, "abha_id": null}},
  "encounter_information": {{"date": null, "department": null, "hospital_name": null, "doctor_name": null, "visit_type": null, "fees": null, "room_or_queue_no": null}},
  "complaints_or_diagnosis": [],
  "observations": [],
  "medications": [],
  "procedures": [],
  "advice": [],
  "follow_up": null,
  "allergy_mentions": [],
  "other_notes": [],
  "lab_observations": []
}}

Rules:
- Put visible drug/prescription lines in medications. Each medication must include raw_line_text and page_number when possible.
- Put diagnoses, complaints, tests, ophthalmology markers, procedures, and advice in the matching arrays as RawEntityItem objects with raw_text and page_number.
- Use null or empty arrays for fields not visible in the OCR text.
- Do not invent facts that are not supported by the OCR text.

OCR text:
{ocr_text}
"""


def configure_output_paths(prompt_version: str) -> None:
    global OUTPUT_DIR, RAW_DIR, USAGE_LOG, MANIFEST_PATH, STATUS_REPORT
    if prompt_version == "phase_d_ocr_text_v2":
        OUTPUT_DIR = PROJECT_ROOT / "outputs/raw_extractions/doctr_text_llm_smoke_v2"
        RAW_DIR = PROJECT_ROOT / "outputs/raw_responses/doctr_text_llm_smoke_v2"
        USAGE_LOG = PROJECT_ROOT / "logs/doctr_text_llm_usage_v2.csv"
        MANIFEST_PATH = PROJECT_ROOT / "data/manifest_phase_d_doctr_text_llm_smoke_v2.csv"
        STATUS_REPORT = PROJECT_ROOT / "reports/phase_d_doctr_text_llm_v2_status.md"


def read_ocr_text(document_id: str) -> tuple[str, list[str]]:
    combined = OCR_DIR / f"{document_id}_combined.txt"
    if combined.exists():
        return combined.read_text(encoding="utf-8").strip(), [str(combined.relative_to(PROJECT_ROOT))]
    single = OCR_DIR / f"{document_id}_page1.txt"
    if single.exists():
        return single.read_text(encoding="utf-8").strip(), [str(single.relative_to(PROJECT_ROOT))]
    return "", []


def add_line_ids(ocr_text: str) -> str:
    numbered = []
    for idx, line in enumerate(ocr_text.splitlines(), start=1):
        numbered.append(f"[L{idx:03d}] {line}")
    return "\n".join(numbered)


def build_prompt(document_id: str, ocr_text: str, prompt_version: str, include_line_ids: bool) -> str:
    text = add_line_ids(ocr_text) if include_line_ids else ocr_text
    if prompt_version == "phase_d_ocr_text_v2":
        with PROMPT_CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        template = config["doctr_ocr_text_to_raw_rx_v2_prompt_v2"]["user_prompt"]
        return template.replace("{{document_id}}", document_id).replace("{{ocr_text}}", text)
    return PROMPT_TEMPLATE.format(document_id=document_id, ocr_text=text)


def append_usage(row: dict[str, Any]) -> None:
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "document_id",
        "backend_name",
        "model",
        "ocr_text_chars",
        "max_tokens",
        "latency_ms",
        "status",
        "validation_status",
        "input_tokens_estimated",
        "output_tokens_estimated",
        "api_reported_prompt_tokens",
        "api_reported_completion_tokens",
        "api_reported_total_tokens",
        "error",
    ]
    exists = USAGE_LOG.exists() and USAGE_LOG.stat().st_size > 0
    if exists:
        with USAGE_LOG.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fieldnames != fieldnames:
            with USAGE_LOG.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    if "backend_name" not in existing_row or existing_row.get("backend_name") is None:
                        model = existing_row.get("model", "")
                        existing_row["backend_name"] = "qwen3_27b_text_only" if model == "qwen3-27b" else ""
                    writer.writerow({name: existing_row.get(name, "") for name in fieldnames})
            exists = True
    with USAGE_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_metadata(
    response: dict[str, Any],
    model: str,
    backend_name: str,
    prompt_version: str,
    processing_time_ms: float,
    warnings: list[str],
) -> Metadata:
    return Metadata(
        model_name=model,
        model_version=None,
        prompt_version=prompt_version,
        backend_name=backend_name,
        processing_time_ms=processing_time_ms,
        decoding_parameters=response.get("decoding_parameters", {}),
        schema_version="raw_rx_v2",
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence_score=None,
        uncertainty_notes="Generated from docTR OCR text only; image was not provided to the text LLM.",
        pages=[],
        document_type=None,
        validation_warnings=warnings,
        type_coercions=[],
    )


def normalize_parsed_dict(parsed: dict[str, Any], document_id: str, warnings: list[str]) -> dict[str, Any]:
    parsed["document_id"] = document_id
    parsed["schema_version"] = "raw_rx_v2"
    scalar_sections = {
        "patient_information": ["name", "age", "gender", "address", "phone", "patient_identifier", "abha_id"],
        "encounter_information": ["date", "department", "hospital_name", "doctor_name", "visit_type", "fees", "room_or_queue_no"],
    }
    for section, keys in scalar_sections.items():
        if isinstance(parsed.get(section), dict):
            for key in keys:
                value = parsed[section].get(key)
                if value is not None and not isinstance(value, str):
                    parsed[section][key] = str(value)
                    warnings.append(f"{section}.{key}: coerced {type(value).__name__} to string")

    def normalize_page_number(item: Any, key: str, idx: int) -> None:
        if not isinstance(item, dict):
            return
        value = item.get("page_number")
        if value is None:
            item["page_number"] = 1
            warnings.append(f"{key}[{idx}].page_number: set null/missing to 1")
        elif isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                item["page_number"] = int(match.group(0))
                warnings.append(f"{key}[{idx}].page_number: coerced string to integer")
            else:
                item["page_number"] = 1
                warnings.append(f"{key}[{idx}].page_number: nonnumeric string set to 1")

    def normalize_entity_item(item: Any, key: str, idx: int) -> dict[str, Any] | None:
        if isinstance(item, str):
            value = item.strip()
            if not value:
                return None
            warnings.append(f"{key}[{idx}]: converted string to RawEntityItem")
            return {"raw_text": value, "evidence_text": value, "page_number": 1}
        if not isinstance(item, dict):
            warnings.append(f"{key}[{idx}]: dropped non-dict/non-string item")
            return None
        if not item.get("raw_text"):
            fallback = item.get("raw_line_text") or item.get("evidence_text") or item.get("text") or item.get("value")
            if fallback:
                item["raw_text"] = str(fallback)
                warnings.append(f"{key}[{idx}].raw_text: populated from fallback field")
        normalize_page_number(item, key, idx)
        return item

    def normalize_medication_item(item: Any, idx: int) -> dict[str, Any] | None:
        if isinstance(item, str):
            value = item.strip()
            if not value:
                return None
            warnings.append(f"medications[{idx}]: converted string to RawMedicationItem")
            return {"raw_line_text": value, "raw_name": value, "evidence_text": value, "page_number": 1}
        if not isinstance(item, dict):
            warnings.append(f"medications[{idx}]: dropped non-dict/non-string item")
            return None
        if not item.get("raw_line_text"):
            parts = [
                item.get("raw_medication_text"),
                item.get("raw_name"),
                item.get("raw_dosage_text"),
                item.get("raw_dose_text"),
                item.get("raw_route_text"),
                item.get("raw_frequency_text"),
                item.get("raw_duration_text"),
                item.get("raw_instruction_text"),
                item.get("raw_notes"),
                item.get("evidence_text"),
            ]
            fallback = " ".join(str(part).strip() for part in parts if part)
            if fallback:
                item["raw_line_text"] = fallback
                warnings.append(f"medications[{idx}].raw_line_text: constructed from medication fallback fields")
        field_map = {
            "raw_medication_text": "raw_name",
            "raw_dosage_text": "raw_dosage",
            "raw_route_text": "raw_route",
            "raw_frequency_text": "raw_frequency",
            "raw_duration_text": "raw_duration",
            "raw_instruction_text": "raw_instruction",
            "raw_timing_text": "raw_timing",
        }
        for source, target in field_map.items():
            if source in item and not item.get(target):
                item[target] = item[source]
        normalize_page_number(item, "medications", idx)
        return item

    def normalize_lab_item(item: Any, idx: int) -> dict[str, Any] | None:
        if isinstance(item, str):
            value = item.strip()
            if not value:
                return None
            warnings.append(f"lab_observations[{idx}]: converted string to RawLabObservationItem")
            return {"raw_line_text": value, "test_name": value, "evidence_text": value, "page_number": 1}
        if not isinstance(item, dict):
            warnings.append(f"lab_observations[{idx}]: dropped non-dict/non-string item")
            return None
        if not item.get("raw_line_text"):
            parts = [item.get("date"), item.get("test_name"), item.get("result"), item.get("unit"), item.get("other"), item.get("evidence_text")]
            fallback = " ".join(str(part).strip() for part in parts if part)
            if fallback:
                item["raw_line_text"] = fallback
                warnings.append(f"lab_observations[{idx}].raw_line_text: constructed from fallback fields")
        normalize_page_number(item, "lab_observations", idx)
        return item

    for key in [
        "complaints_or_diagnosis",
        "observations",
        "procedures",
        "advice",
        "allergy_mentions",
        "other_notes",
    ]:
        if parsed.get(key) is None:
            parsed[key] = []
            warnings.append(f"{key}: coerced null to empty list")
        if isinstance(parsed.get(key), list):
            cleaned = []
            for idx, item in enumerate(parsed[key]):
                normalized = normalize_entity_item(item, key, idx)
                if normalized and normalized.get("raw_text"):
                    cleaned.append(normalized)
            parsed[key] = cleaned

    if parsed.get("medications") is None:
        parsed["medications"] = []
        warnings.append("medications: coerced null to empty list")
    if isinstance(parsed.get("medications"), list):
        cleaned_meds = []
        for idx, item in enumerate(parsed["medications"]):
            normalized = normalize_medication_item(item, idx)
            if normalized and normalized.get("raw_line_text"):
                cleaned_meds.append(normalized)
        parsed["medications"] = cleaned_meds

    if parsed.get("lab_observations") is None:
        parsed["lab_observations"] = []
        warnings.append("lab_observations: coerced null to empty list")
    if isinstance(parsed.get("lab_observations"), list):
        cleaned_labs = []
        for idx, item in enumerate(parsed["lab_observations"]):
            normalized = normalize_lab_item(item, idx)
            if normalized and (normalized.get("raw_line_text") or normalized.get("test_name") or normalized.get("result")):
                cleaned_labs.append(normalized)
        parsed["lab_observations"] = cleaned_labs

    if isinstance(parsed.get("follow_up"), str):
        value = parsed["follow_up"].strip()
        parsed["follow_up"] = {"raw_text": value} if value else None
        warnings.append("follow_up: converted string to RawFollowUp")

    if "patient_information" not in parsed or parsed["patient_information"] is None:
        parsed["patient_information"] = {}
        warnings.append("patient_information: added empty object")
    if "encounter_information" not in parsed or parsed["encounter_information"] is None:
        parsed["encounter_information"] = {}
        warnings.append("encounter_information: added empty object")
    return parsed


def parse_and_validate(
    raw_content: str,
    document_id: str,
    response: dict[str, Any],
    model: str,
    backend_name: str,
    prompt_version: str,
    processing_time_ms: float,
) -> tuple[CanonicalRawDoc | None, str, list[str]]:
    warnings: list[str] = []
    try:
        repaired = clean_and_repair_json(raw_content)
        parsed = json.loads(repaired)
        if not isinstance(parsed, dict):
            return None, "top-level JSON was not an object", warnings
        parsed = normalize_parsed_dict(parsed, document_id, warnings)
        doc = CanonicalRawDoc(**parsed)
        doc.metadata = build_metadata(response, model, backend_name, prompt_version, processing_time_ms, warnings)
        return doc, "", warnings
    except Exception as exc:
        return None, re.sub(r"\s+", " ", str(exc)).strip()[:1000], warnings


def write_manifest(valid_docs: list[str]) -> None:
    source_manifest = PROJECT_ROOT / "data/manifest_qwen3_27b_full.csv"
    with source_manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    selected = []
    for row in rows:
        if row["document_id"] in valid_docs:
            row = dict(row)
            row["prediction_path"] = f"outputs/raw_extractions/doctr_text_llm_smoke/{row['document_id']}.json"
            try:
                row["prediction_path"] = str((OUTPUT_DIR / f"{row['document_id']}.json").relative_to(PROJECT_ROOT))
            except ValueError:
                row["prediction_path"] = str(OUTPUT_DIR / f"{row['document_id']}.json")
            selected.append(row)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


def write_status_report(results: list[dict[str, Any]], benchmark_summary_path: Path | None = None) -> None:
    STATUS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    valid_docs = [row["document_id"] for row in results if row["validation_status"] == "valid"]
    p36_recovered = "p36_1" in valid_docs
    backends = sorted({row.get("backend_name", "") for row in results if row.get("backend_name")})
    lines = [
        "# Phase D docTR + Text LLM Structuring Smoke Status",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "Text-only LLM structuring from saved docTR OCR text. No images were sent to the LLM, and this is not a full Phase D benchmark.",
        "",
        "## Backend",
        "",
        f"- backend used: {', '.join(backends) if backends else 'none'}",
        "",
        "## Schema Validity",
        "",
        "| document_id | backend | attempted | validation_status | output_json | raw_response | error |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['document_id']} | {row.get('backend_name', '')} | {row['attempted']} | {row['validation_status']} | "
            f"`{row.get('output_json', '')}` | `{row.get('raw_response', '')}` | {row.get('error', '')} |"
        )
    lines.extend([
        "",
        f"- Valid JSON documents: {len(valid_docs)}",
        f"- p36_1 recovered as valid CanonicalRawDoc JSON: {'yes' if p36_recovered else 'no'}",
        f"- Smoke manifest: `{MANIFEST_PATH.relative_to(PROJECT_ROOT)}`" if valid_docs else "- Smoke manifest: not created because no valid JSON outputs were produced.",
        "",
    ])

    if benchmark_summary_path and benchmark_summary_path.exists():
        summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
        lines.extend([
            "## Smoke Benchmark Metrics",
            "",
            f"- total documents: {summary.get('total_documents', 0)}",
            f"- schema parse success: {summary.get('schema_parse_success_rate', 0) * 100:.2f}%",
            f"- scalar exact: {summary.get('scalar_accuracy_exact', 0) * 100:.2f}%",
            f"- scalar lenient: {summary.get('scalar_accuracy_lenient', 0) * 100:.2f}%",
            f"- entity exact F1: {summary.get('entity_exact_f1_macro', 0) * 100:.2f}%",
            f"- entity lenient F1: {summary.get('entity_lenient_f1_macro', 0) * 100:.2f}%",
            f"- hallucination rate: {summary.get('hallucination_rate', 0) * 100:.2f}%",
            f"- missing entity rate: {summary.get('missing_entity_rate', 0) * 100:.2f}%",
            f"- annotation-gap rate: {summary.get('annotation_gap_rate', 0) * 100:.2f}%",
            f"- overall score: {summary.get('experimental_overall_score', 0) * 100:.2f}%",
            "",
            "## Comparison Note",
            "",
            "Compare this smoke result only against matching smoke documents. It is not directly comparable to the Phase C 13-document partial benchmark.",
            "Phase C quality on its 13 successful documents was: scalar lenient 79.12%, entity lenient F1 21.49%, hallucination rate 5.75%, missing entity rate 58.35%, overall score 46.63%.",
            "",
        ])
    else:
        lines.extend([
            "## Smoke Benchmark Metrics",
            "",
            "Benchmark not run because no valid CanonicalRawDoc JSON outputs were produced by the text-only LLM smoke.",
            "",
        ])
    lines.append("Production readiness is not established.")
    lines.append("")
    STATUS_REPORT.write_text("\n".join(lines), encoding="utf-8")


def usage_from_response(response: dict[str, Any]) -> tuple[int, int, int]:
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or 0
    if not total_tokens and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    return int(prompt_tokens or 0), int(completion_tokens or 0), int(total_tokens or 0)


def run_chat_completion(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float | None,
    timeout: int,
) -> tuple[dict[str, Any], float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,
    }
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    start = time.time()
    try:
        response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
        latency_ms = (time.time() - start) * 1000
        try:
            raw = response.json()
        except Exception:
            raw = {"raw_text": response.text}
        content = ""
        if isinstance(raw, dict):
            choices = raw.get("choices") or []
            if choices:
                content = ((choices[0].get("message") or {}).get("content") or "")
        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}: {response.text[:1000]}",
                "content": content,
                "raw_response": raw,
                "usage": raw.get("usage", {}) if isinstance(raw, dict) else {},
                "decoding_parameters": {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "repetition_penalty": repetition_penalty},
            }, latency_ms
        return {
            "content": content,
            "raw_response": raw,
            "usage": raw.get("usage", {}) if isinstance(raw, dict) else {},
            "decoding_parameters": {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "repetition_penalty": repetition_penalty},
        }, latency_ms
    except Exception as exc:
        latency_ms = (time.time() - start) * 1000
        return {
            "error": str(exc),
            "content": "",
            "raw_response": {},
            "usage": {},
            "decoding_parameters": {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "repetition_penalty": repetition_penalty},
        }, latency_ms


def run(args: argparse.Namespace) -> int:
    configure_output_paths(args.prompt_version)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.backend == "internal":
        api_key = os.getenv("QWEN3_27B_API_KEY")
        base_url = os.getenv("QWEN3_27B_BASE_URL", "http://10.10.110.37:4000/v1")
        model = os.getenv("QWEN3_27B_MODEL", "qwen3-27b")
        backend_name = "qwen3_27b_text_only"
    else:
        api_key = args.api_key or os.getenv("LOCAL_QWEN3_MOE_API_KEY") or None
        base_url = args.model_base_url
        model = args.model_id
        backend_name = "local_qwen3_moe_text_vllm"

    if args.backend == "internal" and not api_key:
        results = [
            {
                "document_id": doc_id,
                "backend_name": "qwen3_27b_text_only",
                "attempted": "no",
                "validation_status": "not_attempted",
                "error": "QWEN3_27B_API_KEY is not set",
            }
            for doc_id in SMOKE_DOCS
        ]
        write_status_report(results)
        print("QWEN3_27B_API_KEY is not set")
        return 2
    init_global_limiter(tpm_limit=args.tpm_limit, rpm_limit=args.rpm_limit, window_seconds=60, buffer_seconds=15, max_retries_rate_limit=1)

    results: list[dict[str, Any]] = []
    valid_docs: list[str] = []
    for doc_id in SMOKE_DOCS:
        ocr_text, ocr_paths = read_ocr_text(doc_id)
        if len(ocr_text.strip()) < args.min_ocr_chars:
            result = {
                "document_id": doc_id,
                "backend_name": backend_name,
                "attempted": "no",
                "validation_status": "not_attempted",
                "error": f"OCR text below minimum char threshold ({len(ocr_text)} < {args.min_ocr_chars})",
            }
            results.append(result)
            print(f"{doc_id}: skipped, OCR chars={len(ocr_text)}")
            continue

        prompt = build_prompt(doc_id, ocr_text, args.prompt_version, args.include_ocr_lines_as_evidence)
        estimated_input = max(1, len(prompt) // 4)
        doc_max_tokens = args.max_tokens if args.prompt_version == "phase_d_ocr_text_v2" else (args.p36_max_tokens if doc_id == "p36_1" else args.max_tokens)
        estimated_output = doc_max_tokens
        limiter = get_global_limiter()
        limiter.wait_if_needed(f"doctr_text_{doc_id}", estimated_input + estimated_output)

        raw_response_path = RAW_DIR / f"{doc_id}.txt"
        output_json_path = OUTPUT_DIR / f"{doc_id}.json"
        response, latency_ms = run_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=doc_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            timeout=args.timeout,
        )
        content = response.get("content") or ""
        raw_response_path.write_text(content, encoding="utf-8")
        (RAW_DIR / f"{doc_id}_response.json").write_text(
            json.dumps(
                {
                    "document_id": doc_id,
                    "backend_name": backend_name,
                    "base_url": base_url,
                    "model": model,
                    "prompt_version": args.prompt_version,
                    "ocr_paths": ocr_paths,
                    "include_ocr_lines_as_evidence": args.include_ocr_lines_as_evidence,
                    "status": "failed" if response.get("error") else "success",
                    "error": response.get("error", ""),
                    "content": content,
                    "raw_response": response.get("raw_response", {}),
                    "usage": response.get("usage", {}),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        api_prompt, api_completion, api_total = usage_from_response(response)
        if api_total:
            limiter.record_usage(f"doctr_text_{doc_id}", api_prompt, api_completion, estimated=False, reason="doctr_text_llm")
        else:
            limiter.record_usage(f"doctr_text_{doc_id}", estimated_input, estimated_output, estimated=True, reason="doctr_text_llm_estimated")

        status = "success" if content.strip() and "error" not in response else "failed"
        doc, validation_error, warnings = parse_and_validate(content, doc_id, response, model, backend_name, args.prompt_version, latency_ms)
        validation_status = "valid" if doc else "invalid"
        if doc:
            output_json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            valid_docs.append(doc_id)

        error = response.get("error", "") or validation_error
        append_usage(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "document_id": doc_id,
                "model": model,
                "backend_name": backend_name,
                "ocr_text_chars": len(ocr_text),
                "max_tokens": doc_max_tokens,
                "latency_ms": f"{latency_ms:.2f}",
                "status": status,
                "validation_status": validation_status,
                "input_tokens_estimated": estimated_input,
                "output_tokens_estimated": estimated_output,
                "api_reported_prompt_tokens": api_prompt,
                "api_reported_completion_tokens": api_completion,
                "api_reported_total_tokens": api_total,
                "error": error[:500],
            }
        )
        result = {
            "document_id": doc_id,
            "backend_name": backend_name,
            "attempted": "yes",
            "validation_status": validation_status,
            "output_json": str(output_json_path.relative_to(PROJECT_ROOT)) if doc else "",
            "raw_response": str(raw_response_path.relative_to(PROJECT_ROOT)),
            "ocr_paths": ";".join(ocr_paths),
            "warnings": "; ".join(warnings),
            "error": error,
        }
        results.append(result)
        print(f"{doc_id}: {status}, validation={validation_status}, chars={len(content)}")
        if error:
            print(f"  error: {error[:300]}")
        if args.backend == "internal" and doc_id != SMOKE_DOCS[-1]:
            time.sleep(30)

    if valid_docs:
        write_manifest(valid_docs)
    write_status_report(results)
    return 0 if valid_docs else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase D docTR OCR text-only structuring smoke.")
    parser.add_argument("--prompt-version", default="phase_d_text_llm_smoke_v1")
    parser.add_argument("--backend", choices=["internal", "local"], default="internal")
    parser.add_argument("--model-base-url", "--base-url", dest="model_base_url", default="http://localhost:8090/v1")
    parser.add_argument("--model-id", "--model", dest="model_id", default="/model_weight")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--p36-max-tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--include-ocr-lines-as-evidence", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--min-ocr-chars", type=int, default=80)
    parser.add_argument("--tpm-limit", type=int, default=500000)
    parser.add_argument("--rpm-limit", type=int, default=120)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
