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

"""Phase D docTR OCR-only smoke runner.

Runs OCR page by page for the approved smoke set and writes raw text, raw
docTR exports, usage logs, and a small qualitative status report.
"""

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OCR_OUTPUT_DIR = PROJECT_ROOT / "outputs/ocr_transcriptions/doctr_baseline"
RAW_OUTPUT_DIR = PROJECT_ROOT / "outputs/raw_responses/doctr_baseline"
USAGE_LOG = PROJECT_ROOT / "logs/doctr_ocr_usage.csv"
STATUS_REPORT = PROJECT_ROOT / "reports/phase_d_doctr_ocr_smoke_status.md"


@dataclass(frozen=True)
class SmokePage:
    document_id: str
    page_number: int
    image_path: str


SMOKE_PAGES = [
    SmokePage("p1", 1, "prescriptions/p1.jpeg"),
    SmokePage("p36_1", 1, "prescriptions/p36/p36_1/presc_image16902685410.jpg"),
    SmokePage("p36_1", 2, "prescriptions/p36/p36_1/presc_image16902685411.jpg"),
    SmokePage("p2", 1, "prescriptions/p2.jpeg"),
    SmokePage("p45_4", 1, "prescriptions/p45/p45_4/Adobe Scan May 15, 2026 (3)_page-0001.jpg"),
]

CLINICAL_PATTERNS = [
    r"\btab\b", r"\bcap\b", r"\bdrop", r"\bmg\b", r"\bml\b", r"\bod\b", r"\bbd\b",
    r"\btid\b", r"\bqid\b", r"\bre\b", r"\ble\b", r"\bbe\b", r"\biop\b",
    r"\bdiagn", r"\bcomplain", r"\bfollow", r"\breview", r"\bdate\b", r"\bdr\b",
]


def append_usage(row: dict[str, Any]) -> None:
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "document_id",
        "page_number",
        "image_path",
        "image_width",
        "image_height",
        "status",
        "latency_ms",
        "output_chars",
        "error",
    ]
    exists = USAGE_LOG.exists() and USAGE_LOG.stat().st_size > 0
    with USAGE_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def page_export_to_text(page_export: dict[str, Any]) -> str:
    lines: list[str] = []
    for block in page_export.get("blocks", []) or []:
        block_lines: list[str] = []
        for line in block.get("lines", []) or []:
            words = [word.get("value", "") for word in line.get("words", []) or []]
            text = " ".join(word for word in words if word).strip()
            if text:
                block_lines.append(text)
        if block_lines:
            if lines:
                lines.append("")
            lines.extend(block_lines)
    return "\n".join(lines).strip()


def compact_error(exc: Exception) -> str:
    return re.sub(r"\s+", " ", str(exc)).strip()[:500]


def clinical_examples(text: str, limit: int = 8) -> list[str]:
    examples: list[str] = []
    lower = text.lower()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.+/-]{1,}", text)
    for pattern in CLINICAL_PATTERNS:
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if match:
            examples.append(text[match.start():match.end()])
    for token in tokens:
        if re.search(r"\d", token) or token.lower() in {"re", "le", "be", "iop", "mg", "ml", "tab", "cap"}:
            examples.append(token)
        if len(examples) >= limit:
            break
    seen = set()
    deduped = []
    for item in examples:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:limit]


def quality_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("text", "") or ""
    chars = len(text)
    alpha_count = sum(ch.isalpha() for ch in text)
    alpha_ratio = alpha_count / max(chars, 1)
    lines = [line for line in text.splitlines() if line.strip()]
    examples = clinical_examples(text)
    looks_readable = row["status"] == "success" and chars >= 40 and alpha_ratio >= 0.35
    preserves_line_breaks = len(lines) >= 3
    recommended = looks_readable and (chars >= 80 or bool(examples))
    major_noise = ""
    if row["status"] != "success":
        major_noise = row.get("error", "")
    elif chars == 0:
        major_noise = "No OCR text produced"
    elif not looks_readable:
        major_noise = "Very short or low alphabetic OCR output"
    elif alpha_ratio < 0.45:
        major_noise = "OCR contains substantial non-letter noise"
    return {
        "document_id": row["document_id"],
        "page_number": row["page_number"],
        "output_chars": chars,
        "looks_readable_yes_no": "yes" if looks_readable else "no",
        "preserves_line_breaks_yes_no": "yes" if preserves_line_breaks else "no",
        "clinical_terms_detected_examples": ", ".join(examples) if examples else "",
        "major_noise_or_failure": major_noise,
        "recommended_for_text_llm_yes_no": "yes" if recommended else "no",
    }


def write_combined_files(rows: list[dict[str, Any]]) -> None:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_doc.setdefault(row["document_id"], []).append(row)
    for doc_id, doc_rows in by_doc.items():
        if len(doc_rows) <= 1:
            continue
        doc_rows.sort(key=lambda item: item["page_number"])
        chunks = []
        for row in doc_rows:
            text = (row.get("text") or "").strip()
            chunks.append(f"--- page {row['page_number']} ---\n{text}")
        (OCR_OUTPUT_DIR / f"{doc_id}_combined.txt").write_text("\n\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def write_status_report(rows: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> None:
    STATUS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    successful = [row for row in rows if row["status"] == "success"]
    p36_rows = [row for row in rows if row["document_id"] == "p36_1"]
    p36_snapshots = [row for row in snapshots if row["document_id"] == "p36_1"]
    p36_usable = any(row["recommended_for_text_llm_yes_no"] == "yes" for row in p36_snapshots)
    p1_usable = any(row["document_id"] == "p1" and row["recommended_for_text_llm_yes_no"] == "yes" for row in snapshots)
    p2_usable = any(row["document_id"] == "p2" and row["recommended_for_text_llm_yes_no"] == "yes" for row in snapshots)
    dates_readable = any(
        re.search(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b20\d{2}\b", row.get("text", ""))
        for row in rows
    )
    ophthalmology_markers_preserved = any(
        re.search(r"\b(RE|LE|BE|OD|OS|OU|IOP)\b", row.get("text", ""), re.IGNORECASE)
        for row in rows
    )

    lines = [
        "# Phase D docTR OCR Smoke Status",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "OCR-only smoke run using docTR. No text-to-JSON structuring, semantic normalization, ontology mapping, Gemma run, or p36_1 direct VLM retry is included in this report.",
        "",
        "## Execution Status",
        "",
        f"- docTR import/run status: {'success' if len(successful) == len(rows) else 'partial_or_failed'}",
        f"- Pages attempted: {len(rows)}",
        f"- Pages with OCR text: {sum(1 for row in rows if len(row.get('text', '')) > 0)}",
        f"- Can docTR recover any usable text from p36_1? {'Yes' if p36_usable else 'No'}",
        "",
        "## OCR Output Lengths",
        "",
        "| document_id | page | status | output_chars | text_output | raw_response | error |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        text_path = row.get("text_output_path", "")
        raw_path = row.get("raw_output_path", "")
        lines.append(
            f"| {row['document_id']} | {row['page_number']} | {row['status']} | {len(row.get('text', ''))} | "
            f"`{text_path}` | `{raw_path}` | {row.get('error', '')} |"
        )

    lines.extend([
        "",
        "## Manual Quality Snapshot",
        "",
        "| row | output_chars | looks_readable_yes_no | preserves_line_breaks_yes_no | clinical_terms_detected_examples | major_noise_or_failure | recommended_for_text_llm_yes_no |",
        "|---|---:|---|---|---|---|---|",
    ])
    for snap in snapshots:
        row_name = snap["document_id"] if snap["page_number"] == 1 and snap["document_id"] not in {"p36_1", "p45_4"} else f"{snap['document_id']} page {snap['page_number']}"
        lines.append(
            f"| {row_name} | {snap['output_chars']} | {snap['looks_readable_yes_no']} | "
            f"{snap['preserves_line_breaks_yes_no']} | {snap['clinical_terms_detected_examples']} | "
            f"{snap['major_noise_or_failure']} | {snap['recommended_for_text_llm_yes_no']} |"
        )

    lines.extend([
        "",
        "## Qualitative Notes",
        "",
        f"- Medication/clinical names readable: {'yes in at least one page' if any(s['clinical_terms_detected_examples'] for s in snapshots) else 'not clearly detected by smoke heuristics'}",
        f"- Dates readable: {'yes' if dates_readable else 'not clearly detected'}",
        f"- Ophthalmology markers preserved: {'yes' if ophthalmology_markers_preserved else 'not clearly detected'}",
        f"- Line breaks preserved: {'yes' if any(s['preserves_line_breaks_yes_no'] == 'yes' for s in snapshots) else 'no'}",
        f"- p36_1 page-wise readability: page 1 = {next((s['looks_readable_yes_no'] for s in p36_snapshots if s['page_number'] == 1), 'no')}; page 2 = {next((s['looks_readable_yes_no'] for s in p36_snapshots if s['page_number'] == 2), 'no')}",
        "",
        "## Stop/Proceed Assessment",
        "",
    ])
    if p1_usable and p36_usable:
        lines.append("- OCR is usable for p1 and at least one p36_1 page, so text-only LLM structuring smoke is eligible to proceed.")
    elif not p36_usable:
        lines.append("- Stop rule triggered for structuring: p36_1 produced no usable OCR text.")
    elif not p1_usable or not p2_usable:
        lines.append("- OCR is weak on p1/p2; do not scale beyond smoke without manual review.")
    lines.append("- This is a smoke/diagnostic report only. It does not establish production readiness.")
    lines.append("")

    STATUS_REPORT.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except Exception as exc:
        RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RAW_OUTPUT_DIR.joinpath("import_error.json").write_text(
            json.dumps({"status": "failed_import", "error": compact_error(exc)}, indent=2),
            encoding="utf-8",
        )
        print(f"docTR import failed: {exc}")
        return 2

    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictor = ocr_predictor(
        det_arch=args.det_arch,
        reco_arch=args.reco_arch,
        pretrained=True,
        assume_straight_pages=not args.detect_orientation,
    )

    rows: list[dict[str, Any]] = []
    for page in SMOKE_PAGES:
        abs_image_path = PROJECT_ROOT / page.image_path
        output_txt = OCR_OUTPUT_DIR / f"{page.document_id}_page{page.page_number}.txt"
        raw_json = RAW_OUTPUT_DIR / f"{page.document_id}_page{page.page_number}.json"
        status = "success"
        error = ""
        text = ""
        width = ""
        height = ""
        start = time.time()

        try:
            with Image.open(abs_image_path) as img:
                width, height = img.size
            doc = DocumentFile.from_images([str(abs_image_path)])
            result = predictor(doc)
            exported = result.export()
            page_export = (exported.get("pages") or [{}])[0]
            text = page_export_to_text(page_export)
            output_txt.write_text(text.rstrip() + "\n", encoding="utf-8")
            raw_json.write_text(
                json.dumps(
                    {
                        "document_id": page.document_id,
                        "page_number": page.page_number,
                        "image_path": page.image_path,
                        "status": status,
                        "text": text,
                        "doctr_export": exported,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            status = "failed"
            error = compact_error(exc)
            raw_json.write_text(
                json.dumps(
                    {
                        "document_id": page.document_id,
                        "page_number": page.page_number,
                        "image_path": page.image_path,
                        "status": status,
                        "error": error,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        latency_ms = (time.time() - start) * 1000
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_id": page.document_id,
            "page_number": page.page_number,
            "image_path": page.image_path,
            "image_width": width,
            "image_height": height,
            "status": status,
            "latency_ms": f"{latency_ms:.2f}",
            "output_chars": len(text),
            "error": error,
            "text": text,
            "text_output_path": str(output_txt.relative_to(PROJECT_ROOT)),
            "raw_output_path": str(raw_json.relative_to(PROJECT_ROOT)),
        }
        append_usage(row)
        rows.append(row)
        print(f"{page.document_id} page {page.page_number}: {status}, chars={len(text)}, latency_ms={latency_ms:.1f}")
        if error:
            print(f"  error: {error}")

    write_combined_files(rows)
    snapshots = [quality_snapshot(row) for row in rows]
    write_status_report(rows, snapshots)
    return 0 if all(row["status"] == "success" for row in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase D docTR OCR-only smoke.")
    parser.add_argument("--det-arch", default="db_resnet50")
    parser.add_argument("--reco-arch", default="crnn_vgg16_bn")
    parser.add_argument("--detect-orientation", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
