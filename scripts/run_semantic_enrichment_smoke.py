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

"""Phase E conservative semantic enrichment smoke.

Reads CanonicalRawDoc raw extraction outputs and writes separate enriched JSON
files plus summary/manual-review reports. This script does not call images,
LLMs, ontology services, or mutate raw extraction files.
"""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOCS = ["p1", "p2", "p45_1", "p45_4"]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs/raw_extractions/internal_qwen3_27b_prompt_v2_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/semantic_enriched/internal_qwen3_27b_phase_e_smoke"
DEFAULT_STATUS_REPORT = PROJECT_ROOT / "reports/phase_e_semantic_enrichment_smoke_status.md"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "reports/phase_e_semantic_enrichment_summary.csv"
DEFAULT_REVIEW_CSV = PROJECT_ROOT / "reports/phase_e_semantic_manual_review_queue.csv"

FREQUENCY_MAP = {
    "od": "once daily",
    "bd": "twice daily",
    "bid": "twice daily",
    "tds": "three times daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "q.i.d": "four times daily",
    "hs": "at bedtime",
    "sos": "as needed",
}

ROUTE_MAP = {
    "po": "oral",
    "oral": "oral",
    "p/o": "oral",
    "e/d": "eye drop",
    "ed": "eye drop",
    "eye drop": "eye drop",
    "inj": "injection",
    "injection": "injection",
    "iv": "intravenous",
    "i.v": "intravenous",
    "i.v.": "intravenous",
    "tab": "tablet",
    "tablet": "tablet",
    "cap": "capsule",
    "capsule": "capsule",
}

LATERALITY_MAP = {
    "re": "right eye",
    "od": "right eye",
    "right eye": "right eye",
    "le": "left eye",
    "os": "left eye",
    "left eye": "left eye",
    "be": "both eyes",
    "ou": "both eyes",
    "both eyes": "both eyes",
}

DIAGNOSIS_HINTS = {
    "dm": "diabetes mellitus",
    "htn": "hypertension",
    "hypothyroid": "hypothyroidism",
    "cholecystitis": "cholecystitis",
    "cholelithiasis": "cholelithiasis",
    "fatty liver": "fatty liver",
    "nuclear cat": "nuclear cataract",
    "cataract": "cataract",
}

ACTION_HINTS = {
    "dilat": "dilation",
    "dilate": "dilation",
    "glass": "glasses advised",
    "phaco": "phacoemulsification",
    "iol": "intraocular lens",
    "adv": "advice",
    "follow": "follow-up",
    "review": "review",
}


def first_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(0) if match else None


def normalize_whitespace(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def detect_frequency(text: str) -> tuple[str | None, str | None]:
    lower = text.lower()
    for token, normalized in FREQUENCY_MAP.items():
        if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lower):
            return token, normalized
    schedule = first_match(r"\b\d\s*-\s*\d(?:\s*-\s*\d)?\b", text)
    if schedule:
        compact = re.sub(r"\s+", "", schedule)
        return schedule, compact
    if "once daily" in lower:
        return "once daily", "once daily"
    return None, None


def detect_route(text: str) -> tuple[str | None, str | None]:
    lower = text.lower()
    if re.search(r"\bdrops?\b", lower):
        return "drops", "eye drop"
    for token, normalized in ROUTE_MAP.items():
        if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lower):
            return token, normalized
    return None, None


def detect_laterality(text: str) -> tuple[str | None, str | None]:
    lower = text.lower()
    for token, normalized in LATERALITY_MAP.items():
        if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lower):
            return token, normalized
    return None, None


def detect_dose(text: str) -> str | None:
    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|mL|cc|IU|units?|amp|tablet|tab|cap)\b",
        r"\b\d+\s*-\s*\d+(?:\s*-\s*\d+)?\b",
        r"\b\d+\s*/\s*\d+\b",
    ]
    for pattern in patterns:
        match = first_match(pattern, text)
        if match:
            return match
    return None


def detect_duration(text: str) -> str | None:
    return first_match(r"\b\d+\s*(?:days?|weeks?|months?|yrs?|years?|yd)\b", text)


def detect_numeric_value(text: str) -> tuple[str | None, str | None]:
    visual_acuity = re.search(r"\b\d+\s*/\s*\d+\b", text)
    if visual_acuity:
        return re.sub(r"\s+", "", visual_acuity.group(0)), None
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(mmhg|mm\s*hg|g/dl|g%|/cummm?|%|mg/dl|ml|iu|units?)?", text, re.IGNORECASE)
    if not match:
        return None, None
    value = match.group(1)
    unit = normalize_whitespace(match.group(2)) if match.group(2) else None
    return value, unit


def medication_candidate(item: dict[str, Any]) -> dict[str, Any]:
    raw = normalize_whitespace(item.get("raw_line_text") or item.get("raw_name") or item.get("evidence_text"))
    evidence = normalize_whitespace(item.get("evidence_text") or raw)
    freq_raw, freq_norm = detect_frequency(" ".join([raw, item.get("raw_frequency") or ""]))
    route_raw, route_norm = detect_route(" ".join([raw, item.get("raw_route") or ""]))
    dose = item.get("raw_dosage") or detect_dose(raw)
    duration = item.get("raw_duration") or detect_duration(raw)
    name = normalize_whitespace(item.get("raw_name"))
    if not name:
        name = re.sub(r"^(rx|tab|cap|inj)\s+", "", raw, flags=re.IGNORECASE).strip()
        name = re.split(r"\b(?:od|bd|tds|tid|qid|hs|sos|po|e/d|inj|injection|iv)\b", name, flags=re.IGNORECASE)[0].strip()
        name = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|cc|amp|units?)\b.*$", "", name, flags=re.IGNORECASE).strip()
    components = {
        "drug_name": name or None,
        "dose": dose,
        "route": route_norm,
        "frequency": freq_norm or item.get("raw_frequency"),
        "duration": duration,
        "instruction": item.get("raw_instruction"),
    }
    dose_is_schedule = bool(dose and re.fullmatch(r"\d\s*-\s*\d(?:\s*-\s*\d)?", str(dose)))
    component_hits = sum(1 for value in [None if dose_is_schedule else dose, route_norm, freq_norm, duration, item.get("raw_instruction")] if value)
    known_form_prefix = bool(re.match(r"^(rx|tab|cap|inj|syp)\b", raw, re.IGNORECASE))
    short_ambiguous_name = bool(name and re.fullmatch(r"[A-Z0-9/+.-]{1,4}", name))
    duration_only_yd = bool(duration and re.search(r"\byd\b", str(duration), re.IGNORECASE) and not route_norm)
    if name and not short_ambiguous_name and not duration_only_yd and (component_hits >= 2 or (known_form_prefix and component_hits >= 1)):
        confidence = 0.65
        status = "candidate"
    elif name and component_hits >= 1:
        confidence = 0.55
        status = "needs_review"
    else:
        confidence = 0.35
        status = "needs_review"
    notes = []
    raw_route = normalize_whitespace(item.get("raw_route"))
    if raw_route and not route_norm:
        notes.append(f"unmapped raw route preserved: {raw_route}")
    if route_raw and route_norm:
        notes.append(f"route pattern: {route_raw}->{route_norm}")
    if freq_raw and freq_norm:
        notes.append(f"frequency pattern: {freq_raw}->{freq_norm}")
    if confidence < 0.6:
        notes.append("low-confidence medication candidate")
    return {
        "raw_text": raw,
        "entity_type": "medication",
        "normalized_candidate": name or "unmapped",
        "components": components,
        "evidence_text": evidence,
        "confidence": round(confidence, 2),
        "mapping_status": status,
        "notes": notes,
    }


def diagnosis_candidate(item: dict[str, Any]) -> dict[str, Any]:
    raw = normalize_whitespace(item.get("raw_text") or item.get("evidence_text"))
    evidence = normalize_whitespace(item.get("evidence_text") or raw)
    lower = raw.lower()
    candidates = []
    notes = []
    for hint, normalized in DIAGNOSIS_HINTS.items():
        if hint in lower:
            candidates.append(normalized)
            notes.append(f"diagnosis hint: {hint}->{normalized}")
    candidates = list(dict.fromkeys(candidates))
    candidate = "; ".join(candidates) if candidates else None
    confidence = 0.68 if candidate else 0.35
    return {
        "raw_text": raw,
        "entity_type": "diagnosis_or_complaint",
        "normalized_candidate": candidate or "unmapped",
        "components": {},
        "evidence_text": evidence,
        "confidence": round(confidence, 2),
        "mapping_status": "candidate" if candidate else "needs_review",
        "notes": notes or ["no deterministic diagnosis normalization"],
    }


def observation_candidate(item: dict[str, Any], entity_type: str = "observation") -> dict[str, Any]:
    raw = normalize_whitespace(item.get("raw_text") or item.get("raw_line_text") or item.get("evidence_text") or item.get("test_name"))
    evidence = normalize_whitespace(item.get("evidence_text") or raw)
    value, unit = detect_numeric_value(raw)
    laterality_raw, laterality = detect_laterality(raw)
    name = normalize_whitespace(item.get("test_name"))
    if not name:
        name = raw
        if value:
            name = normalize_whitespace(raw.split(value)[0])
    if not name and laterality:
        name = "eye observation"
    if re.search(r"\bfundus\b", raw, re.IGNORECASE):
        name = f"fundus {laterality}" if laterality else "fundus"
        if re.search(r"\b(WNL|UNH)\b", raw, re.IGNORECASE):
            value = first_match(r"\b(WNL|UNH)\b", raw)
    if re.search(r"\bmm\s*hg\b|\bmmhg\b", raw, re.IGNORECASE) and re.search(r"\b(right|left)\s+eye\b", raw, re.IGNORECASE):
        name = f"IOP {laterality.title()}" if laterality else "IOP"
    elif re.search(r"\b\d+\s*/\s*\d+\b", raw) and laterality:
        name = f"visual acuity {laterality}"
    elif re.search(r"\bSPH\b|\bCYL\b|\bAXIS\b", raw, re.IGNORECASE) and laterality:
        name = f"refraction {laterality}"
    components = {
        "observation_name": name or None,
        "value": item.get("result") or value,
        "unit": item.get("unit") or unit,
        "laterality": laterality,
        "reference_range": item.get("reference_range"),
    }
    confidence = 0.75 if (components["value"] or laterality or item.get("test_name")) else 0.4
    notes = []
    if laterality_raw:
        notes.append(f"laterality pattern: {laterality_raw}->{laterality}")
    if components["value"]:
        notes.append("numeric value detected")
    return {
        "raw_text": raw,
        "entity_type": entity_type,
        "normalized_candidate": name or "unmapped",
        "components": components,
        "evidence_text": evidence,
        "confidence": round(confidence, 2),
        "mapping_status": "candidate" if confidence >= 0.65 else "needs_review",
        "notes": notes or ["observation preserved without deterministic normalization"],
    }


def action_candidate(item: dict[str, Any], entity_type: str) -> dict[str, Any]:
    raw = normalize_whitespace(item.get("raw_text") or item.get("evidence_text"))
    evidence = normalize_whitespace(item.get("evidence_text") or raw)
    lower = raw.lower()
    candidate = None
    notes = []
    for hint, normalized in ACTION_HINTS.items():
        if hint in lower:
            candidate = normalized
            notes.append(f"action hint: {hint}->{normalized}")
            break
    confidence = 0.65 if candidate and candidate != "advice" else 0.4
    return {
        "raw_text": raw,
        "entity_type": entity_type,
        "normalized_candidate": candidate or "unmapped",
        "components": {},
        "evidence_text": evidence,
        "confidence": round(confidence, 2),
        "mapping_status": "candidate" if confidence >= 0.6 else "needs_review",
        "notes": notes or ["action preserved without deterministic normalization"],
    }


def follow_up_candidate(follow_up: dict[str, Any]) -> dict[str, Any] | None:
    raw = normalize_whitespace(follow_up.get("raw_text") or follow_up.get("date") or follow_up.get("review_after"))
    if not raw:
        return None
    components = {
        "date": follow_up.get("date"),
        "review_after": follow_up.get("review_after") or detect_duration(raw),
    }
    confidence = 0.7 if components["date"] or components["review_after"] else 0.45
    return {
        "raw_text": raw,
        "entity_type": "follow_up",
        "normalized_candidate": "follow-up",
        "components": components,
        "evidence_text": raw,
        "confidence": round(confidence, 2),
        "mapping_status": "candidate" if confidence >= 0.6 else "needs_review",
        "notes": ["follow-up/date pattern"],
    }


def enrich_document(data: dict[str, Any]) -> dict[str, Any]:
    semantic_entities = []
    category_counts = {
        "medications": 0,
        "diagnoses_or_complaints": 0,
        "observations": 0,
        "lab_observations": 0,
        "procedures": 0,
        "advice": 0,
        "follow_up": 0,
    }

    for item in data.get("medications") or []:
        semantic_entities.append(medication_candidate(item))
        category_counts["medications"] += 1
    for item in data.get("complaints_or_diagnosis") or []:
        semantic_entities.append(diagnosis_candidate(item))
        category_counts["diagnoses_or_complaints"] += 1
    for item in data.get("observations") or []:
        semantic_entities.append(observation_candidate(item, "observation"))
        category_counts["observations"] += 1
    for item in data.get("lab_observations") or []:
        semantic_entities.append(observation_candidate(item, "lab_observation"))
        category_counts["lab_observations"] += 1
    for item in data.get("procedures") or []:
        semantic_entities.append(action_candidate(item, "procedure"))
        category_counts["procedures"] += 1
    for item in data.get("advice") or []:
        semantic_entities.append(action_candidate(item, "advice"))
        category_counts["advice"] += 1
    if data.get("follow_up"):
        candidate = follow_up_candidate(data["follow_up"])
        if candidate:
            semantic_entities.append(candidate)
            category_counts["follow_up"] += 1

    review_items = [
        item for item in semantic_entities
        if item["mapping_status"] != "candidate" or item["confidence"] < 0.6
    ]
    return {
        "schema_version": "semantic_enrichment_v0_smoke",
        "document_id": data.get("document_id"),
        "source_raw_extraction": "outputs/raw_extractions/internal_qwen3_27b_prompt_v2_full",
        "raw_document": data,
        "semantic_entities": semantic_entities,
        "semantic_metadata": {
            "method": "rule_based_phase_e_smoke",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ontology_mapping_performed": False,
            "counts": {
                **category_counts,
                "raw_entities_processed": len(semantic_entities),
                "normalized_candidates": sum(1 for item in semantic_entities if item["mapping_status"] == "candidate"),
                "unmapped_or_needs_review": len(review_items),
            },
            "limitations": [
                "Rule-based normalization only",
                "No official RxNorm/SNOMED/LOINC/ICD mapping",
                "Requires human review before semantic correctness claims",
            ],
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_status_report(summary_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]], examples: list[dict[str, Any]]) -> str:
    totals = {
        "documents": len(summary_rows),
        "medications": sum(int(r["medications_processed"]) for r in summary_rows),
        "observations": sum(int(r["observations_processed"]) for r in summary_rows),
        "lab_observations": sum(int(r["lab_observations_processed"]) for r in summary_rows),
        "diagnoses": sum(int(r["diagnoses_or_complaints_processed"]) for r in summary_rows),
        "candidates": sum(int(r["normalized_candidates"]) for r in summary_rows),
        "review": sum(int(r["unmapped_or_needs_review"]) for r in summary_rows),
    }
    good_examples = [e for e in examples if e["mapping_status"] == "candidate"][:8]
    review_examples = review_rows[:8]
    lines = [
        "# Phase E Semantic Enrichment Smoke Status",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "Rule-based semantic enrichment smoke over direct-VLM raw extraction outputs. No image calls, no LLM calls, no official ontology mapping, no FHIR conversion, and no graph construction were performed.",
        "",
        "## Counts",
        "",
        f"- documents processed: {totals['documents']}",
        f"- medications processed: {totals['medications']}",
        f"- observations processed: {totals['observations']}",
        f"- lab observations processed: {totals['lab_observations']}",
        f"- diagnoses/complaints processed: {totals['diagnoses']}",
        f"- normalized candidates generated: {totals['candidates']}",
        f"- unmapped/needs-review items: {totals['review']}",
        "",
        "## Per-Document Summary",
        "",
        "| document_id | raw_entities | candidates | needs_review | medications | observations | labs | diagnoses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['document_id']} | {row['raw_entities_processed']} | {row['normalized_candidates']} | "
            f"{row['unmapped_or_needs_review']} | {row['medications_processed']} | {row['observations_processed']} | "
            f"{row['lab_observations_processed']} | {row['diagnoses_or_complaints_processed']} |"
        )
    lines.extend(["", "## Useful Candidate Examples", ""])
    for item in good_examples:
        lines.append(
            f"- `{item['document_id']}` {item['entity_type']}: `{item['raw_text']}` -> `{item['normalized_candidate']}` "
            f"(confidence {item['confidence']})"
        )
    lines.extend(["", "## Needs Review Examples", ""])
    for item in review_examples:
        lines.append(
            f"- `{item['document_id']}` {item['entity_type']}: `{item['raw_text']}` -> `{item['normalized_candidate']}` "
            f"(confidence {item['confidence']}; {item['reason']})"
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is deterministic pattern extraction only.",
        "- `candidate` does not mean clinically correct or ontology-mapped.",
        "- Unmapped and low-confidence items require human review.",
        "- Official RxNorm/SNOMED CT/LOINC/ICD-10 mapping is deferred.",
        "- Production readiness is not established.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = [doc.strip() for doc in args.document_ids.split(",") if doc.strip()]
    summary_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for doc_id in docs:
        input_path = input_dir / f"{doc_id}.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing raw extraction for {doc_id}: {input_path}")
        data = json.loads(input_path.read_text(encoding="utf-8"))
        enriched = enrich_document(data)
        output_path = output_dir / f"{doc_id}.json"
        output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

        counts = enriched["semantic_metadata"]["counts"]
        summary_rows.append({
            "document_id": doc_id,
            "input_path": str(input_path.relative_to(PROJECT_ROOT)),
            "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            "raw_entities_processed": counts["raw_entities_processed"],
            "normalized_candidates": counts["normalized_candidates"],
            "unmapped_or_needs_review": counts["unmapped_or_needs_review"],
            "medications_processed": counts["medications"],
            "observations_processed": counts["observations"],
            "lab_observations_processed": counts["lab_observations"],
            "diagnoses_or_complaints_processed": counts["diagnoses_or_complaints"],
            "procedures_processed": counts["procedures"],
            "advice_processed": counts["advice"],
            "follow_up_processed": counts["follow_up"],
        })

        for entity in enriched["semantic_entities"]:
            example = {
                "document_id": doc_id,
                "entity_type": entity["entity_type"],
                "raw_text": entity["raw_text"],
                "normalized_candidate": entity["normalized_candidate"],
                "confidence": entity["confidence"],
                "mapping_status": entity["mapping_status"],
            }
            examples.append(example)
            if entity["mapping_status"] != "candidate" or entity["confidence"] < 0.6:
                review_rows.append({
                    **example,
                    "evidence_text": entity.get("evidence_text", ""),
                    "reason": "; ".join(entity.get("notes", [])),
                })
        print(f"{doc_id}: wrote {output_path.relative_to(PROJECT_ROOT)} ({counts['raw_entities_processed']} entities)")

    write_csv(
        PROJECT_ROOT / args.summary_csv,
        summary_rows,
        [
            "document_id", "input_path", "output_path", "raw_entities_processed",
            "normalized_candidates", "unmapped_or_needs_review", "medications_processed",
            "observations_processed", "lab_observations_processed",
            "diagnoses_or_complaints_processed", "procedures_processed", "advice_processed",
            "follow_up_processed",
        ],
    )
    write_csv(
        PROJECT_ROOT / args.manual_review_csv,
        review_rows,
        ["document_id", "entity_type", "raw_text", "normalized_candidate", "confidence", "mapping_status", "evidence_text", "reason"],
    )
    status = build_status_report(summary_rows, review_rows, examples)
    status_path = PROJECT_ROOT / args.status_report
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(status, encoding="utf-8")
    print(f"wrote {status_path.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase E rule-based semantic enrichment smoke.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--document-ids", default=",".join(DEFAULT_DOCS))
    parser.add_argument("--status-report", default=str(DEFAULT_STATUS_REPORT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV.relative_to(PROJECT_ROOT)))
    parser.add_argument("--manual-review-csv", default=str(DEFAULT_REVIEW_CSV.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
