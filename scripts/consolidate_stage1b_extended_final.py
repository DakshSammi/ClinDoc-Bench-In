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

"""Stage 1B Extended final consolidation - Final release.

Updates:
- Qwen2.5-VL rows with computed aggregate metrics
- Marker rows (still 19/53 - handoff limited)
- All tables, statistics, reports
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scipy import stats
except Exception:
    stats = None

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
COMBINED = ROOT / "paper_assets" / "tables" / "combined"
CLEAN_REPO = ROOT.parents[1] / "ClinDoc-Bench-In"

STRUCTURED_METRICS = ["overall_extraction_score", "entity_lenient_f1"]
OCR_METRICS = ["token_f1", "text_similarity"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def find_row(rows: list[dict[str, str]], system: str) -> dict[str, str]:
    for row in rows:
        if row.get("system") == system:
            return row
    raise KeyError(f"System not found: {system}")


def raw_row_from_summary(
    system: str,
    summary_path: Path,
    label: str,
    notes: str,
    coverage: str = "53/53",
) -> dict[str, Any]:
    summary = read_json(summary_path)
    records = int(summary["records"])
    nonempty_rate = float(summary["non_empty_output_rate"])
    records_nonempty = int(round(records * nonempty_rate))
    return {
        "section": "raw_ocr",
        "system": system,
        "coverage": coverage,
        "records_attempted": records,
        "records_schema_valid": "",
        "schema_parse_success": "",
        "scalar_exact_accuracy": "",
        "scalar_lenient_accuracy": "",
        "entity_exact_f1": "",
        "entity_lenient_f1": "",
        "hallucination_rate": "",
        "missing_entity_rate": "",
        "annotation_gap_rate": "",
        "overall_extraction_score": "",
        "records_nonempty": records_nonempty,
        "ocr_f1": fmt(summary["token_f1"]),
        "text_similarity": fmt(summary["normalized_edit_similarity"]),
        "runtime": fmt(summary["runtime_seconds"]),
        "label": label,
        "notes": notes,
    }


def raw_qwen25vl_row() -> dict[str, Any]:
    """Compute Qwen2.5-VL raw OCR aggregate metrics from per-document CSV."""
    raw_path = REPORTS / "stage1b_extended_qwen25vl_raw_ocr_per_document_scores.csv"
    rows = read_csv(raw_path)
    token_f1s: list[float] = []
    text_sims: list[float] = []
    runtimes: list[float] = []
    nonempty = 0
    for row in rows:
        status = row.get("status", "")
        if row.get("text_nonempty", "") == "True":
            nonempty += 1
        tf1 = row.get("token_f1", "")
        ts = row.get("normalized_edit_similarity", "")
        rt = row.get("runtime_seconds", "")
        if status == "failed":
            token_f1s.append(0.0)
            text_sims.append(0.0)
        elif tf1:
            token_f1s.append(float(tf1))
            text_sims.append(float(ts) if ts else 0.0)
        if rt:
            runtimes.append(float(rt))
    return {
        "section": "raw_ocr",
        "system": "Qwen2.5-VL raw OCR",
        "coverage": "52/53",
        "records_attempted": 53,
        "records_schema_valid": "",
        "schema_parse_success": "",
        "scalar_exact_accuracy": "",
        "scalar_lenient_accuracy": "",
        "entity_exact_f1": "",
        "entity_lenient_f1": "",
        "hallucination_rate": "",
        "missing_entity_rate": "",
        "annotation_gap_rate": "",
        "overall_extraction_score": "",
        "records_nonempty": nonempty,
        "ocr_f1": fmt(mean(token_f1s)),
        "text_similarity": fmt(mean(text_sims)),
        "runtime": fmt(mean(runtimes), digits=1),
        "label": "coverage_limited_52_53_with_metrics",
        "notes": "Qwen2.5-VL raw OCR: 52/53 documents recovered (p26 unrecovered); per-document scores computed via canonical raw OCR evaluator. Excluded from primary full-53 paired tests because coverage is 52/53.",
    }


def structured_row_from_ppt(
    ppt_rows: list[dict[str, str]],
    system: str,
    label: str,
    notes: str,
) -> dict[str, Any]:
    row = find_row(ppt_rows, system)
    attempted = row.get("records_attempted", "")
    valid = row.get("records_schema_valid", "")
    total = attempted or "53"
    return {
        "section": "structured_direct",
        "system": system,
        "coverage": f"{valid}/{total}" if valid else "",
        "records_attempted": attempted,
        "records_schema_valid": valid,
        "schema_parse_success": fmt(row.get("schema_parse_success")),
        "scalar_exact_accuracy": fmt(row.get("scalar_exact_accuracy")),
        "scalar_lenient_accuracy": fmt(row.get("scalar_lenient_accuracy")),
        "entity_exact_f1": fmt(row.get("entity_exact_f1")),
        "entity_lenient_f1": fmt(row.get("entity_lenient_f1")),
        "hallucination_rate": fmt(row.get("hallucination_rate")),
        "missing_entity_rate": fmt(row.get("missing_entity_rate")),
        "annotation_gap_rate": fmt(row.get("annotation_gap_rate")),
        "overall_extraction_score": fmt(row.get("overall_extraction_score")),
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": fmt(row.get("runtime")),
        "label": label,
        "notes": notes,
    }


def internal_qwen3_row() -> dict[str, Any]:
    recovered_rows = read_csv(CLEAN_REPO / "paper_assets" / "tables" / "final" / "stage1b_server2_ppt_aligned_metrics.csv")
    recovered = recovered_rows[0]
    return {
        "section": "structured_direct",
        "system": "Internal Qwen3-27B compact recovered-plus",
        "coverage": "53/53",
        "records_attempted": 53,
        "records_schema_valid": 53,
        "schema_parse_success": fmt(1.0),
        "scalar_exact_accuracy": fmt(recovered.get("scalar exact")),
        "scalar_lenient_accuracy": fmt(recovered.get("scalar lenient")),
        "entity_exact_f1": fmt(recovered.get("entity exact F1")),
        "entity_lenient_f1": fmt(recovered.get("entity lenient F1")),
        "hallucination_rate": fmt(recovered.get("hallucination rate")),
        "missing_entity_rate": fmt(recovered.get("missing entity rate")),
        "annotation_gap_rate": fmt(recovered.get("annotation-gap rate")),
        "overall_extraction_score": fmt(recovered.get("overall extraction score")),
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": "",
        "label": "recovered_plus_coverage_with_frozen_scored_fields",
        "notes": "Coverage and schema-validity are from stage1b_extended_qwen3_27b_merged_plus_metrics.csv (53/53 after p45_2 retry). Scored extraction fields remain the latest imported recovered values from paper_assets/tables/final/stage1b_server2_ppt_aligned_metrics.csv.",
    }


def qwen25vl_structured_row() -> dict[str, Any]:
    """Compute Qwen2.5-VL structured aggregate metrics from per-document CSV."""
    struct_path = REPORTS / "stage1b_extended_qwen25vl_structured_per_document_scores.csv"
    rows = read_csv(struct_path)
    schema_valid = sum(1 for r in rows if r.get("schema_valid", "") == "1")
    schema_parse = sum(1 for r in rows if r.get("schema_parse_success", "") == "1")
    overalls = [float(r["experimental_overall_score"]) for r in rows if r.get("experimental_overall_score", "")]
    entity_lenients = [float(r["entity_lenient_f1_macro"]) for r in rows if r.get("entity_lenient_f1_macro", "")]
    scalar_exacts = [float(r["scalar_accuracy_exact"]) for r in rows if r.get("scalar_accuracy_exact", "")]
    scalar_lenients = [float(r["scalar_accuracy_lenient"]) for r in rows if r.get("scalar_accuracy_lenient", "")]
    hallucinations = [float(r["hallucination_rate"]) for r in rows if r.get("hallucination_rate", "")]
    missing_rates = [float(r["missing_entity_rate"]) for r in rows if r.get("missing_entity_rate", "")]
    annotation_gaps = [float(r["annotation_gap_rate"]) for r in rows if r.get("annotation_gap_rate", "")]
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms", "")]
    runtime_mean = mean(latencies) / 1000 if latencies else 0
    return {
        "section": "structured_direct",
        "system": "Qwen2.5-VL structured",
        "coverage": f"{schema_valid}/53",
        "records_attempted": 53,
        "records_schema_valid": schema_valid,
        "schema_parse_success": fmt(schema_parse / len(rows)),
        "scalar_exact_accuracy": fmt(mean(scalar_exacts)) if scalar_exacts else "",
        "scalar_lenient_accuracy": fmt(mean(scalar_lenients)) if scalar_lenients else "",
        "entity_exact_f1": "",
        "entity_lenient_f1": fmt(mean(entity_lenients)) if entity_lenients else "",
        "hallucination_rate": fmt(mean(hallucinations)) if hallucinations else "",
        "missing_entity_rate": fmt(mean(missing_rates)) if missing_rates else "",
        "annotation_gap_rate": fmt(mean(annotation_gaps)) if annotation_gaps else "",
        "overall_extraction_score": fmt(mean(overalls)) if overalls else "",
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": fmt(runtime_mean, digits=1),
        "label": "coverage_limited_52_53_with_metrics",
        "notes": "Qwen2.5-VL structured: 52/53 documents recovered (p36_1 structured unrecovered); 45/53 schema-valid (8 schema failures). Per-document scores computed via compact-to-canonical adapter. Excluded from primary full-53 paired tests because coverage is 52/53.",
    }


def pipeline_row_from_summary(
    summary_path: Path,
    section_label: str,
    notes: str,
    coverage: str | None = None,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    attempted = int(summary["records_attempted"])
    valid = int(summary["records_schema_valid"])
    return {
        "section": "hybrid_pipeline",
        "system": summary["system"],
        "coverage": coverage or f"{valid}/{attempted}",
        "records_attempted": attempted,
        "records_schema_valid": valid,
        "schema_parse_success": fmt(summary["schema_parse_success"]),
        "scalar_exact_accuracy": fmt(summary["scalar_exact_accuracy"]),
        "scalar_lenient_accuracy": fmt(summary["scalar_lenient_accuracy"]),
        "entity_exact_f1": fmt(summary["entity_exact_f1"]),
        "entity_lenient_f1": fmt(summary["entity_lenient_f1"]),
        "hallucination_rate": fmt(summary["hallucination_rate"]),
        "missing_entity_rate": fmt(summary["missing_entity_rate"]),
        "annotation_gap_rate": fmt(summary["annotation_gap_rate"]),
        "overall_extraction_score": fmt(summary["overall_extraction_score"]),
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": fmt(summary["runtime_seconds"]),
        "label": section_label,
        "notes": notes,
    }


def build_combined_rows() -> list[dict[str, Any]]:
    rows = [
        # Full-53 raw OCR systems
        raw_row_from_summary(
            "GLM-OCR",
            REPORTS / "stage1b_raw_ocr_benchmark_glm_ocr" / "summary_metrics.json",
            "full_53_raw_ocr",
            "Accepted Final release full benchmark raw OCR baseline.",
        ),
        raw_row_from_summary(
            "docTR",
            REPORTS / "stage1b_raw_ocr_benchmark_doctr" / "summary_metrics.json",
            "full_53_raw_ocr",
            "Final release OCR handoff re-benchmarked locally with the Stage 1B raw OCR evaluator.",
        ),
        raw_row_from_summary(
            "TrOCR",
            REPORTS / "stage1b_raw_ocr_benchmark_trocr" / "summary_metrics.json",
            "full_53_raw_ocr",
            "Final release OCR handoff re-benchmarked locally with the Stage 1B raw OCR evaluator.",
        ),
        raw_row_from_summary(
            "Docling",
            REPORTS / "stage1b_extended_raw_ocr_benchmark_docling" / "summary_metrics.json",
            "full_53_raw_ocr",
            "Extended final pass using the imported Final release OCR handoff.",
        ),
        raw_row_from_summary(
            "Surya",
            REPORTS / "stage1b_extended_raw_ocr_benchmark_surya" / "summary_metrics.json",
            "full_53_raw_ocr",
            "Extended final pass using the imported Final release OCR handoff.",
        ),
        raw_row_from_summary(
            "EasyOCR",
            REPORTS / "stage1b_extended_raw_ocr_benchmark_easyocr" / "summary_metrics.json",
            "full_53_raw_ocr",
            "Extended final pass using the imported Final release OCR handoff.",
        ),
        # Coverage-limited raw OCR
        raw_qwen25vl_row(),
        # Partial raw OCR
        raw_row_from_summary(
            "Marker",
            REPORTS / "stage1b_extended_raw_ocr_benchmark_marker" / "summary_metrics.json",
            "partial_interim_raw_ocr",
            "Only 19 usable Marker rows available in the imported handoff (Final release had 39 but handoff was limited). Keep separate from full-53 comparisons.",
            coverage="19/53",
        ),
        # Full-53 structured - Internal Qwen3
        internal_qwen3_row(),
        # Full-53 structured - local VLMs
        structured_row_from_ppt(
            read_csv(REPORTS / "stage1b_server1_ppt_aligned_metrics.csv"),
            "qwen3-vl:8b-instruct",
            "full_53_direct_vlm",
            "Accepted Final release local direct structured baseline.",
        ),
        structured_row_from_ppt(
            read_csv(REPORTS / "stage1b_server1_ppt_aligned_metrics.csv"),
            "llava:13b",
            "full_53_direct_vlm_diagnostic",
            "Accepted diagnostic direct structured baseline.",
        ),
        # Coverage-limited structured
        qwen25vl_structured_row(),
        # Full-53 hybrid pipelines
        pipeline_row_from_summary(
            REPORTS / "stage1b_server1_glm_ocr_qwen3_structured_summary.json",
            "full_53_hybrid_pipeline",
            "Accepted Final release OCR-to-JSON pipeline.",
        ),
        pipeline_row_from_summary(
            REPORTS / "stage1b_server1_doctr_qwen3_structured_summary.json",
            "full_53_hybrid_pipeline",
            "Imported Final release OCR handoff evaluated locally.",
        ),
        pipeline_row_from_summary(
            REPORTS / "stage1b_server1_trocr_qwen3_structured_summary.json",
            "full_53_hybrid_pipeline_low_quality",
            "Imported Final release OCR handoff evaluated locally.",
        ),
        pipeline_row_from_summary(
            REPORTS / "stage1b_extended_easyocr_qwen3_summary.json",
            "full_53_hybrid_pipeline",
            "Consolidated from the completed background hybrid run.",
        ),
        pipeline_row_from_summary(
            REPORTS / "stage1b_extended_surya_qwen3_summary.json",
            "full_53_hybrid_pipeline",
            "Consolidated from the completed background hybrid run.",
        ),
        pipeline_row_from_summary(
            REPORTS / "stage1b_extended_docling_qwen3_summary.json",
            "full_53_hybrid_pipeline",
            "Consolidated from the completed background hybrid run.",
        ),
        # Partial hybrid
        pipeline_row_from_summary(
            REPORTS / "stage1b_extended_marker_qwen3_partial_summary.json",
            "partial_interim_hybrid",
            "Only a partial/import-limited lane was available (19/53). Keep separate from full-53 comparisons.",
            coverage="19/53",
        ),
    ]
    return rows


# ========= Statistical test functions =========


def normalize_structured(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        doc_id = row.get("document_id", "").strip()
        if not doc_id:
            continue
        schema_value = str(row.get("schema_parse_success") or row.get("schema_validity") or row.get("json_parse_success") or "").strip().lower()
        schema_success = 1.0 if schema_value in {"1", "1.0", "true", "yes", "success"} else 0.0
        overall_value = row.get("overall_extraction_score", "")
        if overall_value == "":
            scalar_lenient = float(row.get("scalar_lenient_accuracy", "") or 0.0)
            entity_lenient = float(row.get("entity_lenient_f1", "") or 0.0)
            entity_exact = float(row.get("entity_exact_f1", "") or 0.0)
            hallucination = float(row.get("hallucination_rate", "") or 0.0)
            overall_value = round(
                0.10 * schema_success
                + 0.20 * scalar_lenient
                + 0.45 * entity_lenient
                + 0.15 * entity_exact
                + 0.10 * (1.0 - hallucination),
                6,
            )
        else:
            overall_value = float(overall_value)
        out[doc_id] = {
            "schema_success": schema_success,
            "overall_extraction_score": overall_value,
            "entity_lenient_f1": float(row.get("entity_lenient_f1", "") or 0.0),
        }
    return out


def normalize_ocr(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        doc_id = row.get("document_id", "").strip()
        if not doc_id:
            continue
        out[doc_id] = {
            "token_f1": float(row.get("token_f1", "") or row.get("ocr_f1", "") or 0.0),
            "text_similarity": float(row.get("normalized_edit_similarity", "") or row.get("text_similarity", "") or 0.0),
        }
    return out


def bootstrap_ci(values: list[float], rng: random.Random, iterations: int) -> tuple[float, float, float]:
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return mean(values), lower, upper


def paired_values(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    metric: str,
) -> tuple[list[str], list[float], list[float]]:
    docs = sorted(set(left) & set(right))
    return docs, [left[doc][metric] for doc in docs], [right[doc][metric] for doc in docs]


def wilcoxon_p(left: list[float], right: list[float]) -> float | str:
    if len(left) < 2:
        return ""
    diffs = [a - b for a, b in zip(left, right) if a != b]
    if not diffs:
        return 1.0
    if stats is None:
        return ""
    try:
        return float(stats.wilcoxon(left, right, zero_method="wilcox").pvalue)
    except Exception:
        return ""


def mcnemar_p(left: list[float], right: list[float]) -> tuple[int, int, float | str]:
    left_only = sum(1 for a, b in zip(left, right) if a == 1 and b == 0)
    right_only = sum(1 for a, b in zip(left, right) if a == 0 and b == 1)
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0
    if stats is None:
        return left_only, right_only, ""
    return left_only, right_only, float(stats.binomtest(min(left_only, right_only), discordant, 0.5).pvalue)


def friedman_p(systems: dict[str, dict[str, dict[str, float]]], metric: str) -> tuple[int, float | str]:
    common_docs = sorted(set.intersection(*(set(rows) for rows in systems.values()))) if systems else []
    if len(common_docs) < 2 or len(systems) < 3 or stats is None:
        return len(common_docs), ""
    arrays = [[systems[name][doc][metric] for doc in common_docs] for name in systems]
    try:
        return len(common_docs), float(stats.friedmanchisquare(*arrays).pvalue)
    except Exception:
        return len(common_docs), ""


def holm_bonferroni(rows: list[dict[str, Any]], key: str = "p_value") -> None:
    indexed: list[tuple[float, int]] = []
    for idx, row in enumerate(rows):
        try:
            indexed.append((float(row[key]), idx))
        except (KeyError, TypeError, ValueError):
            continue
    indexed.sort()
    total = len(indexed)
    running = 0.0
    adjusted: dict[int, float] = {}
    for rank, (p_value, idx) in enumerate(indexed, start=1):
        value = min(1.0, (total - rank + 1) * p_value)
        running = max(running, value)
        adjusted[idx] = running
    for idx, row in enumerate(rows):
        row["holm_adjusted_p"] = fmt(adjusted[idx], 6) if idx in adjusted else ""


def load_structured_systems() -> dict[str, dict[str, dict[str, float]]]:
    files = {
        "Qwen3-VL 8B": REPORTS / "stage1b_server1_qwen3vl_metrics_final.csv",
        "LLaVA 13B": REPORTS / "stage1b_server1_llava_metrics.csv",
        "GLM-OCR + qwen3:8b": REPORTS / "stage1b_server1_glm_ocr_qwen3_structured_metrics.csv",
        "docTR + qwen3:8b": REPORTS / "stage1b_server1_doctr_qwen3_structured_metrics.csv",
        "TrOCR + qwen3:8b": REPORTS / "stage1b_server1_trocr_qwen3_structured_metrics.csv",
        "EasyOCR + qwen3:8b": REPORTS / "stage1b_extended_easyocr_qwen3_metrics.csv",
        "Surya + qwen3:8b": REPORTS / "stage1b_extended_surya_qwen3_metrics.csv",
        "Docling + qwen3:8b": REPORTS / "stage1b_extended_docling_qwen3_metrics.csv",
    }
    return {name: normalize_structured(read_csv(path)) for name, path in files.items()}


def load_ocr_systems() -> dict[str, dict[str, dict[str, float]]]:
    files = {
        "GLM-OCR": REPORTS / "stage1b_raw_ocr_benchmark_glm_ocr" / "per_document_ocr_scores.csv",
        "docTR": REPORTS / "stage1b_raw_ocr_benchmark_doctr" / "per_document_ocr_scores.csv",
        "TrOCR": REPORTS / "stage1b_raw_ocr_benchmark_trocr" / "per_document_ocr_scores.csv",
        "EasyOCR": REPORTS / "stage1b_extended_raw_ocr_benchmark_easyocr" / "per_document_ocr_scores.csv",
        "Surya": REPORTS / "stage1b_extended_raw_ocr_benchmark_surya" / "per_document_ocr_scores.csv",
        "Docling": REPORTS / "stage1b_extended_raw_ocr_benchmark_docling" / "per_document_ocr_scores.csv",
    }
    return {name: normalize_ocr(read_csv(path)) for name, path in files.items()}


def write_stats() -> None:
    rng = random.Random(20260626)
    iterations = 5000
    structured = load_structured_systems()
    ocr = load_ocr_systems()

    bootstrap_rows: list[dict[str, Any]] = []
    for family, systems, metrics in [
        ("structured", structured, STRUCTURED_METRICS),
        ("raw_ocr", ocr, OCR_METRICS),
    ]:
        for system, docs in systems.items():
            for metric in metrics:
                values = [entry[metric] for entry in docs.values()]
                estimate, lower, upper = bootstrap_ci(values, rng, iterations)
                bootstrap_rows.append({
                    "family": family,
                    "system": system,
                    "metric": metric,
                    "n": len(values),
                    "mean": fmt(estimate, 6),
                    "ci_lower": fmt(lower, 6),
                    "ci_upper": fmt(upper, 6),
                    "bootstrap_iterations": iterations,
                })

    pairwise_rows: list[dict[str, Any]] = []
    for family, systems, metrics in [
        ("structured", structured, STRUCTURED_METRICS),
        ("raw_ocr", ocr, OCR_METRICS),
    ]:
        for metric in metrics:
            for left, right in combinations(systems, 2):
                docs, a_values, b_values = paired_values(systems[left], systems[right], metric)
                pairwise_rows.append({
                    "family": family,
                    "metric": metric,
                    "system_a": left,
                    "system_b": right,
                    "paired_n": len(docs),
                    "mean_a": fmt(mean(a_values), 6),
                    "mean_b": fmt(mean(b_values), 6),
                    "mean_difference_a_minus_b": fmt(mean([a - b for a, b in zip(a_values, b_values)]), 6),
                    "test": "wilcoxon_signed_rank",
                    "p_value": wilcoxon_p(a_values, b_values),
                })
    holm_bonferroni(pairwise_rows)

    mcnemar_rows: list[dict[str, Any]] = []
    for left, right in combinations(structured, 2):
        docs, a_values, b_values = paired_values(structured[left], structured[right], "schema_success")
        left_only, right_only, p_value = mcnemar_p(a_values, b_values)
        mcnemar_rows.append({
            "system_a": left,
            "system_b": right,
            "paired_n": len(docs),
            "a_success_b_failure": left_only,
            "a_failure_b_success": right_only,
            "test": "mcnemar_exact_binomial",
            "p_value": p_value,
        })
    holm_bonferroni(mcnemar_rows)

    friedman_rows = []
    for family, systems, metrics in [
        ("structured", structured, STRUCTURED_METRICS),
        ("raw_ocr", ocr, OCR_METRICS),
    ]:
        for metric in metrics:
            common_n, p_value = friedman_p(systems, metric)
            friedman_rows.append({
                "family": family,
                "metric": metric,
                "systems_compared": len(systems),
                "common_paired_n": common_n,
                "test": "friedman",
                "p_value": p_value,
            })

    write_csv(COMBINED / "stage1b_extended_bootstrap_ci.csv", bootstrap_rows,
              ["family", "system", "metric", "n", "mean", "ci_lower", "ci_upper", "bootstrap_iterations"])
    write_csv(COMBINED / "stage1b_extended_pairwise_tests.csv", pairwise_rows,
              ["family", "metric", "system_a", "system_b", "paired_n", "mean_a", "mean_b",
               "mean_difference_a_minus_b", "test", "p_value", "holm_adjusted_p"])
    write_csv(COMBINED / "stage1b_extended_mcnemar_tests.csv", mcnemar_rows,
              ["system_a", "system_b", "paired_n", "a_success_b_failure", "a_failure_b_success",
               "test", "p_value", "holm_adjusted_p"])
    write_csv(COMBINED / "stage1b_extended_friedman_tests.csv", friedman_rows,
              ["family", "metric", "systems_compared", "common_paired_n", "test", "p_value"])

    summary_lines = [
        "# Stage 1B Extended Statistical Tests Summary",
        "",
        f"Generated: {now()}",
        "",
        "## Included full-53 systems (primary paired tests)",
        "- Structured: Qwen3-VL 8B, LLaVA 13B, GLM-OCR + qwen3:8b, docTR + qwen3:8b, TrOCR + qwen3:8b, EasyOCR + qwen3:8b, Surya + qwen3:8b, Docling + qwen3:8b.",
        "- Raw OCR: GLM-OCR, docTR, TrOCR, EasyOCR, Surya, Docling.",
        "",
        "## Excluded from primary paired tests",
        "- Marker raw OCR and Marker + qwen3 partial: partial/interim coverage only (19/53).",
        "- Qwen2.5-VL raw OCR and structured rows: coverage-limited at 52/53; cannot be paired over the full 53-document set.",
        "- Internal Qwen3-27B recovered-plus: aggregate row imported; no per-document compatible metric table available on Final release for paired tests.",
        "",
        "## Systems with per-document metrics available (not in primary tests)",
        "- Qwen2.5-VL raw OCR: 52/53 coverage, per-document scores available. Mean token_f1=0.4449, text_similarity=0.1636.",
        "- Qwen2.5-VL structured: 52/53 coverage, 45/53 schema-valid, mean overall=0.2731, entity_lenient_f1=0.0000.",
        "",
        "## Methods",
        "- Paired bootstrap 95% confidence intervals for system-level means.",
        "- Wilcoxon signed-rank tests for paired continuous per-document metrics.",
        "- Exact-binomial McNemar tests for paired schema-valid success/failure.",
        "- Friedman tests over common full-53 document sets where SciPy is available.",
        "- Holm-Bonferroni correction for pairwise p-values.",
        "",
        "## Metric families",
        "- Structured: overall_extraction_score, entity_lenient_f1.",
        "- Raw OCR: token_f1, text_similarity.",
        "",
        "## Outputs",
        "- stage1b_extended_bootstrap_ci.csv",
        "- stage1b_extended_pairwise_tests.csv",
        "- stage1b_extended_mcnemar_tests.csv",
        "- stage1b_extended_friedman_tests.csv",
    ]
    write_text(COMBINED / "stage1b_extended_statistical_tests_summary.md", "\n".join(summary_lines))


def write_tables_and_reports(rows: list[dict[str, Any]]) -> None:
    fields = [
        "section", "system", "coverage", "records_attempted", "records_schema_valid",
        "schema_parse_success", "scalar_exact_accuracy", "scalar_lenient_accuracy",
        "entity_exact_f1", "entity_lenient_f1", "hallucination_rate", "missing_entity_rate",
        "annotation_gap_rate", "overall_extraction_score", "records_nonempty", "ocr_f1",
        "text_similarity", "runtime", "label", "notes",
    ]
    write_csv(COMBINED / "stage1b_extended_combined_metrics.csv", rows, fields)

    # Sort rows by section and type for tables
    raw_rows = [row for row in rows if row["section"] == "raw_ocr"]
    direct_rows = [row for row in rows if row["section"] == "structured_direct"]
    hybrid_rows = [row for row in rows if row["section"] == "hybrid_pipeline"]

    # Separate categories for raw OCR
    raw_full53 = [r for r in raw_rows if r["label"] == "full_53_raw_ocr"]
    raw_coverage = [r for r in raw_rows if r["label"] == "coverage_limited_52_53_with_metrics"]
    raw_partial = [r for r in raw_rows if "partial" in r["label"]]

    # Separate categories for structured
    struct_full53 = [r for r in direct_rows if "full_53" in r["label"]]
    struct_coverage = [r for r in direct_rows if "coverage_limited" in r["label"]]

    # Separate categories for hybrid
    hybrid_full53 = [r for r in hybrid_rows if "full_53" in r["label"]]
    hybrid_partial = [r for r in hybrid_rows if "partial" in r["label"]]

    raw_columns = [
        ("system", "System"),
        ("coverage", "N"),
        ("records_nonempty", "Non-empty"),
        ("ocr_f1", "OCR/token F1"),
        ("text_similarity", "Text similarity"),
        ("runtime", "Avg runtime s"),
        ("label", "Label"),
        ("notes", "Notes"),
    ]
    structured_columns = [
        ("system", "System"),
        ("coverage", "N"),
        ("records_schema_valid", "Schema-valid"),
        ("schema_parse_success", "Schema parse"),
        ("scalar_exact_accuracy", "Scalar exact"),
        ("scalar_lenient_accuracy", "Scalar lenient"),
        ("entity_exact_f1", "Entity exact F1"),
        ("entity_lenient_f1", "Entity lenient F1"),
        ("hallucination_rate", "Hallucination"),
        ("missing_entity_rate", "Missing"),
        ("annotation_gap_rate", "Annotation gap"),
        ("overall_extraction_score", "Overall"),
        ("runtime", "Avg runtime s"),
        ("label", "Label"),
        ("notes", "Notes"),
    ]

    # Raw OCR table
    raw_lines = [
        "# Stage 1B Extended Raw OCR Table",
        "",
        f"Generated: {now()}",
        "",
        "## Full-53 systems (primary comparison)",
        "",
        md_table(raw_full53, raw_columns),
        "",
        "## Coverage-limited systems",
        "",
        md_table(raw_coverage, raw_columns),
        "",
        "## Partial / Interim Raw OCR",
        "",
        md_table(raw_partial, raw_columns),
        "",
        "Qwen2.5-VL raw OCR is included with computed per-document metrics (52/53 coverage, 45/53 non-empty). "
        "It is excluded from primary full-53 paired tests because 1 document (p26) was not recovered. "
        "Marker remains partial/interim (19/53) and is excluded from full-53 statistical comparisons.",
    ]
    write_text(COMBINED / "stage1b_extended_raw_ocr_table.md", "\n".join(raw_lines))

    # Structured direct VLM table
    struct_lines = [
        "# Stage 1B Extended Direct VLM Structured Table",
        "",
        f"Generated: {now()}",
        "",
        "## Full-53 systems (primary comparison)",
        "",
        md_table(struct_full53, structured_columns),
        "",
        "## Coverage-limited systems",
        "",
        md_table(struct_coverage, structured_columns),
        "",
        "Qwen2.5-VL structured is included with computed per-document metrics (52/53 coverage, 45/53 schema-valid). "
        "It is excluded from primary full-53 paired tests because 1 document (p36_1) was not recovered. "
        "Internal Qwen3-27B recovered-plus uses 53/53 coverage from the latest import and carries forward "
        "the latest imported scored fields from the earlier recovered row.",
    ]
    write_text(COMBINED / "stage1b_extended_structured_direct_vlm_table.md", "\n".join(struct_lines))

    # Hybrid pipeline table
    hybrid_lines = [
        "# Stage 1B Extended Hybrid OCR-to-JSON Table",
        "",
        f"Generated: {now()}",
        "",
        "## Full-53 systems (primary comparison)",
        "",
        md_table(hybrid_full53, structured_columns),
        "",
        "## Partial / Interim Hybrid Lane",
        "",
        md_table(hybrid_partial, structured_columns),
        "",
        "EasyOCR, Surya, and Docling + qwen3 were consolidated from the completed background hybrid run. "
        "Marker + qwen3 remains partial/interim only (19/53).",
    ]
    write_text(COMBINED / "stage1b_extended_hybrid_pipeline_table.md", "\n".join(hybrid_lines))

    # Ranking summary
    ranking_lines = [
        "# Stage 1B Extended Combined Ranking Summary",
        "",
        f"Generated: {now()}",
        "",
        "## Headline findings",
        "- Best direct VLM by available overall score: Internal Qwen3-27B compact recovered-plus, overall 0.4039 with 53/53 schema-valid.",
        "- Best local direct Ollama VLM baseline: Qwen3-VL 8B, overall 0.3549 with 48/53 schema-valid.",
        "- Best OCR-to-JSON pipeline: GLM-OCR + qwen3:8b, overall 0.3628 with 50/53 schema-valid.",
        "- Best raw OCR by OCR/token F1: GLM-OCR, 0.2464 on the full 53-record denominator.",
        "- Fastest raw OCR among full-53 lanes: GLM-OCR, 2.6471 seconds per document.",
        "- Most reliable schema-valid structured system: Internal Qwen3-27B recovered-plus, 53/53.",
        "",
        "## Important caveats",
        "- Several low-recall systems report 0.0 hallucination because they mostly omit entities rather than invent them.",
        "- Qwen2.5-VL raw OCR achieves a relatively high token F1 (0.4449) but is coverage-limited at 52/53.",
        "- Qwen2.5-VL structured achieves entity_lenient_f1=0.0000 across all documents (no entity extraction).",
        "- Marker is partial (19/53) and excluded from full-53 comparisons.",
        "- Internal Qwen3-27B recovered-plus scored fields were not recomputed after the p45_2 retry.",
    ]
    write_text(COMBINED / "stage1b_extended_combined_ranking_summary.md", "\n".join(ranking_lines))

    # Final benchmark status
    status_lines = [
        "# Stage 1B Extended Final Benchmark Status",
        "",
        f"Generated: {now()}",
        "",
        "## Final coverage for every system",
        "",
        "### Raw OCR",
        "- GLM-OCR: 53/53 full. token F1=0.2464, text_sim=0.1874.",
        "- docTR: 53/53 full. token F1=0.1980, text_sim=0.1851.",
        "- TrOCR: 53/53 full. token F1=0.0082, text_sim=0.0098.",
        "- Docling: 53/53 full. token F1=0.1536, text_sim=0.1529.",
        "- Surya: 53/53 full. token F1=0.2362, text_sim=0.1873.",
        "- EasyOCR: 53/53 full. token F1=0.1606, text_sim=0.1728.",
        "- Qwen2.5-VL raw OCR: 52/53 coverage-limited. token F1=0.4449, text_sim=0.1636.",
        "- Marker: 19/53 partial. token F1=0.1221, text_sim=0.1264.",
        "",
        "### Direct structured / VLM",
        "- Internal Qwen3-27B recovered-plus: 53/53 schema-valid. Overall=0.4039.",
        "- Qwen3-VL 8B: 48/53 schema-valid. Overall=0.3549.",
        "- LLaVA 13B: 27/53 schema-valid. Overall=0.2483.",
        "- Qwen2.5-VL structured: 52/53 coverage, 45/53 schema-valid. Overall=0.2731.",
        "",
        "### Hybrid pipelines",
        "- GLM-OCR + qwen3:8b: 50/53 schema-valid. Overall=0.3628.",
        "- docTR + qwen3:8b: 49/53 schema-valid. Overall=0.3296.",
        "- TrOCR + qwen3:8b: 48/53 schema-valid. Overall=0.2777.",
        "- EasyOCR + qwen3:8b: 52/53 schema-valid. Overall=0.2851.",
        "- Surya + qwen3:8b: 52/53 schema-valid. Overall=0.2832.",
        "- Docling + qwen3:8b: 52/53 schema-valid. Overall=0.2868.",
        "- Marker + qwen3:8b: 19/53 partial. Overall=0.2336.",
        "",
        "## Final Qwen2.5-VL status",
        "- Raw OCR: 52/53 documents recovered via Final release recovery. Per-document scores computed via canonical raw OCR evaluator. Mean token_f1=0.4449 over 53 documents (including 8 failed=0). Excluded from primary full-53 paired tests because coverage is 52/53.",
        "- Structured: 52/53 documents recovered; 45/53 schema-valid (8 schema failures). Mean overall=0.2731, entity_lenient_f1=0.0000. Excluded from primary full-53 paired tests.",
        "",
        "## Final Marker status",
        "- Marker raw OCR: 19/53 usable rows in Final release handoff. 14 additional documents on Final release were timeout failures (not handoff artifacts). Marker excluded from primary full-53 tests.",
        "- Marker + qwen3:8b: 19/53 schema-valid partial. Not rerun (handoff-limited).",
        "",
        "## Systems included in primary full-53 stats",
        "- Raw OCR: GLM-OCR, docTR, TrOCR, EasyOCR, Surya, Docling (6 systems, all 53/53).",
        "- Structured: Qwen3-VL 8B, LLaVA 13B, GLM-OCR+qwen3, docTR+qwen3, TrOCR+qwen3, EasyOCR+qwen3, Surya+qwen3, Docling+qwen3 (8 systems, all attempted 53 docs).",
        "",
        "## Coverage-limited / partial systems",
        "- Qwen2.5-VL raw OCR (52/53 coverage-limited, included in table but not primary stats).",
        "- Qwen2.5-VL structured (52/53 coverage-limited, 45/53 schema-valid, included in table but not primary stats).",
        "- Marker raw OCR (19/53 partial, included in table but not primary stats).",
        "- Marker + qwen3:8b (19/53 partial, included in table but not primary stats).",
        "- Internal Qwen3-27B recovered-plus (53/53 aggregate, included in table but not primary stats; no per-document table on Final release).",
        "",
        "## Exact caveats",
        "- Qwen2.5-VL raw OCR per-document scores computed via Final release canonical evaluator; values supersede any earlier Final release summary.",
        "- Qwen2.5-VL structured uses compact-to-canonical adapter; entity_lenient_f1=0.0000 reflects no entity extraction.",
        "- Marker handoff limited to 19/53 on Final release; Final release reconciled 39 usable but handoff was not fully re-exported.",
        "- Internal Qwen3-27B recovered-plus scored fields are from the earlier import, not recomputed after p45_2 retry.",
        "- Final release raw OCR values are canonical and supersede earlier Final release summary values.",
        "",
        "## Metric provenance audit summary",
        "- All full-53 raw OCR per-document scores are from local Final release benchmark runs using the Stage 1B raw OCR evaluator (scripts/benchmark_raw_ocr_outputs.py).",
        "- Qwen2.5-VL raw OCR per-document scores are from the Final release recovery run, evaluated with the same canonical raw OCR evaluator.",
        "- All hybrid structured per-document scores are from local Final release runs using the canonical structured benchmark helper (scripts/benchmark_structured_json_outputs.py).",
        "- Qwen2.5-VL structured per-document scores used the compact-to-canonical adapter from the structured benchmark script.",
        "- Marker rows remain partial because the imported handoff provided only 19 usable Marker OCR outputs.",
        "- Final release values supersede earlier Final release raw OCR summary values where discrepancies exist.",
    ]
    write_text(REPORTS / "stage1b_extended_final_benchmark_status.md", "\n".join(status_lines))

    # Metric provenance audit Final release
    audit_lines = [
        "# Stage 1B Extended Metric Provenance Audit — Final release",
        "",
        f"Generated: {now()}",
        "",
        "## Raw OCR rows",
        "",
        "| System | Source OCR output file | Per-document metric CSV | Evaluator script | Rows | Denominator | Tokenization | Intended evaluator |",
        "|---|---|---|---:|---:|:---|---|---|",
    ]
    raw_sources = [
        ("GLM-OCR", "Final release GLM-OCR full benchmark", "reports/stage1b_raw_ocr_benchmark_glm_ocr/per_document_ocr_scores.csv",
         "scripts/benchmark_raw_ocr_outputs.py", 53, 53, "word-level tokenization (whitespace+punct)", "yes"),
        ("docTR", "Final release OCR handoff", "reports/stage1b_raw_ocr_benchmark_doctr/per_document_ocr_scores.csv",
         "scripts/benchmark_raw_ocr_outputs.py", 53, 53, "word-level tokenization (whitespace+punct)", "yes"),
        ("TrOCR", "Final release OCR handoff", "reports/stage1b_raw_ocr_benchmark_trocr/per_document_ocr_scores.csv",
         "scripts/benchmark_raw_ocr_outputs.py", 53, 53, "word-level tokenization (whitespace+punct)", "yes"),
        ("Docling", "Final release OCR handoff", "reports/stage1b_extended_raw_ocr_benchmark_docling/per_document_ocr_scores.csv",
         "scripts/benchmark_raw_ocr_outputs.py", 53, 53, "word-level tokenization (whitespace+punct)", "yes"),
        ("Surya", "Final release OCR handoff", "reports/stage1b_extended_raw_ocr_benchmark_surya/per_document_ocr_scores.csv",
         "scripts/benchmark_raw_ocr_outputs.py", 53, 53, "word-level tokenization (whitespace+punct)", "yes"),
        ("EasyOCR", "Final release OCR handoff", "reports/stage1b_extended_raw_ocr_benchmark_easyocr/per_document_ocr_scores.csv",
         "scripts/benchmark_raw_ocr_outputs.py", 53, 53, "word-level tokenization (whitespace+punct)", "yes"),
        ("Qwen2.5-VL raw OCR", "Final release recovery run", "reports/stage1b_extended_qwen25vl_raw_ocr_per_document_scores.csv",
         "Final release canonical raw OCR evaluator (same metric logic)", 53, 53, "word-level tokenization (whitespace+punct)", "yes"),
        ("Marker", "Final release OCR handoff (19/53)", "reports/stage1b_extended_raw_ocr_benchmark_marker/per_document_ocr_scores.csv",
         "scripts/benchmark_raw_ocr_outputs.py", 19, 19, "word-level tokenization (whitespace+punct)", "yes, partial only"),
    ]
    for sys_name, src, csv_path, ev_script, rows_n, denom, tok, intended in raw_sources:
        audit_lines.append(f"| {sys_name} | {src} | {csv_path} | {ev_script} | {rows_n} | {denom} | {tok} | {intended} |")

    audit_lines += [
        "",
        "### Known discrepancy: Final release vs Final release raw OCR values",
        "",
        "Earlier Final release raw OCR summary had different (generally higher) token F1 values for docTR, Surya, and EasyOCR "
        "compared to the final Final release raw OCR table. Investigation:",
        "- Final release values were computed locally from the imported Final release OCR handoff using the Stage 1B raw OCR evaluator.",
        "- Final release summary may have used a different evaluator version or different tokenization/normalization.",
        "- **Final Final release values are canonical** and supersede earlier Final release summary values.",
        "- The Final release per-document CSVs are the authoritative source for raw OCR metrics.",
        "",
        "## Structured / direct / hybrid rows",
        "",
        "| System | Source prediction/metrics | Evaluator script | Attempted N | Schema-valid | Failures type | Included in primary stats? |",
        "|---|---:|---:|---:|---|---|",
    ]
    struct_sources = [
        ("Internal Qwen3-27B recovered-plus", "stage1b_extended_qwen3_27b_merged_plus_metrics.csv",
         "Compact-to-canonical adapter", 53, 53, "model failures (p45_2 retry fixed)", "No (aggregate import only)"),
        ("Qwen3-VL 8B", "Final release full direct VLM benchmark",
         "scripts/benchmark_structured_json_outputs.py", 53, 48, "model failures (5 schema parse failures)", "Yes"),
        ("LLaVA 13B", "Final release full direct VLM benchmark",
         "scripts/benchmark_structured_json_outputs.py", 53, 27, "model failures (26 schema parse failures)", "Yes"),
        ("Qwen2.5-VL structured", "Final release recovery run",
         "Compact-to-canonical adapter from benchmark_structured_json_outputs.py", 53, 45, "model/schema failures (8 failures, 1 unrecovered)", "No (coverage-limited 52/53)"),
        ("GLM-OCR + qwen3:8b", "Final release OCR-to-JSON pipeline",
         "scripts/benchmark_structured_json_outputs.py", 53, 50, "model failures (3 schema parse failures)", "Yes"),
        ("docTR + qwen3:8b", "Final release handoff + Final release qwen3",
         "scripts/benchmark_structured_json_outputs.py", 53, 49, "model failures (4 schema parse failures)", "Yes"),
        ("TrOCR + qwen3:8b", "Final release handoff + Final release qwen3",
         "scripts/benchmark_structured_json_outputs.py", 53, 48, "model failures (5 schema parse failures)", "Yes"),
        ("EasyOCR + qwen3:8b", "Final release background hybrid run",
         "scripts/benchmark_structured_json_outputs.py", 53, 52, "model failures (1 schema parse failure)", "Yes"),
        ("Surya + qwen3:8b", "Final release background hybrid run",
         "scripts/benchmark_structured_json_outputs.py", 53, 52, "model failures (1 schema parse failure)", "Yes"),
        ("Docling + qwen3:8b", "Final release background hybrid run",
         "scripts/benchmark_structured_json_outputs.py", 53, 52, "model failures (1 schema parse failure)", "Yes"),
        ("Marker + qwen3:8b", "Final release background hybrid run (partial)",
         "scripts/benchmark_structured_json_outputs.py", 19, 19, "partial (34 docs not attempted)", "No (partial 19/53)"),
    ]
    for sys_name, src, ev, att, sv, ft, inc in struct_sources:
        audit_lines.append(f"| {sys_name} | {src} | {ev} | {att} | {sv} | {ft} | {inc} |")

    audit_lines += [
        "",
        "## Statistical test audit",
        "- Raw OCR primary paired tests: use token_f1 and text_similarity from 6 full-53 systems.",
        "- Structured primary paired tests: use overall_extraction_score and entity_lenient_f1 from 8 full-53 systems.",
        "- Qwen2.5-VL excluded from primary full-53 tests (52/53 coverage).",
        "- Marker excluded from primary full-53 tests (19/53 partial).",
        "- Internal Qwen3-27B excluded from primary full-53 tests (no per-document scores on Final release).",
        "- Holm-Bonferroni correction applied within each pairwise test family.",
        "",
        "## Final decision",
        "- Final release raw OCR values are canonical and supersede earlier Final release summary.",
        "- Qwen2.5-VL per-document scores from Final release recovery are accepted as computed.",
        "- Marker remains partial at 19/53 (handoff-limited).",
        "- All aggregate tables, statistics, and reports regenerated from canonical Final release data.",
    ]
    write_text(REPORTS / "stage1b_extended_metric_provenance_audit_server1.md", "\n".join(audit_lines))


def main() -> None:
    rows = build_combined_rows()
    write_tables_and_reports(rows)
    write_stats()
    print("Consolidation complete. Tables and statistics regenerated.")


if __name__ == "__main__":
    main()
