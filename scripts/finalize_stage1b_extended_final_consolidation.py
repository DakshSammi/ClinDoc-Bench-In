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

"""Build the final Stage 1B Extended paper package and statistics bundle."""

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
except Exception:  # pragma: no cover
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


def parse_benchmark_summary(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for key, pattern in {
        "successful": r"- Successful:\s+(\d+)",
        "failed": r"- Failed:\s+(\d+)",
        "runtime": r"- Avg runtime per doc:\s+([0-9.]+)s",
        "total": r"- Total docs:\s+(\d+)",
    }.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else ""
    return out


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


def raw_row_from_ppt(
    ppt_rows: list[dict[str, str]],
    system: str,
    label: str,
    notes: str,
) -> dict[str, Any]:
    row = find_row(ppt_rows, system)
    attempted = row.get("records_attempted", "")
    nonempty = row.get("records_nonempty", "")
    return {
        "section": "raw_ocr",
        "system": system,
        "coverage": f"{attempted}/53" if attempted else "",
        "records_attempted": attempted,
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
        "ocr_f1": fmt(row.get("ocr_f1")),
        "text_similarity": fmt(row.get("text_similarity")),
        "runtime": fmt(row.get("runtime")),
        "label": label,
        "notes": notes,
    }


def raw_qwen25_row() -> dict[str, Any]:
    summary = parse_benchmark_summary(REPORTS / "stage1b_extended_qwen25vl_raw_full_summary.md")
    successful = int(summary["successful"])
    total = int(summary["total"])
    return {
        "section": "raw_ocr",
        "system": "Qwen2.5-VL raw OCR",
        "coverage": f"{successful}/{total}",
        "records_attempted": total,
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
        "records_nonempty": successful,
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": fmt(summary["runtime"], digits=1),
        "label": "coverage_only_imported_status",
        "notes": "5 no_images cases are data-availability gaps, not model failures; OCR quality metrics were not included in the imported Final release package.",
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
        "notes": "Coverage and schema-validity are from stage1b_extended_qwen3_27b_merged_plus_metrics.csv (53/53 after p45_2 retry). Scored extraction fields remain the latest imported recovered values from paper_assets/tables/final/stage1b_server2_ppt_aligned_metrics.csv because the recovered-plus import did not include a fresh CanonicalRawDoc recomputation.",
    }


def qwen25_structured_row() -> dict[str, Any]:
    summary = parse_benchmark_summary(REPORTS / "stage1b_extended_qwen25vl_structured_full_summary.md")
    successful = int(summary["successful"])
    total = int(summary["total"])
    return {
        "section": "structured_direct",
        "system": "Qwen2.5-VL structured",
        "coverage": f"{successful}/{total}",
        "records_attempted": total,
        "records_schema_valid": successful,
        "schema_parse_success": fmt(successful / total),
        "scalar_exact_accuracy": "",
        "scalar_lenient_accuracy": "",
        "entity_exact_f1": "",
        "entity_lenient_f1": "",
        "hallucination_rate": "",
        "missing_entity_rate": "",
        "annotation_gap_rate": "",
        "overall_extraction_score": "",
        "records_nonempty": "",
        "ocr_f1": "",
        "text_similarity": "",
        "runtime": fmt(summary["runtime"], digits=1),
        "label": "coverage_only_imported_status",
        "notes": "6 no_images cases are data-availability gaps, not model failures; the imported Final release package included success/runtime status but not a per-document canonical benchmark export for scalar/entity metrics.",
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
    ppt_rows = read_csv(REPORTS / "stage1b_server1_ppt_aligned_metrics.csv")
    rows = [
        raw_row_from_summary(
            "GLM-OCR",
            REPORTS / "stage1b_raw_ocr_benchmark_glm_ocr" / "summary_metrics.json",
            "full_53_best_server1_raw_ocr",
            "Accepted Final release full benchmark raw OCR baseline.",
        ),
        raw_row_from_summary(
            "docTR",
            REPORTS / "stage1b_raw_ocr_benchmark_doctr" / "summary_metrics.json",
            "full_53_server2_handoff_rebenchmarked",
            "Final release OCR handoff re-benchmarked locally with the Stage 1B raw OCR evaluator.",
        ),
        raw_row_from_summary(
            "TrOCR",
            REPORTS / "stage1b_raw_ocr_benchmark_trocr" / "summary_metrics.json",
            "full_53_server2_handoff_rebenchmarked",
            "Final release OCR handoff re-benchmarked locally with the Stage 1B raw OCR evaluator.",
        ),
        raw_row_from_summary(
            "Docling",
            REPORTS / "stage1b_extended_raw_ocr_benchmark_docling" / "summary_metrics.json",
            "full_53_server2_handoff_rebenchmarked",
            "Extended final pass using the imported Final release OCR handoff.",
        ),
        raw_row_from_summary(
            "Surya",
            REPORTS / "stage1b_extended_raw_ocr_benchmark_surya" / "summary_metrics.json",
            "full_53_server2_handoff_rebenchmarked",
            "Extended final pass using the imported Final release OCR handoff.",
        ),
        raw_row_from_summary(
            "EasyOCR",
            REPORTS / "stage1b_extended_raw_ocr_benchmark_easyocr" / "summary_metrics.json",
            "full_53_server2_handoff_rebenchmarked",
            "Extended final pass using the imported Final release OCR handoff.",
        ),
        raw_qwen25_row(),
        raw_row_from_summary(
            "Marker",
            REPORTS / "stage1b_extended_raw_ocr_benchmark_marker" / "summary_metrics.json",
            "partial_interim_only",
            "Only 19 usable Final release marker rows were imported; keep separate from full-53 comparisons.",
            coverage="19/53",
        ),
        internal_qwen3_row(),
        structured_row_from_ppt(
            ppt_rows,
            "qwen3-vl:8b-instruct",
            "full_53_local_direct_vlm",
            "Accepted Final release local direct structured baseline.",
        ),
        structured_row_from_ppt(
            ppt_rows,
            "llava:13b",
            "full_53_diagnostic_direct_vlm",
            "Accepted diagnostic direct structured baseline.",
        ),
        qwen25_structured_row(),
        pipeline_row_from_summary(
            REPORTS / "stage1b_server1_glm_ocr_qwen3_structured_summary.json",
            "full_53_best_hybrid_pipeline",
            "Accepted Final release OCR-to-JSON pipeline.",
        ),
        pipeline_row_from_summary(
            REPORTS / "stage1b_server1_doctr_qwen3_structured_summary.json",
            "full_53_hybrid_pipeline",
            "Imported Final release OCR handoff evaluated locally with the canonical structured benchmark helper.",
        ),
        pipeline_row_from_summary(
            REPORTS / "stage1b_server1_trocr_qwen3_structured_summary.json",
            "full_53_hybrid_pipeline_low_quality",
            "Imported Final release OCR handoff evaluated locally with the canonical structured benchmark helper.",
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
        pipeline_row_from_summary(
            REPORTS / "stage1b_extended_marker_qwen3_partial_summary.json",
            "partial_interim_only",
            "Only a partial/import-limited lane was available; keep separate from full-53 comparisons.",
            coverage="19/53",
        ),
    ]
    return rows


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
                bootstrap_rows.append(
                    {
                        "family": family,
                        "system": system,
                        "metric": metric,
                        "n": len(values),
                        "mean": fmt(estimate, 6),
                        "ci_lower": fmt(lower, 6),
                        "ci_upper": fmt(upper, 6),
                        "bootstrap_iterations": iterations,
                    }
                )

    pairwise_rows: list[dict[str, Any]] = []
    for family, systems, metrics in [
        ("structured", structured, STRUCTURED_METRICS),
        ("raw_ocr", ocr, OCR_METRICS),
    ]:
        for metric in metrics:
            for left, right in combinations(systems, 2):
                docs, a_values, b_values = paired_values(systems[left], systems[right], metric)
                pairwise_rows.append(
                    {
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
                    }
                )
    holm_bonferroni(pairwise_rows)

    mcnemar_rows: list[dict[str, Any]] = []
    for left, right in combinations(structured, 2):
        docs, a_values, b_values = paired_values(structured[left], structured[right], "schema_success")
        left_only, right_only, p_value = mcnemar_p(a_values, b_values)
        mcnemar_rows.append(
            {
                "system_a": left,
                "system_b": right,
                "paired_n": len(docs),
                "a_success_b_failure": left_only,
                "a_failure_b_success": right_only,
                "test": "mcnemar_exact_binomial",
                "p_value": p_value,
            }
        )
    holm_bonferroni(mcnemar_rows)

    friedman_rows = []
    for family, systems, metrics in [
        ("structured", structured, STRUCTURED_METRICS),
        ("raw_ocr", ocr, OCR_METRICS),
    ]:
        for metric in metrics:
            common_n, p_value = friedman_p(systems, metric)
            friedman_rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "systems_compared": len(systems),
                    "common_paired_n": common_n,
                    "test": "friedman",
                    "p_value": p_value,
                }
            )

    write_csv(
        COMBINED / "stage1b_extended_bootstrap_ci.csv",
        bootstrap_rows,
        ["family", "system", "metric", "n", "mean", "ci_lower", "ci_upper", "bootstrap_iterations"],
    )
    write_csv(
        COMBINED / "stage1b_extended_pairwise_tests.csv",
        pairwise_rows,
        ["family", "metric", "system_a", "system_b", "paired_n", "mean_a", "mean_b", "mean_difference_a_minus_b", "test", "p_value", "holm_adjusted_p"],
    )
    write_csv(
        COMBINED / "stage1b_extended_mcnemar_tests.csv",
        mcnemar_rows,
        ["system_a", "system_b", "paired_n", "a_success_b_failure", "a_failure_b_success", "test", "p_value", "holm_adjusted_p"],
    )
    write_csv(
        COMBINED / "stage1b_extended_friedman_tests.csv",
        friedman_rows,
        ["family", "metric", "systems_compared", "common_paired_n", "test", "p_value"],
    )

    summary_lines = [
        "# Stage 1B Extended Statistical Tests Summary",
        "",
        f"Generated: {now()}",
        "",
        "## Included full-53 systems",
        "- Structured: Qwen3-VL 8B, LLaVA 13B, GLM-OCR + qwen3:8b, docTR + qwen3:8b, TrOCR + qwen3:8b, EasyOCR + qwen3:8b, Surya + qwen3:8b, Docling + qwen3:8b.",
        "- Raw OCR: GLM-OCR, docTR, TrOCR, EasyOCR, Surya, Docling.",
        "",
        "## Excluded from primary paired tests",
        "- Marker raw OCR and Marker + qwen3 partial: partial/interim coverage only.",
        "- Qwen2.5-VL raw OCR and structured rows: imported package provides coverage/runtime status but not per-document benchmark scores needed for paired testing.",
        "- Internal Qwen3-27B recovered-plus: aggregate row imported, but no per-document compatible metric table is available on Final release for paired tests.",
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
        "section",
        "system",
        "coverage",
        "records_attempted",
        "records_schema_valid",
        "schema_parse_success",
        "scalar_exact_accuracy",
        "scalar_lenient_accuracy",
        "entity_exact_f1",
        "entity_lenient_f1",
        "hallucination_rate",
        "missing_entity_rate",
        "annotation_gap_rate",
        "overall_extraction_score",
        "records_nonempty",
        "ocr_f1",
        "text_similarity",
        "runtime",
        "label",
        "notes",
    ]
    write_csv(COMBINED / "stage1b_extended_combined_metrics.csv", rows, fields)

    raw_rows = [row for row in rows if row["section"] == "raw_ocr"]
    direct_rows = [row for row in rows if row["section"] == "structured_direct"]
    hybrid_rows = [row for row in rows if row["section"] == "hybrid_pipeline"]

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

    write_text(
        COMBINED / "stage1b_extended_raw_ocr_table.md",
        "\n".join(
            [
                "# Stage 1B Extended Raw OCR Table",
                "",
                f"Generated: {now()}",
                "",
                md_table(raw_rows[:-1], raw_columns),
                "",
                "## Partial / Interim Raw OCR",
                "",
                md_table([raw_rows[-1]], raw_columns),
                "",
                "Qwen2.5-VL raw OCR is included as a coverage/runtime row only because the imported Final release package did not include per-document OCR benchmark scores. Marker remains partial/interim and is excluded from full-53 statistical comparisons.",
            ]
        ),
    )

    write_text(
        COMBINED / "stage1b_extended_structured_direct_vlm_table.md",
        "\n".join(
            [
                "# Stage 1B Extended Direct VLM Structured Table",
                "",
                f"Generated: {now()}",
                "",
                md_table(direct_rows, structured_columns),
                "",
                "Qwen2.5-VL structured is included with coverage and runtime because the imported package did not include the per-document canonical score export needed to recompute scalar/entity metrics on Final release. Internal Qwen3-27B recovered-plus uses 53/53 coverage from the latest import and carries forward the latest imported scored fields from the earlier recovered row.",
            ]
        ),
    )

    write_text(
        COMBINED / "stage1b_extended_hybrid_pipeline_table.md",
        "\n".join(
            [
                "# Stage 1B Extended Hybrid OCR-to-JSON Table",
                "",
                f"Generated: {now()}",
                "",
                md_table(hybrid_rows[:-1], structured_columns),
                "",
                "## Partial / Interim Hybrid Lane",
                "",
                md_table([hybrid_rows[-1]], structured_columns),
                "",
                "EasyOCR, Surya, and Docling + qwen3 were consolidated from the completed background hybrid run. Marker + qwen3 remains partial/interim only.",
            ]
        ),
    )

    ranking_lines = [
        "# Stage 1B Extended Combined Ranking Summary",
        "",
        f"Generated: {now()}",
        "",
        "## Headline findings",
        "- Best direct VLM by available overall score: Internal Qwen3-27B compact recovered-plus coverage row, carrying forward the latest imported recovered score of 0.4039 while the recovered-plus import upgrades schema-valid coverage to 53/53.",
        "- Best local direct Ollama VLM baseline: Qwen3-VL 8B, overall 0.3549 with 48/53 schema-valid.",
        "- Best OCR-to-JSON pipeline: GLM-OCR + qwen3:8b, overall 0.3628 with 50/53 schema-valid.",
        "- Best raw OCR by OCR/token F1: GLM-OCR, 0.2464 on the full 53-record denominator.",
        "- Fastest raw OCR among full-53 lanes: GLM-OCR, 2.6471 seconds per document.",
        "- Most reliable schema-valid structured system by available coverage row: Internal Qwen3-27B recovered-plus, 53/53. Among fully local scored lanes, EasyOCR/Surya/Docling + qwen3 each reached 52/53, but with near-zero entity F1 and very high missing-entity rates.",
        "",
        "## Important caveats",
        "- Several low-recall systems report 0.0 hallucination because they mostly omit entities rather than invent them.",
        "- The worst missing-entity rates are Marker + qwen3 partial (1.0000), EasyOCR + qwen3 (0.9993), Surya + qwen3 (0.9993), Docling + qwen3 (0.9993), and TrOCR + qwen3 (0.9992).",
        "- Qwen2.5-VL no_images failures are treated as data-availability gaps, not model-performance failures.",
        "- Qwen2.5-VL is shown in paper-facing tables with coverage, but it is excluded from primary full-53 paired tests because the imported Final release package lacks per-document benchmark scores on Final release.",
        "- Internal Qwen3-27B recovered-plus scored fields were not recomputed after the p45_2 retry in the imported package; that row is explicitly labelled as a combined-source consolidation.",
    ]
    write_text(COMBINED / "stage1b_extended_combined_ranking_summary.md", "\n".join(ranking_lines))

    imported_files = [
        "stage1b_extended_server2_transfer_manifest.md",
        "stage1b_extended_qwen25vl_structured_metrics.csv",
        "stage1b_extended_qwen25vl_raw_ocr_metrics.csv",
        "stage1b_extended_server2_final_status.md",
        "stage1b_extended_qwen25vl_smoke_summary.md",
        "stage1b_extended_qwen25vl_structured_full_summary.md",
        "stage1b_extended_qwen25vl_raw_full_summary.md",
        "stage1b_extended_marker_completion_diagnosis.md",
        "stage1b_extended_server2_statistical_test_inputs_ready.md",
        "stage1b_extended_server2_structured_benchmark_summary.md",
        "stage1b_extended_qwen3_27b_merged_plus_metrics.csv",
    ]
    imported_status = [f"- {'present' if (REPORTS / name).exists() else 'missing'}: {name}" for name in imported_files]

    status_lines = [
        "# Stage 1B Extended Final Benchmark Status",
        "",
        f"Generated: {now()}",
        "",
        "## Final release background hybrid run",
        "- PID 18576 is no longer running.",
        "- extended_full_run.log shows EasyOCR Full, Surya Full, Docling Full, and Marker Partial lanes were launched.",
        "- Consolidated completed outputs:",
        "  - EasyOCR + qwen3:8b: 52/53 schema-valid, overall 0.2851.",
        "  - Surya + qwen3:8b: 52/53 schema-valid, overall 0.2832.",
        "  - Docling + qwen3:8b: 52/53 schema-valid, overall 0.2868.",
        "  - Marker + qwen3 partial: 19/53 schema-valid, overall 0.2336.",
        "",
        "## Completed systems for paper-facing tables",
        "- Raw OCR full-53: GLM-OCR, docTR, TrOCR, Docling, Surya, EasyOCR.",
        "- Direct structured: Internal Qwen3-27B recovered-plus coverage row, Qwen3-VL 8B, LLaVA 13B, Qwen2.5-VL coverage row.",
        "- Hybrid OCR-to-JSON full-53: GLM-OCR + qwen3:8b, docTR + qwen3:8b, TrOCR + qwen3:8b, EasyOCR + qwen3:8b, Surya + qwen3:8b, Docling + qwen3:8b.",
        "",
        "## Partial / interim systems",
        "- Marker raw OCR: 19/53 usable imported rows only.",
        "- Marker + qwen3 partial: 19/53 schema-valid over the imported partial lane.",
        "",
        "## Excluded / blocked systems",
        "- Qwen2.5 OCR-to-JSON wrong-schema lane: excluded from structured success.",
        "- Tesseract, PaddleOCR, Firenze, Moondream: blocked or not included as final systems.",
        "- Qwen3 full-capability smoke: 2/5 valid, excluded from final paper tables.",
        "- Paid API systems remain excluded from completed benchmark results.",
        "",
        "## Qwen2.5-VL caveat",
        "- Structured coverage: 47/53 successful; 6 no_images cases treated as data-availability gaps.",
        "- Raw OCR coverage: 48/53 successful; 5 no_images cases treated as data-availability gaps.",
        "- Imported Final release package provides coverage/runtime status but not the per-document benchmark exports required for primary paired statistics on Final release.",
        "",
        "## Internal Qwen3 recovered-plus caveat",
        "- stage1b_extended_qwen3_27b_merged_plus_metrics.csv upgrades schema-valid coverage to 53/53 after the p45_2 retry.",
        "- The imported recovered-plus package does not include a fresh CanonicalRawDoc recomputation, so overall/scalar/entity fields in the final table are carried forward from the latest imported recovered row (52/53) and labelled accordingly.",
        "",
        "## Imported Final release verification",
        *imported_status,
        "",
        "## Included in primary full-53 statistical tests",
        "- Structured: Qwen3-VL 8B, LLaVA 13B, GLM-OCR + qwen3:8b, docTR + qwen3:8b, TrOCR + qwen3:8b, EasyOCR + qwen3:8b, Surya + qwen3:8b, Docling + qwen3:8b.",
        "- Raw OCR: GLM-OCR, docTR, TrOCR, EasyOCR, Surya, Docling.",
        "",
        "## Excluded from primary full-53 statistical tests",
        "- Internal Qwen3 recovered-plus: aggregate import only, no per-document compatible score table on Final release.",
        "- Qwen2.5-VL raw and structured rows: coverage/runtime imported, but no per-document score tables available locally.",
        "- Marker raw and Marker + qwen3: partial/interim only.",
    ]
    write_text(REPORTS / "stage1b_extended_final_benchmark_status.md", "\n".join(status_lines))


def main() -> None:
    rows = build_combined_rows()
    write_tables_and_reports(rows)
    write_stats()


if __name__ == "__main__":
    main()
