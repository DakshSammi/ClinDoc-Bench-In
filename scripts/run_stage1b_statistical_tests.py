#!/usr/bin/env python3
"""Run paired statistical tests for Stage 1B paper-facing metrics."""

from __future__ import annotations

import argparse
import csv
import math
import random
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scipy import stats
except Exception:  # pragma: no cover - fallback keeps the script usable.
    stats = None


STRUCTURED_METRICS = ["overall_extraction_score", "entity_lenient_f1"]
OCR_METRICS = ["token_f1"]


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
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SYSTEM=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("System name cannot be empty")
    return name.strip(), Path(raw_path.strip())


def as_float(row: dict[str, str], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            try:
                return float(value)
            except ValueError:
                continue
    return default


def schema_success(row: dict[str, str]) -> int:
    for key in ("schema_parse_success", "schema_validity", "json_parse_success"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"1", "1.0", "true", "yes", "success"}:
            return 1
    status = str(row.get("status", "")).strip().lower()
    return 1 if status == "success" else 0


def structured_score(row: dict[str, str]) -> float:
    explicit = as_float(row, "overall_extraction_score", default=math.nan)
    if not math.isnan(explicit):
        return explicit
    schema = schema_success(row)
    scalar_lenient = as_float(row, "scalar_lenient_accuracy")
    entity_lenient = as_float(row, "entity_lenient_f1")
    entity_exact = as_float(row, "entity_exact_f1")
    hallucination = as_float(row, "hallucination_rate")
    return round(
        0.10 * schema
        + 0.20 * scalar_lenient
        + 0.45 * entity_lenient
        + 0.15 * entity_exact
        + 0.10 * (1.0 - hallucination),
        6,
    )


def normalize_structured(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        doc_id = row.get("document_id", "").strip()
        if not doc_id:
            continue
        out[doc_id] = {
            "schema_success": float(schema_success(row)),
            "overall_extraction_score": structured_score(row),
            "entity_lenient_f1": as_float(row, "entity_lenient_f1"),
        }
    return out


def normalize_ocr(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        doc_id = row.get("document_id", "").strip()
        if not doc_id:
            continue
        out[doc_id] = {
            "token_f1": as_float(row, "token_f1", "ocr_f1"),
            "text_similarity": as_float(row, "text_similarity", "normalized_edit_similarity"),
        }
    return out


def paired_values(
    a: dict[str, dict[str, float]],
    b: dict[str, dict[str, float]],
    metric: str,
) -> tuple[list[str], list[float], list[float]]:
    docs = sorted(set(a) & set(b))
    return docs, [a[doc][metric] for doc in docs], [b[doc][metric] for doc in docs]


def bootstrap_ci(values: list[float], rng: random.Random, iterations: int) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return mean(values), lower, upper


def wilcoxon_p(a_values: list[float], b_values: list[float]) -> float | str:
    if len(a_values) < 2:
        return ""
    diffs = [a - b for a, b in zip(a_values, b_values) if a != b]
    if not diffs:
        return 1.0
    if stats is None:
        return ""
    try:
        return float(stats.wilcoxon(a_values, b_values, zero_method="wilcox").pvalue)
    except Exception:
        return ""


def mcnemar_p(a_success: list[float], b_success: list[float]) -> tuple[int, int, float | str]:
    b_only = sum(1 for a, b in zip(a_success, b_success) if a == 0 and b == 1)
    a_only = sum(1 for a, b in zip(a_success, b_success) if a == 1 and b == 0)
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    if stats is None:
        return a_only, b_only, ""
    return a_only, b_only, float(stats.binomtest(min(a_only, b_only), discordant, 0.5).pvalue)


def friedman_p(systems: dict[str, dict[str, dict[str, float]]], metric: str) -> tuple[int, float | str]:
    common_docs = sorted(set.intersection(*(set(rows) for rows in systems.values()))) if systems else []
    if len(common_docs) < 2 or len(systems) < 3 or stats is None:
        return len(common_docs), ""
    arrays = [[systems[name][doc][metric] for doc in common_docs] for name in systems]
    try:
        return len(common_docs), float(stats.friedmanchisquare(*arrays).pvalue)
    except Exception:
        return len(common_docs), ""


def holm_bonferroni(rows: list[dict[str, Any]], p_key: str = "p_value") -> None:
    indexed = []
    for idx, row in enumerate(rows):
        try:
            p_value = float(row.get(p_key, ""))
        except (TypeError, ValueError):
            continue
        indexed.append((p_value, idx))
    indexed.sort()
    m = len(indexed)
    adjusted = [None] * len(rows)
    running = 0.0
    for rank, (p_value, idx) in enumerate(indexed, start=1):
        value = min(1.0, (m - rank + 1) * p_value)
        running = max(running, value)
        adjusted[idx] = running
    for idx, value in enumerate(adjusted):
        rows[idx]["holm_adjusted_p"] = "" if value is None else round(value, 6)


def load_named_tables(named_paths: list[tuple[str, Path]], family: str) -> dict[str, dict[str, dict[str, float]]]:
    loaded = {}
    for name, path in named_paths:
        if not path.exists():
            raise FileNotFoundError(f"{family} table not found for {name}: {path}")
        rows = read_csv(path)
        loaded[name] = normalize_structured(rows) if family == "structured" else normalize_ocr(rows)
    return loaded


def bootstrap_rows(
    systems: dict[str, dict[str, dict[str, float]]],
    metrics: list[str],
    rng: random.Random,
    iterations: int,
    family: str,
) -> list[dict[str, Any]]:
    rows = []
    for system, docs in systems.items():
        for metric in metrics:
            values = [item[metric] for item in docs.values() if metric in item]
            estimate, lower, upper = bootstrap_ci(values, rng, iterations)
            rows.append(
                {
                    "family": family,
                    "system": system,
                    "metric": metric,
                    "n": len(values),
                    "mean": round(estimate, 6) if not math.isnan(estimate) else "",
                    "ci_lower": round(lower, 6) if not math.isnan(lower) else "",
                    "ci_upper": round(upper, 6) if not math.isnan(upper) else "",
                    "bootstrap_iterations": iterations,
                }
            )
    return rows


def pairwise_rows(
    systems: dict[str, dict[str, dict[str, float]]],
    metrics: list[str],
    family: str,
) -> list[dict[str, Any]]:
    rows = []
    for metric in metrics:
        for left, right in combinations(systems, 2):
            docs, a_values, b_values = paired_values(systems[left], systems[right], metric)
            if not docs:
                continue
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "system_a": left,
                    "system_b": right,
                    "paired_n": len(docs),
                    "mean_a": round(mean(a_values), 6),
                    "mean_b": round(mean(b_values), 6),
                    "mean_difference_a_minus_b": round(mean([a - b for a, b in zip(a_values, b_values)]), 6),
                    "test": "wilcoxon_signed_rank",
                    "p_value": wilcoxon_p(a_values, b_values),
                }
            )
    holm_bonferroni(rows)
    return rows


def mcnemar_rows(systems: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(systems, 2):
        docs, a_values, b_values = paired_values(systems[left], systems[right], "schema_success")
        if not docs:
            continue
        a_only, b_only, p_value = mcnemar_p(a_values, b_values)
        rows.append(
            {
                "system_a": left,
                "system_b": right,
                "paired_n": len(docs),
                "a_success_b_failure": a_only,
                "a_failure_b_success": b_only,
                "test": "mcnemar_exact_binomial",
                "p_value": p_value,
            }
        )
    holm_bonferroni(rows)
    return rows


def summary_md(
    structured: dict[str, dict[str, dict[str, float]]],
    ocr: dict[str, dict[str, dict[str, float]]],
    bootstrap: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    mcnemar: list[dict[str, Any]],
) -> str:
    lines = ["# Stage 1B Statistical Tests Summary", ""]
    lines.append("## Inputs")
    for name, docs in structured.items():
        lines.append(f"- Structured: {name} ({len(docs)} documents)")
    for name, docs in ocr.items():
        lines.append(f"- Raw OCR: {name} ({len(docs)} documents)")
    lines.append("")
    lines.append("## Methods")
    lines.append("- Paired bootstrap 95% confidence intervals for system-level means.")
    lines.append("- Wilcoxon signed-rank tests for paired continuous per-document metrics.")
    lines.append("- Exact-binomial McNemar tests for paired schema-valid success/failure.")
    lines.append("- Friedman tests for multiple systems over common documents where SciPy is available.")
    lines.append("- Holm-Bonferroni correction for pairwise p-values.")
    lines.append("")
    lines.append("## Friedman Tests")
    if structured:
        n, p_value = friedman_p(structured, "overall_extraction_score")
        lines.append(f"- Structured overall extraction score: common N={n}, p={p_value}")
    if ocr:
        n, p_value = friedman_p(ocr, "token_f1")
        lines.append(f"- Raw OCR token F1: common N={n}, p={p_value}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("- `stage1b_bootstrap_ci.csv`")
    lines.append("- `stage1b_pairwise_tests.csv`")
    lines.append("- `stage1b_mcnemar_tests.csv`")
    lines.append("")
    lines.append(f"Bootstrap rows: {len(bootstrap)}")
    lines.append(f"Pairwise rows: {len(pairwise)}")
    lines.append(f"McNemar rows: {len(mcnemar)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured", action="append", type=parse_named_path, default=[], help="Structured SYSTEM=CSV")
    parser.add_argument("--ocr", action="append", type=parse_named_path, default=[], help="Raw OCR SYSTEM=CSV")
    parser.add_argument("--output-dir", default="paper_assets/tables/combined")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)
    structured = load_named_tables(args.structured, "structured")
    ocr = load_named_tables(args.ocr, "ocr")

    bootstrap = []
    bootstrap.extend(bootstrap_rows(structured, STRUCTURED_METRICS, rng, args.bootstrap_iters, "structured"))
    bootstrap.extend(bootstrap_rows(ocr, OCR_METRICS, rng, args.bootstrap_iters, "raw_ocr"))
    pairwise = []
    pairwise.extend(pairwise_rows(structured, STRUCTURED_METRICS, "structured"))
    pairwise.extend(pairwise_rows(ocr, OCR_METRICS, "raw_ocr"))
    mcnemar = mcnemar_rows(structured)

    write_csv(
        output_dir / "stage1b_bootstrap_ci.csv",
        bootstrap,
        ["family", "system", "metric", "n", "mean", "ci_lower", "ci_upper", "bootstrap_iterations"],
    )
    write_csv(
        output_dir / "stage1b_pairwise_tests.csv",
        pairwise,
        [
            "family",
            "metric",
            "system_a",
            "system_b",
            "paired_n",
            "mean_a",
            "mean_b",
            "mean_difference_a_minus_b",
            "test",
            "p_value",
            "holm_adjusted_p",
        ],
    )
    write_csv(
        output_dir / "stage1b_mcnemar_tests.csv",
        mcnemar,
        [
            "system_a",
            "system_b",
            "paired_n",
            "a_success_b_failure",
            "a_failure_b_success",
            "test",
            "p_value",
            "holm_adjusted_p",
        ],
    )
    write_text(output_dir / "stage1b_statistical_tests_summary.md", summary_md(structured, ocr, bootstrap, pairwise, mcnemar))


if __name__ == "__main__":
    main()
