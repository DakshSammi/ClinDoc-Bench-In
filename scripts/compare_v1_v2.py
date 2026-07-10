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

import csv
import json
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.resolve()
    v1_dir = project_root / "reports" / "raw_benchmark_qwen25_vl_7b_v1_subset"
    v2_dir = project_root / "reports" / "raw_benchmark_qwen25_vl_7b_v2_subset"
    out_csv = project_root / "reports" / "prompt_v1_vs_v2_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load per document scores
    v1_docs = {}
    v1_docs_csv = v1_dir / "per_document_scores.csv"
    with open(v1_docs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v1_docs[row["document_id"]] = row

    v2_docs = {}
    v2_docs_csv = v2_dir / "per_document_scores.csv"
    with open(v2_docs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v2_docs[row["document_id"]] = row

    # 2. Load summary metrics
    with open(v1_dir / "summary_metrics.json", "r", encoding="utf-8") as f:
        v1_summary = json.load(f)
    with open(v2_dir / "summary_metrics.json", "r", encoding="utf-8") as f:
        v2_summary = json.load(f)

    metrics_list = [
        "scalar_accuracy_exact",
        "scalar_accuracy_lenient",
        "entity_exact_f1_macro",
        "entity_lenient_f1_macro",
        "hallucination_rate",
        "missing_entity_rate",
        "annotation_gap_rate",
        "experimental_overall_score"
    ]

    comparison_rows = []

    # Process individual documents
    for doc_id, v2_row in v2_docs.items():
        v1_row = v1_docs.get(doc_id, {})
        doc_type = v2_row["document_type"]
        for metric in metrics_list:
            v1_val = float(v1_row.get(metric, 0.0))
            v2_val = float(v2_row.get(metric, 0.0))
            delta = v2_val - v1_val
            comparison_rows.append({
                "document_id": doc_id,
                "document_type": doc_type,
                "metric": metric,
                "v1_score": f"{v1_val:.4f}",
                "v2_score": f"{v2_val:.4f}",
                "delta": f"{delta:+.4f}"
            })

    # Process global dataset summary
    for metric in metrics_list:
        v1_val = float(v1_summary.get(metric, 0.0))
        v2_val = float(v2_summary.get(metric, 0.0))
        delta = v2_val - v1_val
        comparison_rows.append({
            "document_id": "dataset_summary",
            "document_type": "dataset_summary",
            "metric": metric,
            "v1_score": f"{v1_val:.4f}",
            "v2_score": f"{v2_val:.4f}",
            "delta": f"{delta:+.4f}"
        })

    # Write output CSV
    headers = ["document_id", "document_type", "metric", "v1_score", "v2_score", "delta"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"[OK] Saved comparison report to: {out_csv}")
    print("\nPROMPT V1 VS PROMPT V2 SUBSET DATASET SUMMARY COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<30} | {'V1 (Baseline)':<15} | {'V2 (Prompt v2)':<15} | {'Delta':<10}")
    print("-" * 80)
    for metric in metrics_list:
        v1_val = float(v1_summary.get(metric, 0.0))
        v2_val = float(v2_summary.get(metric, 0.0))
        delta = v2_val - v1_val
        print(f"{metric:<30} | {v1_val*100:13.2f}% | {v2_val*100:13.2f}% | {delta*100:+9.2f}%")
    print("-" * 80)

if __name__ == "__main__":
    main()
