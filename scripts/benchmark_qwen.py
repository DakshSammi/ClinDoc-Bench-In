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

import json
import os
import csv
import time
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt

def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def compare_extractions(pred: Dict[str, Any], gt: Dict[str, Any]):
    # Patient Info Match
    pred_info = pred.get("patient_info", {})
    gt_info = gt.get("patient_info", {})
    
    pred_name = str(pred_info.get("name", "") or "").lower()
    gt_name = str(gt_info.get("name", "") or "").lower()
    name_match = (pred_name == gt_name) if gt_name and gt_name != "none" else False
    
    # Item Count
    pred_items = pred.get("items", [])
    gt_items = gt.get("items", [])
    
    # JSON Validity Check (Was it parsed correctly?)
    is_valid = "raw_json_string" in pred
    
    return {
        "pred_name": pred_info.get("name"),
        "gt_name": gt_info.get("name"),
        "name_match": name_match,
        "pred_item_count": len(pred_items),
        "gt_item_count": len(gt_items),
        "item_count_diff": abs(len(pred_items) - len(gt_items)),
        "is_valid": is_valid
    }

def main():
    start_time = time.time()
    project_root = Path(__file__).parent.parent
    qwen_dir = project_root / "outputs" / "raw_extractions" / "Qwen_Final"
    gt_dir = project_root / "raw_ground_truths"
    bench_dir = project_root / "outputs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Walk through Qwen extractions
    all_files = list(qwen_dir.rglob("*.json"))
    for pred_path in all_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            parent_name = pred_path.parent.name
            possible_gt = gt_dir / pred_path.parent.parent.name / f"{parent_name}.json"
            if possible_gt.exists():
                gt_path = possible_gt
            else:
                possible_gt = gt_dir / parent_name / f"{parent_name}.json"
                if possible_gt.exists():
                    gt_path = possible_gt

        if not gt_path or not gt_path.exists():
            continue
            
        pred_data = load_json(pred_path)
        gt_data = load_json(gt_path)
        
        comparison = compare_extractions(pred_data, gt_data)
        comparison["filename"] = str(pred_path.relative_to(qwen_dir))
        results.append(comparison)
    
    duration = time.time() - start_time
    
    if not results:
        print("No matches found to benchmark.")
        return
        
    # Aggregate Metrics
    total = len(results)
    valid_count = sum(1 for r in results if r["is_valid"])
    valid_pct = (valid_count / total) * 100
    throughput = total / duration if duration > 0 else 0
    
    # Save CSV Report
    results.sort(key=lambda x: x["filename"])
    csv_path = bench_dir / "detailed_results.csv"
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    # Generate Summary JSON
    summary = {
        "total_images": total,
        "json_validity_pct": f"{valid_pct:.1f}%",
        "throughput_images_per_sec": f"{throughput:.2f}",
        "avg_time_per_image": f"{duration/total:.2f}s",
        "gpu_infrastructure": "4x NVIDIA RTX PRO 6000 Blackwell"
    }
    with open(bench_dir / "summary_metrics.json", 'w') as f:
        json.dump(summary, f, indent=4)

    # Plot Comparison
    filenames = [r["filename"].split('/')[-1][:10] for r in results]
    pred_counts = [r["pred_item_count"] for r in results]
    gt_counts = [r["gt_item_count"] for r in results]

    plt.figure(figsize=(15, 7))
    x = range(len(results))
    plt.bar(x, pred_counts, width=0.4, label='Qwen Item Count', align='center', color='skyblue', alpha=0.8)
    plt.bar(x, gt_counts, width=0.4, label='Ground Truth Count', align='edge', color='orange', alpha=0.8)
    plt.xlabel('Prescription Files')
    plt.ylabel('Number of Items')
    plt.title(f'Qwen2-VL Accuracy | JSON Validity: {valid_pct:.1f}% | {throughput:.2f} img/sec')
    plt.xticks(x, filenames, rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(bench_dir / "accuracy_plot.png")
    
    print(f"--- Engineering Metrics ---")
    print(f"JSON Validity: {valid_pct:.1f}%")
    print(f"Throughput:    {throughput:.2f} images/sec")
    print(f"Metrics saved to: {bench_dir / 'summary_metrics.json'}")
    print("-" * 30)

if __name__ == "__main__":
    main()
