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
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import precision_recall_fscore_support

class EvaluationLayer:
    def __init__(self, ground_truth_dir: Path, output_dir: Path):
        self.ground_truth_dir = ground_truth_dir
        self.output_dir = output_dir

    def load_json(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return json.load(f)

    def calculate_metrics(self, ground_truth: List[str], predicted: List[str]):
        # Simple exact match precision/recall for tokens/items
        gt_set = set(ground_truth)
        pred_set = set(predicted)
        
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1

    def run_benchmarking(self, layer: str):
        results = []
        layer_path = self.output_dir / layer
        
        for model_dir in layer_path.iterdir():
            if not model_dir.is_dir(): continue
            
            for pred_file in model_dir.glob("*.json"):
                gt_file = self.ground_truth_dir / pred_file.name
                if not gt_file.exists(): continue
                
                gt_data = self.load_json(gt_file)
                pred_data = self.load_json(pred_file)
                
                # Extract items for comparison
                gt_items = [item.get("raw_text", "") for item in gt_data.get("items", [])]
                pred_items = [item.get("raw_text", "") for item in pred_data.get("items", [])]
                
                p, r, f1 = self.calculate_metrics(gt_items, pred_items)
                
                results.append({
                    "model": model_dir.name,
                    "file": pred_file.name,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "latency_ms": pred_data.get("metadata", {}).get("processing_time_ms", 0)
                })
        
        df = pd.DataFrame(results)
        return df

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    import configs.config as config
    evaluator = EvaluationLayer(config.PROJECT_ROOT / "raw_ground_truths", config.OUTPUTS_DIR)
    
    # Raw Extraction Benchmarks
    raw_df = evaluator.run_benchmarking("raw_extractions")
    raw_df.to_csv(config.OUTPUTS_DIR / "benchmarks" / "raw_benchmarks.csv", index=False)
    print("Raw Extraction Benchmarks saved.")
    
    # Summary Table
    summary = raw_df.groupby("model")[["precision", "recall", "f1", "latency_ms"]].mean()
    summary.to_markdown(config.OUTPUTS_DIR / "benchmarks" / "summary_report.md")
    print("Summary report generated.")
