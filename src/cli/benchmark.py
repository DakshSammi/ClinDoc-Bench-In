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

import argparse
from pathlib import Path
import json
from src.benchmark.runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Biomedical Prescription Extraction Benchmarking CLI")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to the manifest-driven source of truth CSV file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_defaults.yaml",
        help="Path to the evaluation YAML configuration file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save the seven separate output evaluation files."
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
        
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
        
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
        
    print("=" * 80)
    print("STARTING BIOMEDICAL PRESCRIPTION zrozum/EXTRACTION BENCHMARK RUNNER")
    print(f"Manifest Source of Truth: {manifest_path}")
    print(f"Config Parameters File:  {config_path}")
    print(f"Destination Output Dir:  {output_dir}")
    print("=" * 80)
    
    try:
        runner = BenchmarkRunner(
            manifest_path=manifest_path,
            project_root=project_root,
            config_path=config_path
        )
        summary = runner.run(output_dir=output_dir)
        
        print("\n" + "=" * 40 + " GLOBAL DATASET SUMMARY " + "=" * 40)
        print(f"Total Evaluated Documents:           {summary.get('total_documents', 0)}")
        print(f"Schema Parse Success Rate:           {summary.get('schema_parse_success_rate', 0.0)*100:.2f}%")
        print(f"Scalar Accuracy (Exact):             {summary.get('scalar_accuracy_exact', 0.0)*100:.2f}%")
        print(f"Scalar Accuracy (Lenient):           {summary.get('scalar_accuracy_lenient', 0.0)*100:.2f}%")
        print(f"Entity F1 Score (Exact Macro):       {summary.get('entity_exact_f1_macro', 0.0)*100:.2f}%")
        print(f"Entity F1 Score (Lenient Macro):     {summary.get('entity_lenient_f1_macro', 0.0)*100:.2f}%")
        print(f"Hallucination Rate (FP/TotalPred):   {summary.get('hallucination_rate', 0.0)*100:.2f}%")
        print(f"Missing Entity Rate (FN/TotalGT):    {summary.get('missing_entity_rate', 0.0)*100:.2f}%")
        print(f"Annotation-Gap Rate (Gaps/TotalPred):{summary.get('annotation_gap_rate', 0.0)*100:.2f}%")
        print(f"Experimental Headline Score (Weighted):{summary.get('experimental_overall_score', 0.0)*100:.2f}%")
        print("=" * 104)
        
        print("\nCategorized mismatch totals:")
        counts = summary.get("counts", {})
        print(f" - Likely Hallucinations:            {counts.get('likely_hallucination_total', 0)}")
        print(f" - Annotation Gap Candidates:        {counts.get('annotation_gap_candidate_total', 0)}")
        print(f" - Manual Review Queue Additions:     {counts.get('manual_review_required_total', 0)}")
        print("-" * 104)
        
        print(f"\nAll 7 reports successfully saved in: {output_dir.resolve()}")
        print("=" * 104)
        
    except Exception as e:
        print(f"\nCritical Benchmark Execution Failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
