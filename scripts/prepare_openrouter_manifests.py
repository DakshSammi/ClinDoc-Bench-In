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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def prepare_manifest(model_id: str):
    safe_model = model_id.replace("/", "_")
    input_path = PROJECT_ROOT / "data" / "manifest_canonical.csv"
    output_path = PROJECT_ROOT / "data" / f"manifest_openrouter_{safe_model}_smoke.csv"
    
    selected_docs = {"p1", "p45_4"}
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)
        
    rows_written = 0
    with open(input_path, "r", encoding="utf-8") as inf, open(output_path, "w", encoding="utf-8", newline="") as outf:
        reader = csv.DictReader(inf)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            if row["document_id"] in selected_docs:
                row["prediction_path"] = f"outputs/raw_extractions/openrouter/{safe_model}_prompt_v2_smoke/{row['document_id']}.json"
                writer.writerow(row)
                rows_written += 1
                
    print(f"Created manifest: {output_path} with {rows_written} documents.")

if __name__ == "__main__":
    prepare_manifest("google/gemini-2.5-flash")
    prepare_manifest("meta-llama/llama-3.2-11b-vision-instruct")
