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
import shutil
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.resolve()
    manifest_canonical_path = project_root / "data" / "manifest_canonical.csv"
    subset_dir = project_root / "outputs" / "raw_extractions" / "qwen25_vl_7b_v2_subset"
    full_dir = project_root / "outputs" / "raw_extractions" / "qwen25_vl_7b_v2_full"
    
    # 1. Create full directory if it doesn't exist
    full_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified directory: {full_dir}")

    # 2. Copy the 5 existing subset files
    subset_files = ["p1.json", "p2.json", "p45_1.json", "p45_3.json", "p45_4.json"]
    copied_count = 0
    for f_name in subset_files:
        src = subset_dir / f_name
        dst = full_dir / f_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src.name} to full directory.")
            copied_count += 1
        else:
            print(f"Warning: Subset file {src} not found!")

    # 3. Read manifest_canonical.csv and partition documents
    if not manifest_canonical_path.exists():
        print(f"Error: manifest_canonical.csv not found at {manifest_canonical_path}")
        return

    rows = []
    with open(manifest_canonical_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # 4. Generate manifest_qwen_v2_full.csv
    # This manifest maps all 14 documents to their prediction paths in qwen25_vl_7b_v2_full
    full_manifest_path = project_root / "data" / "manifest_qwen_v2_full.csv"
    with open(full_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            new_row = dict(row)
            doc_id = row["document_id"]
            new_row["prediction_path"] = f"outputs/raw_extractions/qwen25_vl_7b_v2_full/{doc_id}.json"
            writer.writerow(new_row)
    print(f"Generated full manifest: {full_manifest_path}")

    # 5. Generate manifest_qwen_v2_missing.csv for the remaining 9 documents
    missing_manifest_path = project_root / "data" / "manifest_qwen_v2_missing.csv"
    missing_docs = []
    with open(missing_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            doc_id = row["document_id"]
            if f"{doc_id}.json" not in subset_files:
                new_row = dict(row)
                new_row["prediction_path"] = f"outputs/raw_extractions/qwen25_vl_7b_v2_full/{doc_id}.json"
                writer.writerow(new_row)
                missing_docs.append(doc_id)
    print(f"Generated missing manifest ({len(missing_docs)} docs): {missing_manifest_path}")
    print(f"Missing documents list: {missing_docs}")

if __name__ == "__main__":
    main()
