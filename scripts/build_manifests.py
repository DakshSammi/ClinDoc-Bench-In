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

import os
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def find_image_path(doc_id, filename):
    if isinstance(filename, list):
        filename = filename[0]
    filename = str(filename)
    
    # Search for filename under prescriptions/
    presc_dir = PROJECT_ROOT / "prescriptions"
    
    # Check if direct file exists under prescriptions/
    direct_path = presc_dir / filename
    if direct_path.exists():
        return f"prescriptions/{filename}"
        
    # Check under specific doc subdirectories
    for path in presc_dir.rglob(filename):
        return str(path.relative_to(PROJECT_ROOT))
        
    # As a fallback, construct based on known patterns
    if "_" in doc_id:
        parent = doc_id.split("_")[0]
        return f"prescriptions/{parent}/{doc_id}/{filename}"
    return f"prescriptions/{filename}"

def main():
    gt_dir = PROJECT_ROOT / "raw_ground_truths"
    
    # Discover all GT json files
    gt_files = []
    # Check root GT files
    for f in gt_dir.glob("*.json"):
        gt_files.append(f)
    # Check subdirs p25, p36, p45
    for subdir in ["p25", "p36", "p45"]:
        subdir_path = gt_dir / subdir
        if subdir_path.exists():
            for f in subdir_path.glob("*.json"):
                gt_files.append(f)
                
    # Sort files to have stable order
    gt_files.sort(key=lambda x: x.name)
    
    manifest_rows = []
    for f in gt_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        meta = data.get("document_metadata", {})
        doc_id = meta.get("document_id", f.stem)
        patient_id = meta.get("patient_id", doc_id.split("_")[0])
        prescription_id = meta.get("prescription_id", doc_id)
        
        # Get source images
        src_images = meta.get("source_images", [])
        if not src_images and meta.get("source_image"):
            src_images = [meta.get("source_image")]
            
        # If source_images contains nested lists or unexpected types, flatten them
        flat_src_images = []
        for item in src_images:
            if isinstance(item, list):
                flat_src_images.extend(item)
            else:
                flat_src_images.append(item)
                
        total_pages = meta.get("total_pages", len(flat_src_images))
        
        # Map filenames to relative paths
        mapped_paths = []
        for img in flat_src_images:
            mapped_paths.append(find_image_path(doc_id, img))
            
        image_path_field = ";".join(mapped_paths)
        
        # Determine handwritten or scanned
        source_type = meta.get("source_type", "prescription")
        if "discharge" in source_type or "report" in source_type:
            handwritten_or_scanned = "scanned"
        else:
            handwritten_or_scanned = "handwritten"
            
        # Specialty
        speciality = "general_medicine"
        if "Baba" in meta.get("ocr_engine", "") or "netralaya" in str(f).lower() or doc_id in ["p1", "p2", "p3"]:
            speciality = "ophthalmology"
        elif "endocrinology" in str(data).lower():
            speciality = "endocrinology"
            
        # Template
        layout = data.get("document_layout", {})
        institution_template = layout.get("hospital_header", "generic")
        institution_template = institution_template.lower().replace(" (regd.)", "").replace(" ", "_").replace("/", "_").replace(",", "_")
        
        split = "validation"
        output_path = f"outputs/raw_extractions/qwen25_vl_7b/{doc_id}.json"
        
        manifest_rows.append({
            "document_id": doc_id,
            "patient_id": patient_id,
            "prescription_id": prescription_id,
            "image_path": image_path_field,
            "total_pages": total_pages,
            "handwritten_or_scanned": handwritten_or_scanned,
            "speciality": speciality,
            "institution_template": institution_template,
            "output_path": output_path,
            "split": split
        })
        
    # Write full manifest
    output_manifest = PROJECT_ROOT / "data" / "extraction_manifest.csv"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "document_id", "patient_id", "prescription_id", "image_path", 
        "total_pages", "handwritten_or_scanned", "speciality", 
        "institution_template", "output_path", "split"
    ]
    
    with open(output_manifest, "w", encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=headers)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
            
    print(f"Successfully wrote full manifest to {output_manifest} ({len(manifest_rows)} rows)")
    
    # Write smoke manifest (p1 and p2)
    smoke_manifest = PROJECT_ROOT / "data" / "extraction_manifest_smoke.csv"
    smoke_rows = [r for r in manifest_rows if r["document_id"] in ["p1", "p2"]]
    
    with open(smoke_manifest, "w", encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=headers)
        writer.writeheader()
        for row in smoke_rows:
            writer.writerow(row)
            
    print(f"Successfully wrote smoke manifest to {smoke_manifest} ({len(smoke_rows)} rows)")

if __name__ == "__main__":
    main()
