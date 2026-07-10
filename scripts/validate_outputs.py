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
import json
import sys
from pathlib import Path

# Force project root onto path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.raw_extraction import CanonicalRawDoc

def validate_directory(input_dir: Path) -> bool:
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist", file=sys.stderr)
        return False
        
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return True
        
    print(f"Validating {len(json_files)} JSON files in {input_dir}...")
    print(f"{'File':<25} | {'Status':<10} | {'Coercions':<10} | {'Warnings':<10} | {'Error Details'}")
    print("-" * 100)
    
    all_valid = True
    for p in sorted(json_files):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Attempt validation against Pydantic schema
            doc = CanonicalRawDoc(**data)
            
            # Extract metadata details
            metadata = data.get("metadata", {})
            coercions_count = len(metadata.get("type_coercions", []))
            warnings_count = len(metadata.get("validation_warnings", []))
            
            print(f"{p.name:<25} | {'VALID':<10} | {coercions_count:<10} | {warnings_count:<10} | None")
        except Exception as e:
            print(f"{p.name:<25} | {'INVALID':<10} | {'-':<10} | {'-':<10} | {str(e)}")
            all_valid = False
            
    print("-" * 100)
    return all_valid

def main():
    parser = argparse.ArgumentParser(description="Validate extraction output directories against Pydantic schema")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing CanonicalRawDoc JSON files")
    args = parser.parse_args()
    
    input_path = PROJECT_ROOT / args.input_dir
    success = validate_directory(input_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
