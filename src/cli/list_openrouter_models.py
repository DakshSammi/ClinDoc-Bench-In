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
import urllib.request
import sys

def main():
    parser = argparse.ArgumentParser(description="List available models from OpenRouter")
    parser.add_argument("--vision-only", action="store_true", help="Filter and display only vision-capable models")
    args = parser.parse_args()
    
    url = "https://openrouter.ai/api/v1/models"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Clinical-Prescription-Pipeline/1.0"})
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching models from OpenRouter: {e}", file=sys.stderr)
        sys.exit(1)
        
    models = res_data.get("data", [])
    print(f"{'Model ID':<65} | {'Name':<40} | {'Prompt ($/1M)':<15} | {'Completion ($/1M)':<15} | {'Context':<10}")
    print("-" * 155)
    
    count = 0
    for m in models:
        model_id = m.get("id", "")
        name = m.get("name", "")
        pricing = m.get("pricing", {})
        prompt_price = pricing.get("prompt", "0")
        completion_price = pricing.get("completion", "0")
        
        try:
            prompt_cost_1m = float(prompt_price) * 1_000_000
            prompt_str = f"${prompt_cost_1m:.4f}"
        except ValueError:
            prompt_str = str(prompt_price)
            
        try:
            completion_cost_1m = float(completion_price) * 1_000_000
            completion_str = f"${completion_cost_1m:.4f}"
        except ValueError:
            completion_str = str(completion_price)
            
        context_len = m.get("context_length", "unknown")
        
        # Determine vision capability
        architecture = m.get("architecture", {}) or {}
        input_modalities = architecture.get("input_modalities", []) or []
        modality = architecture.get("modality", "") or ""
        description = m.get("description", "").lower() if m.get("description") else ""
        
        is_vision = False
        if "image" in input_modalities:
            is_vision = True
        elif modality and "image" in modality:
            is_vision = True
        elif "vision" in model_id.lower() or "vl" in model_id.lower() or "gemini" in model_id.lower() or "claude-3" in model_id.lower() or "gpt-4o" in model_id.lower():
            is_vision = True
            
        if args.vision_only and not is_vision:
            continue
            
        # Truncate names/IDs if they are too long for nice layout
        disp_id = model_id[:63] + ".." if len(model_id) > 65 else model_id
        disp_name = name[:38] + ".." if len(name) > 40 else name
        
        print(f"{disp_id:<65} | {disp_name:<40} | {prompt_str:<15} | {completion_str:<15} | {context_len:<10}")
        count += 1
        
    print("-" * 155)
    print(f"Total models displayed: {count}")

if __name__ == "__main__":
    main()
