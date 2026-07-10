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
from typing import Any, Dict, List, Optional
from PIL import Image
from agents.base_agent import BaseAgent
from utils.schemas import RawExtraction, RawPrescriptionItem, Metadata

class RawExtractionAgent(BaseAgent):
    def __init__(self, model_wrapper: Any, prompt_template: str, refiner: Optional[Any] = None):
        super().__init__("RawExtractionAgent", model_wrapper, refiner)
        self.prompt_template = prompt_template

    async def run(self, image: Image.Image) -> RawExtraction:
        self.logger.info(f"Running Raw Extraction with {self.model.model_id}")
        
        response = await self.model.generate(self.prompt_template, image=image)
        
        if "error" in response:
            raise Exception(f"Model Error: {response['error']}")

        try:
            # Attempt to parse JSON from response
            content = response["content"]
            # Basic cleaning if LLM returns markdown blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            try:
                raw_data = json.loads(content)
            except json.JSONDecodeError:
                if self.refiner:
                    self.logger.info("Initial JSON parse failed. Attempting refinement...")
                    raw_data = await self.refiner.refine(content, RawExtraction)
                else:
                    raise

            items = []
            # Handle both raw_data as dict or list of items depending on refiner output
            items_list = raw_data.get("items", []) if isinstance(raw_data, dict) else raw_data
            for item in items_list:
                items.append(RawPrescriptionItem(**item))
            
            metadata = self._create_metadata(
                latency=response["latency"],
                confidence=raw_data.get("confidence", 0.8) if isinstance(raw_data, dict) else 0.8,
                notes=response.get("note")
            )
            
            return RawExtraction(
                metadata=metadata,
                patient_info=raw_data.get("patient_info", {}) if isinstance(raw_data, dict) else {},
                clinical_notes=raw_data.get("clinical_notes") if isinstance(raw_data, dict) else None,
                items=items,
                raw_json_string=response["content"]
            )
        except Exception as e:
            self.logger.error(f"Parsing Error: {str(e)}")
            # Fallback or retry logic can go here
            raise e
