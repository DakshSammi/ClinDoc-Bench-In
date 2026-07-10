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
from agents.base_agent import BaseAgent
from utils.schemas import RawExtraction, SemanticOutput, NormalizedItem

class SemanticNormalizationAgent(BaseAgent):
    def __init__(self, model_wrapper: Any, prompt_template: str, refiner: Optional[Any] = None):
        super().__init__("SemanticNormalizationAgent", model_wrapper, refiner)
        self.prompt_template = prompt_template

    async def run(self, raw_extraction: RawExtraction) -> SemanticOutput:
        self.logger.info(f"Running Semantic Normalization with {self.model.model_id}")
        
        items_json = json.dumps([item.dict() for item in raw_extraction.items], indent=2)
        prompt = self.prompt_template.format(raw_items_json=items_json)
        
        response = await self.model.generate(prompt)
        
        if "error" in response:
            raise Exception(f"Model Error: {response['error']}")

        try:
            content = response["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            try:
                norm_data = json.loads(content)
            except json.JSONDecodeError:
                if self.refiner:
                    self.logger.info("Initial JSON parse failed. Attempting refinement...")
                    norm_data = await self.refiner.refine(content, SemanticOutput)
                else:
                    raise
            
            normalized_items = []
            # Handle both norm_data as dict or list depending on output
            items_list = norm_data.get("normalized_items", []) if isinstance(norm_data, dict) else norm_data
            for i, item in enumerate(items_list):
                # Link back to raw item for provenance
                raw_item = raw_extraction.items[i] if i < len(raw_extraction.items) else None
                normalized_items.append(NormalizedItem(
                    raw_item=raw_item,
                    **item
                ))
            
            metadata = self._create_metadata(
                latency=response["latency"],
                confidence=norm_data.get("confidence", 0.9)
            )
            
            return SemanticOutput(
                metadata=metadata,
                normalized_items=normalized_items,
                summary=norm_data.get("summary")
            )
        except Exception as e:
            self.logger.error(f"Parsing Error: {str(e)}")
            raise e
