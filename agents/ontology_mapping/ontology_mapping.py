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
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from utils.schemas import SemanticOutput, OntologyOutput, OntologyMapping
from utils.ontology_tools import OntologyTools

class OntologyMappingAgent(BaseAgent):
    def __init__(self, model_wrapper: Any, prompt_template: str, ontology_tools: OntologyTools):
        super().__init__("OntologyMappingAgent", model_wrapper)
        self.prompt_template = prompt_template
        self.ontology_tools = ontology_tools

    async def run(self, semantic_output: SemanticOutput) -> OntologyOutput:
        self.logger.info(f"Running Ontology Mapping with {self.model.model_id}")
        
        all_mappings = []
        for item in semantic_output.normalized_items:
            # 1. Retrieval
            candidates = self.ontology_tools.get_candidates(item.normalized_value)
            
            # 2. Reranking with LLM
            prompt = self.prompt_template.format(
                normalized_value=item.normalized_value,
                candidates_json=json.dumps(candidates, indent=2)
            )
            
            response = await self.model.generate(prompt)
            
            if "error" in response:
                self.logger.error(f"Model error for item {item.normalized_value}: {response['error']}")
                continue

            try:
                content = response["content"]
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                
                mapping_data = json.loads(content)
                for m in mapping_data.get("mappings", []):
                    all_mappings.append(OntologyMapping(
                        normalized_item=item,
                        **m
                    ))
            except Exception as e:
                self.logger.error(f"Parsing error for item {item.normalized_value}: {str(e)}")
        
        metadata = self._create_metadata(latency=0) # Latency is sum of items
        return OntologyOutput(metadata=metadata, mappings=all_mappings)
