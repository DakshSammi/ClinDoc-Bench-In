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
import logging
from typing import Any, Dict, Type, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class JSONRefinementAgent:
    """
    An extra layer that converts raw, noisy model output into a valid JSON format
    matching a specific Pydantic schema.
    """
    def __init__(self, primary_model_id: str, fallback_model_id: Optional[str] = None, config: Any = None):
        self.primary_model_id = primary_model_id
        self.fallback_model_id = fallback_model_id
        self.config = config
        self.primary_model = None
        self.fallback_model = None
        self.logger = logging.getLogger("JSONRefinementAgent")

    async def _ensure_models(self):
        from models.model_factory import ModelFactory
        if not self.primary_model:
            self.logger.info(f"Loading primary refinement model: {self.primary_model_id}")
            self.primary_model = ModelFactory.get_model(self.primary_model_id, self.config)
        if self.fallback_model_id and not self.fallback_model:
            self.logger.info(f"Loading fallback refinement model: {self.fallback_model_id}")
            self.fallback_model = ModelFactory.get_model(self.fallback_model_id, self.config)

    def _build_prompt(self, raw_text: str, schema: Type[BaseModel]) -> str:
        return f"""
You are a JSON formatting expert. Your task is to extract information from the noisy raw text below and format it into a valid JSON object that strictly adheres to the provided schema.

RAW TEXT:
---
{raw_text}
---

TARGET SCHEMA:
{schema.schema_json(indent=2)}

INSTRUCTIONS:
1. Extract all relevant fields.
2. Ensure the output is ONLY a valid JSON object.
3. Do not include markdown blocks or extra text.
4. If a field is missing, use null or an empty list/string as appropriate.
5. Preserve clinical details exactly.

JSON OUTPUT:
"""

    async def refine(self, raw_text: str, schema_class: Type[T]) -> T:
        await self._ensure_models()
        prompt = self._build_prompt(raw_text, schema_class)
        
        # Try primary model
        response = await self.primary_model.generate(prompt)
        
        # If primary fails (e.g. 429), try fallback
        if "error" in response and self.fallback_model:
            self.logger.warning(f"Primary refinement model failed: {response['error']}. Trying fallback...")
            response = await self.fallback_model.generate(prompt)
            
        if "error" in response:
            raise Exception(f"Refinement Model Error: {response['error']}")

        content = response["content"]
        # Basic cleaning
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        try:
            data = json.loads(content)
            # Validate with pydantic
            validated = schema(**data)
            return validated.dict()
        except Exception as e:
            self.logger.error(f"Refinement Parsing/Validation Error: {str(e)}")
            self.logger.debug(f"Failed Content: {content}")
            raise e
