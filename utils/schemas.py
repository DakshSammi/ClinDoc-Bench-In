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

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    model_name: str
    timestamp: str
    confidence_score: float
    processing_time_ms: float
    uncertainty_notes: Optional[str] = None

class RawPrescriptionItem(BaseModel):
    # Flexible fields to match Qwen and other LLMs
    medicine_name: Optional[str] = None
    dosage: Optional[str] = None
    instructions: Optional[str] = None
    
    # Compatibility fields
    raw_text: Optional[str] = None
    category: Optional[str] = "medication"
    confidence: Optional[float] = 1.0
    is_ambiguous: bool = False
    alternatives: List[str] = []

class RawExtraction(BaseModel):
    metadata: Metadata
    patient_info: Dict[str, Any]
    # Extremely flexible clinical_notes to handle strings, dicts, or lists of dicts
    clinical_notes: Optional[Any] = None
    items: List[RawPrescriptionItem]
    raw_json_string: str # Full raw output for provenance

class NormalizedItem(BaseModel):
    raw_item: RawPrescriptionItem
    normalized_value: str
    expansion: Optional[str] = None
    semantic_confidence: float
    reasoning: Optional[str] = None

class SemanticOutput(BaseModel):
    metadata: Metadata
    normalized_items: List[NormalizedItem]
    summary: Optional[str] = None

class OntologyMapping(BaseModel):
    normalized_item: NormalizedItem
    mapped_id: str # e.g., SNOMED:123
    ontology_name: str # SNOMED, RxNorm, etc.
    mapped_label: str
    similarity_score: float
    provenance_url: Optional[str] = None

class OntologyOutput(BaseModel):
    metadata: Metadata
    mappings: List[OntologyMapping]
