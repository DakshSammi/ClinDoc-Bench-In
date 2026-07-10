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

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.schemas.normalised import NormalisedClinicalDoc

class OntologyCandidate(BaseModel):
    ontology_id: str = Field(..., description="ID / URI of the concept")
    label: str = Field(..., description="Standard label of the concept")
    score: float = Field(..., description="Relevance or matching score")
    ontology_name: str = Field(..., description="SNOMEDCT, RXNORM, etc.")

class OntologyMappingRecord(BaseModel):
    raw_text: str
    normalized_text: str
    mapped_concept_id: Optional[str] = None
    mapped_concept_label: Optional[str] = None
    ontology_name: Optional[str] = None
    mapping_status: str = Field(..., description="mapped, unmapped, ambiguous, review_required")
    mapping_source: str = Field(..., description="BioPortal, AberOWL, LocalDict")
    confidence: float = 1.0
    top_k_candidates: List[OntologyCandidate] = Field(default_factory=list)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    mapping_reasoning: Optional[str] = None
    requires_human_review: bool = False

class OntologyMappedDoc(BaseModel):
    document_id: str
    schema_version: str = "ontology_mapped_rx_v1"
    normalised_doc: NormalisedClinicalDoc = Field(..., description="Reference back to normalized document")
    conditions: List[OntologyMappingRecord] = Field(default_factory=list)
    medications: List[OntologyMappingRecord] = Field(default_factory=list)
    procedures: List[OntologyMappingRecord] = Field(default_factory=list)
    observations: List[OntologyMappingRecord] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
