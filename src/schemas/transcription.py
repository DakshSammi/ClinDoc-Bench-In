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

class LineItem(BaseModel):
    line_id: str = Field(..., description="Unique identifier for the line")
    text: str = Field(..., description="Verbatim text of the line")
    box: Optional[List[float]] = Field(None, description="Coordinates [x_min, y_min, x_max, y_max]")

class TranscriptionDoc(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    page_number: int = Field(1, description="Page number of the prescription")
    lines: List[LineItem] = Field(default_factory=list, description="Verbatim lines extracted")
    backend_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from VLM/OCR engine")
    processing_time_ms: float = Field(0.0, description="Time taken to process in milliseconds")
