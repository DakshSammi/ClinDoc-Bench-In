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
import time
from typing import Any, Dict, Optional
from datetime import datetime
from utils.schemas import Metadata

class BaseAgent:
    def __init__(self, agent_name: str, model_wrapper: Any, refiner: Optional[Any] = None):
        self.agent_name = agent_name
        self.model = model_wrapper
        self.refiner = refiner
        self.logger = logging.getLogger(agent_name)

    def _create_metadata(self, latency: float, confidence: float = 1.0, notes: str = None) -> Metadata:
        return Metadata(
            model_name=self.model.model_id,
            timestamp=datetime.now().isoformat(),
            confidence_score=confidence,
            processing_time_ms=latency,
            uncertainty_notes=notes
        )

    def save_output(self, output_path: str, data: Dict[str, Any]):
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Output saved to {output_path}")

    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
