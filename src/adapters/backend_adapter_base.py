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

import time
from typing import Dict, Any, Optional
from PIL import Image

class BaseBackendAdapter:
    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    @property
    def supports_image_input(self) -> bool:
        return True

    @property
    def supports_text_input(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return False

    async def run(self, prompt: str, image: Optional[Image.Image] = None, **kwargs) -> Dict[str, Any]:
        """
        Executes inference. Returns dict containing:
        - content: raw generated text string
        - processing_time_ms: latency of model execution
        - model_name: exact model name
        - backend_name: name of this backend
        - decoding_parameters: dict of settings used (temp, etc)
        """
        raise NotImplementedError
