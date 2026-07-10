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

from __future__ import annotations

from typing import Optional


HF_MODEL_CANONICAL = {
    "qwen/qwen2.5-vl-72b-instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen/qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen/qwen2-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "meta-llama/llama-3.2-11b-vision-instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/florence-2-large": "microsoft/Florence-2-large",
}


def canonical_hf_model_id(model_id: str) -> str:
    return HF_MODEL_CANONICAL.get((model_id or "").lower(), model_id)


def is_gemini_model(model_id: Optional[str]) -> bool:
    model = (model_id or "").lower()
    return "gemini" in model or model.startswith("google/")


def provider_for_model(model_id: str) -> str:
    if is_gemini_model(model_id):
        return "gemini"
    return "hf_inference"
