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

import asyncio
import base64
import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import InferenceClient
from PIL import Image

from src.adapters.backend_adapter_base import BaseBackendAdapter


TRANSIENT_MARKERS = (
    "429", "402", "rate", "quota", "timeout", "temporarily", "temporary", "unavailable",
    "overloaded", "503", "502", "504", "connection", "network",
    "payment required", "monthly included credits", "pre-paid credits", "depleted",
)


def _token_label(token: Optional[str], env_name: str = "HF_TOKEN") -> str:
    if not token:
        return f"{env_name}:missing"
    return f"{env_name}:***{token[-4:]}"


@dataclass
class HFTokenState:
    env_name: str
    token: str
    client: InferenceClient
    requests: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    rate_limits: int = 0
    last_error: str = ""

    @property
    def label(self) -> str:
        return _token_label(self.token, self.env_name)


class HFInferenceBackendAdapter(BaseBackendAdapter):
    """Hugging Face Inference API / Inference Providers chat-completions adapter.

    Uses the HF token only; no OpenRouter routing is performed here.
    """

    def __init__(
        self,
        model_id: str,
        token: Optional[str] = None,
        provider: str = "auto",
        max_image_dim: int = 1024,
        jpeg_quality: int = 85,
        timeout: float = 300.0,
        max_retries: int = 5,
        backoff_initial: float = 2.0,
        backoff_max: float = 90.0,
    ):
        super().__init__(name="hf_inference", model_id=model_id)
        self.provider = provider
        self.max_image_dim = max_image_dim
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.logger = logging.getLogger("HFInferenceBackendAdapter")
        token_pairs = []
        if token:
            token_pairs.append(("HF_TOKEN_ARG", token))
        for env_name in ["HF_TOKEN", "HF_TOKEN_2"]:
            value = os.getenv(env_name)
            if value and all(value != existing for _, existing in token_pairs):
                token_pairs.append((env_name, value))
        self.tokens: List[HFTokenState] = [
            HFTokenState(
                env_name=env_name,
                token=value,
                client=InferenceClient(model=model_id, provider=provider, token=value, timeout=timeout),
            )
            for env_name, value in token_pairs
        ]
        self._cursor = 0
        self.usage = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "provider": provider,
            "model": model_id,
            "tokens": [state.label for state in self.tokens],
        }

    @property
    def supports_structured_output(self) -> bool:
        return True

    def _encode_image_url(self, image: Image.Image) -> str:
        if max(image.size) > self.max_image_dim:
            w, h = image.size
            scale = self.max_image_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=self.jpeg_quality)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _is_transient(self, exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in text for marker in TRANSIENT_MARKERS)

    def _is_rate_or_quota(self, exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in text for marker in ("429", "402", "rate", "quota", "payment required", "monthly included credits", "pre-paid credits", "depleted"))

    def _is_rate_or_quota_text(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in ("429", "402", "rate", "quota", "payment required", "monthly included credits", "pre-paid credits", "depleted"))

    def _all_tokens_rate_limited(self) -> bool:
        if not self.tokens:
            return False
        return all(
            state.rate_limits > 0 or self._is_rate_or_quota_text(state.last_error)
            for state in self.tokens
        )

    def _pick_token(self) -> Optional[HFTokenState]:
        if not self.tokens:
            return None
        min_requests = min(t.requests for t in self.tokens)
        candidates = [t for t in self.tokens if t.requests == min_requests]
        for _ in range(len(self.tokens)):
            state = self.tokens[self._cursor % len(self.tokens)]
            self._cursor += 1
            if state in candidates:
                return state
        return candidates[0]

    async def run(self, prompt: str, image: Optional[Union[Image.Image, List[Image.Image]]] = None, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        decoding_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 0.9),
        }

        if not self.tokens:
            return {
                "error": "HF_TOKEN/HF_TOKEN_2 is not configured",
                "content": "",
                "processing_time_ms": 0.0,
                "model_name": self.model_id,
                "backend_name": self.name,
                "provider": self.provider,
                "api_key_label": "HF_TOKEN:missing",
                "decoding_parameters": decoding_params,
            }

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image:
            images = image if isinstance(image, list) else [image]
            for img in images:
                user_content.append({"type": "image_url", "image_url": {"url": self._encode_image_url(img)}})

        messages = [{"role": "user", "content": user_content}]
        last_error = None
        delay = self.backoff_initial
        last_token_label = ""

        for attempt in range(1, self.max_retries + 2):
            token_state = self._pick_token()
            if token_state is None:
                break
            last_token_label = token_state.label
            self.usage["requests"] += 1
            token_state.requests += 1
            try:
                self.logger.info(
                    "HF request model=%s provider=%s key=%s attempt=%s",
                    self.model_id, self.provider, token_state.label, attempt,
                )

                def make_call():
                    try:
                        return token_state.client.chat_completion(
                            messages=messages,
                            model=self.model_id,
                            max_tokens=decoding_params["max_tokens"],
                            temperature=decoding_params["temperature"],
                            top_p=decoding_params["top_p"],
                        )
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"Hugging Face Inference did not return an available provider for {self.model_id}"
                        ) from exc

                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, make_call)
                latency_ms = (time.time() - start_time) * 1000
                choices = getattr(response, "choices", None) or response.get("choices", [])
                content = ""
                if choices:
                    first = choices[0]
                    message = getattr(first, "message", None) or first.get("message", {})
                    content = getattr(message, "content", None) or message.get("content", "") or ""
                usage_obj = getattr(response, "usage", None) or (response.get("usage", {}) if isinstance(response, dict) else {})
                def usage_get(name: str) -> int:
                    if isinstance(usage_obj, dict):
                        return int(usage_obj.get(name, 0) or 0)
                    return int(getattr(usage_obj, name, 0) or 0)
                usage = {
                    "input_tokens": usage_get("prompt_tokens"),
                    "output_tokens": usage_get("completion_tokens"),
                    "total_tokens": usage_get("total_tokens"),
                }
                self.usage["successes"] += 1
                token_state.successes += 1
                return {
                    "content": content,
                    "processing_time_ms": latency_ms,
                    "model_name": self.model_id,
                    "backend_name": self.name,
                    "provider": self.provider,
                    "api_key_label": token_state.label,
                    "decoding_parameters": decoding_params,
                    "usage": usage,
                    "raw_response": response.model_dump() if hasattr(response, "model_dump") else response,
                    "retry_count": attempt - 1,
                }
            except Exception as exc:
                last_error = exc
                transient = self._is_transient(exc)
                token_state.failures += 1
                token_state.last_error = str(exc)
                if self._is_rate_or_quota(exc):
                    token_state.rate_limits += 1
                self.logger.warning(
                    "HF request failed model=%s provider=%s key=%s attempt=%s transient=%s error=%s",
                    self.model_id, self.provider, token_state.label, attempt, transient, str(exc)[:500],
                )
                if self._is_rate_or_quota(exc) and self._all_tokens_rate_limited():
                    self.logger.warning("All configured Hugging Face tokens appear quota/rate limited; stopping retry loop.")
                    break
                if attempt > self.max_retries or not transient:
                    break
                self.usage["retries"] += 1
                token_state.retries += 1
                await asyncio.sleep(delay)
                delay = min(self.backoff_max, delay * 2)

        self.usage["failures"] += 1
        return {
            "error": str(last_error) if last_error else "HF request failed",
            "content": "",
            "processing_time_ms": (time.time() - start_time) * 1000,
            "model_name": self.model_id,
            "backend_name": self.name,
            "provider": self.provider,
            "api_key_label": last_token_label,
            "decoding_parameters": decoding_params,
            "retry_count": self.usage["retries"],
            "hf_token_usage": [
                {
                    "key": state.label,
                    "requests": state.requests,
                    "successes": state.successes,
                    "failures": state.failures,
                    "retries": state.retries,
                    "rate_limits": state.rate_limits,
                    "last_error": state.last_error[:300],
                }
                for state in self.tokens
            ],
        }
