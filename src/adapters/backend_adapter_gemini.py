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
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from src.adapters.backend_adapter_base import BaseBackendAdapter

try:
    from google import genai
except ImportError:
    genai = None


RATE_LIMIT_MARKERS = ("429", "rate", "quota", "resource_exhausted", "too many requests")
TRANSIENT_MARKERS = RATE_LIMIT_MARKERS + (
    "timeout", "temporarily", "temporary", "unavailable", "overloaded",
    "503", "502", "504", "connection", "network", "deadline",
)


@dataclass
class GeminiKeyState:
    env_name: str
    value: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    rate_limits: int = 0
    unavailable_until: float = 0.0
    last_error: str = ""
    client: Any = field(default=None)

    @property
    def label(self) -> str:
        return f"{self.env_name}:***{self.value[-4:]}"


class GeminiBackendAdapter(BaseBackendAdapter):
    """Official Gemini SDK adapter with automatic four-key rotation/failover."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gemini-2.5-flash",
        max_retries: int = 8,
        backoff_initial: float = 2.0,
        backoff_max: float = 90.0,
        rate_limit_cooldown: float = 60.0,
    ):
        super().__init__(name="gemini_rotating", model_id=model_id)
        self.logger = logging.getLogger("GeminiBackendAdapter")
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.rate_limit_cooldown = rate_limit_cooldown
        self._cursor = 0

        key_pairs = []
        for env_name in ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3", "GOOGLE_API_KEY_4"]:
            value = os.getenv(env_name)
            if value:
                key_pairs.append((env_name, value))
        if api_key and all(api_key != v for _, v in key_pairs):
            key_pairs.insert(0, ("GOOGLE_API_KEY_ARG", api_key))

        self.keys: List[GeminiKeyState] = []
        if genai:
            for env_name, value in key_pairs:
                try:
                    self.keys.append(GeminiKeyState(env_name=env_name, value=value, client=genai.Client(api_key=value)))
                except Exception as exc:
                    self.logger.warning("Could not initialise Gemini client for %s: %s", env_name, exc)

        if not genai:
            self.logger.warning("google-genai library is not installed.")
        if not self.keys:
            self.logger.warning("No Gemini API keys are configured.")

    @property
    def supports_structured_output(self) -> bool:
        return True

    def _is_rate_limit(self, exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in text for marker in RATE_LIMIT_MARKERS)

    def _is_transient(self, exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in text for marker in TRANSIENT_MARKERS)

    def _is_rate_limit_text(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in RATE_LIMIT_MARKERS)

    def _all_keys_rate_limited(self) -> bool:
        if not self.keys:
            return False
        return all(
            k.rate_limits > 0 or self._is_rate_limit_text(k.last_error)
            for k in self.keys
        )

    def _pick_key(self) -> Optional[GeminiKeyState]:
        if not self.keys:
            return None
        now = time.time()
        available = [k for k in self.keys if k.unavailable_until <= now]
        if not available:
            soonest = min(k.unavailable_until for k in self.keys)
            for k in self.keys:
                if k.unavailable_until == soonest:
                    return k
        min_requests = min(k.requests for k in available)
        candidates = [k for k in available if k.requests == min_requests]
        # Round-robin among least-used keys so one key is not hammered.
        for _ in range(len(self.keys)):
            key = self.keys[self._cursor % len(self.keys)]
            self._cursor += 1
            if key in candidates:
                return key
        return candidates[0]

    def usage_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model_id,
            "keys": [
                {
                    "key": k.label,
                    "requests": k.requests,
                    "successes": k.successes,
                    "failures": k.failures,
                    "retries": k.retries,
                    "rate_limits": k.rate_limits,
                    "temporarily_unavailable": k.unavailable_until > time.time(),
                    "last_error": k.last_error[:300],
                }
                for k in self.keys
            ],
        }

    async def run(self, prompt: str, image: Optional[Union[Image.Image, List[Image.Image]]] = None, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        decoding_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_output_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 0.9),
        }

        if not self.keys:
            return {
                "error": "google-genai library or GOOGLE_API_KEY(_2/_3/_4) is not configured",
                "content": "",
                "processing_time_ms": 0.0,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params,
                "gemini_usage": self.usage_stats(),
            }

        contents: List[Any] = [prompt]
        if image:
            contents.extend(image if isinstance(image, list) else [image])

        delay = self.backoff_initial
        last_error: Optional[BaseException] = None
        total_retries = 0

        for attempt in range(1, self.max_retries + 2):
            key = self._pick_key()
            if key is None:
                break
            wait_for = max(0.0, key.unavailable_until - time.time())
            if wait_for > 0:
                sleep_for = min(wait_for, delay)
                self.logger.info("All Gemini keys cooling down; sleeping %.1fs before retry.", sleep_for)
                await asyncio.sleep(sleep_for)

            key.requests += 1
            try:
                self.logger.info(
                    "Gemini request model=%s key=%s attempt=%s",
                    self.model_id, key.label, attempt,
                )

                def make_call():
                    return key.client.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config={
                            "temperature": decoding_params["temperature"],
                            "max_output_tokens": decoding_params["max_output_tokens"],
                            "top_p": decoding_params["top_p"],
                        },
                    )

                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, make_call)
                latency_ms = (time.time() - start_time) * 1000
                key.successes += 1
                usage_metadata = getattr(response, "usage_metadata", None)
                usage = {
                    "input_tokens": getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
                    "output_tokens": getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
                    "total_tokens": getattr(usage_metadata, "total_token_count", 0) if usage_metadata else 0,
                }
                return {
                    "content": getattr(response, "text", "") or "",
                    "processing_time_ms": latency_ms,
                    "model_name": self.model_id,
                    "backend_name": self.name,
                    "provider": "gemini",
                    "api_key_label": key.label,
                    "decoding_parameters": decoding_params,
                    "usage": usage,
                    "raw_response": response.model_dump() if hasattr(response, "model_dump") else None,
                    "retry_count": total_retries,
                    "gemini_usage": self.usage_stats(),
                }
            except Exception as exc:
                last_error = exc
                text = str(exc)
                key.last_error = text
                key.failures += 1
                is_rate_limit = self._is_rate_limit(exc)
                transient = self._is_transient(exc)
                if is_rate_limit:
                    key.rate_limits += 1
                    key.unavailable_until = time.time() + self.rate_limit_cooldown
                self.logger.warning(
                    "Gemini request failed model=%s key=%s attempt=%s transient=%s rate_limit=%s error=%s",
                    self.model_id, key.label, attempt, transient, is_rate_limit, text[:500],
                )
                if is_rate_limit and self._all_keys_rate_limited():
                    self.logger.warning("All configured Gemini keys appear quota/rate limited; stopping retry loop.")
                    break
                if attempt > self.max_retries or not transient:
                    break
                key.retries += 1
                total_retries += 1
                await asyncio.sleep(delay)
                delay = min(self.backoff_max, delay * 2)

        return {
            "error": str(last_error) if last_error else "Gemini request failed",
            "content": "",
            "processing_time_ms": (time.time() - start_time) * 1000,
            "model_name": self.model_id,
            "backend_name": self.name,
            "provider": "gemini",
            "decoding_parameters": decoding_params,
            "retry_count": total_retries,
            "gemini_usage": self.usage_stats(),
        }
