#!/usr/bin/env python3
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

"""Controlled page-level p36_1 transcription diagnostic using internal qwen3."""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import dotenv

dotenv.load_dotenv(PROJECT_ROOT / ".env")
dotenv.load_dotenv(PROJECT_ROOT.parent / ".env")

from src.adapters.backend_adapter_openai_compatible_vlm import OpenAICompatibleVLMBackendAdapter
from src.cli.extract import (
    estimate_compressed_image_size_kb,
    get_internal_qwen3_usage_tokens,
    log_internal_qwen3_usage,
)
from src.utils.rate_limiter import init_global_limiter, get_global_limiter

PROMPT = (
    "Transcribe all visible clinical text from this page. Preserve line breaks, "
    "abbreviations, medicine names, dosages, dates, values, headings, and clinical shorthand. "
    "Return only the transcription. Do not summarize. Do not output JSON."
)

PAGES = [
    (1, "prescriptions/p36/p36_1/presc_image16902685410.jpg"),
    (2, "prescriptions/p36/p36_1/presc_image16902685411.jpg"),
]


def record_usage(document_id, response, estimated_input_tokens, estimated_output_tokens, reason):
    api_prompt_tokens, api_completion_tokens, api_total_tokens = get_internal_qwen3_usage_tokens(response)
    limiter = get_global_limiter()
    actual_usage_available = api_total_tokens > 0
    if actual_usage_available:
        limiter.record_usage(document_id, api_prompt_tokens, api_completion_tokens, estimated=False, reason=reason)
    else:
        limiter.record_usage(document_id, estimated_input_tokens, estimated_output_tokens, estimated=True, reason=f"{reason}_estimated_fallback")
    return api_prompt_tokens, api_completion_tokens, api_total_tokens, actual_usage_available


async def main():
    out_dir = PROJECT_ROOT / "outputs/ocr_transcriptions/qwen3_27b_p36_1_pages"
    raw_dir = PROJECT_ROOT / "outputs/raw_responses/qwen3_27b_p36_1_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("QWEN3_27B_API_KEY")
    if not api_key:
        raise SystemExit("QWEN3_27B_API_KEY is not set")

    adapter = OpenAICompatibleVLMBackendAdapter(
        base_url=os.getenv("QWEN3_27B_BASE_URL", "http://10.10.110.37:4000/v1"),
        api_key=api_key,
        model_id=os.getenv("QWEN3_27B_MODEL", "qwen3-27b"),
        max_image_dim=1024,
        jpeg_quality=85,
        timeout=900,
    )
    init_global_limiter(tpm_limit=500000, rpm_limit=120, window_seconds=60, buffer_seconds=15, max_retries_rate_limit=1)

    results = []
    empty_count = 0
    for idx, (page_no, rel_path) in enumerate(PAGES):
        doc_id = f"p36_1_page{page_no}_transcription"
        image = Image.open(PROJECT_ROOT / rel_path).convert("RGB")
        compressed_kb = estimate_compressed_image_size_kb([image], max_image_dim=1024, jpeg_quality=85)
        limiter = get_global_limiter()
        components = limiter.estimate_token_components(
            document_id=doc_id,
            max_tokens=8000,
            num_images=1,
            compressed_image_size_kb=compressed_kb,
            prompt_length_chars=len(PROMPT),
            reserve_full_output_budget=True,
        )
        rolling_before = limiter.get_rolling_tokens()
        sleep_before = limiter.wait_if_needed(doc_id, components["total_tokens"])

        start = time.time()
        response = await adapter.run(
            prompt=PROMPT,
            image=image,
            temperature=0.0,
            max_tokens=8000,
            top_p=0.9,
        )
        latency_ms = response.get("processing_time_ms", (time.time() - start) * 1000)
        content = response.get("content") or ""
        status = "success" if content.strip() and "error" not in response else "failed"
        error_text = response.get("error", "")
        if not content.strip():
            empty_count += 1
            status = "failed"
            error_text = error_text or "empty transcription response"

        raw_payload = {
            "document_id": doc_id,
            "page_number": page_no,
            "image_path": rel_path,
            "prompt": PROMPT,
            "status": status,
            "error": error_text,
            "content": content,
            "response_metadata": {k: v for k, v in response.items() if k not in {"content"}},
        }
        (raw_dir / f"p36_1_page{page_no}.json").write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (raw_dir / f"p36_1_page{page_no}.txt").write_text(content, encoding="utf-8")
        if status == "success":
            (out_dir / f"p36_1_page{page_no}.txt").write_text(content.strip() + "\n", encoding="utf-8")

        api_prompt, api_completion, api_total, actual_usage = record_usage(
            doc_id,
            response,
            components["input_tokens"],
            components["output_budget_tokens"],
            "p36_1_page_transcription",
        )
        log_internal_qwen3_usage(
            log_path=PROJECT_ROOT / "logs/internal_qwen3_usage.csv",
            document_id=doc_id,
            model=os.getenv("QWEN3_27B_MODEL", "qwen3-27b"),
            num_images=1,
            max_tokens=8000,
            latency_ms=latency_ms,
            status=status,
            error_type="" if status == "success" else "page_transcription_error",
            validation_status="text" if status == "success" else "invalid",
            notes=(f"page={page_no}; chars={len(content.strip())}; actual_usage={actual_usage}; error={error_text}"[:200]),
            estimated_input_tokens=components["input_tokens"],
            estimated_output_tokens=components["output_budget_tokens"],
            total_estimated_tokens=components["total_tokens"],
            api_reported_prompt_tokens=api_prompt if api_prompt else None,
            api_reported_completion_tokens=api_completion if api_completion else None,
            api_reported_total_tokens=api_total if api_total else None,
            rolling_tokens_before_request=rolling_before,
            sleep_seconds_before_request=sleep_before,
            retry_after_rate_limit=False,
            rate_limit_reason=None,
        )
        result = {
            "page": page_no,
            "status": status,
            "chars": len(content.strip()),
            "latency_ms": latency_ms,
            "estimated_total_tokens": components["total_tokens"],
            "actual_usage_available": actual_usage,
            "error": error_text,
        }
        results.append(result)
        print(result)

        if empty_count > 2:
            break
        if idx < len(PAGES) - 1:
            time.sleep(75)

    (raw_dir / "p36_1_page_diagnostic_summary.json").write_text(
        json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2),
        encoding="utf-8",
    )
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
