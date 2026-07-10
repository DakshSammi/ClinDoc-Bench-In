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

"""Gemma 4 OCR smoke test for Phase C follow-up.

Runs transcription only, not CanonicalRawDoc extraction.
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OCR_PROMPT = (
    "Transcribe all visible clinical text from this medical document. "
    "Preserve line breaks, abbreviations, medicine names, dosages, frequencies, "
    "eye laterality markers such as RE/LE/BE, lab values, units, and reference ranges. "
    "Return only the transcription. Do not summarize."
)

DOCS = [
    ("p1", "prescriptions/p1.jpeg", 560),
    ("p2", "prescriptions/p2.jpeg", 560),
    ("p45_4_page1", "prescriptions/p45/p45_4/Adobe Scan May 15, 2026 (3)_page-0001.jpg", 1120),
]


def resize_for_budget(image: Image.Image, visual_token_budget: int) -> Image.Image:
    # Keep this conservative and transparent. 560 ~= 1024px long edge; 1120 ~= 1536px.
    max_dim = 1536 if visual_token_budget >= 1120 else 1024
    if max(image.size) <= max_dim:
        return image
    w, h = image.size
    scale = max_dim / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def build_inputs(processor, image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def decode_new_tokens(processor, generated, inputs):
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def append_usage(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists() and log_path.stat().st_size > 0
    fieldnames = [
        "timestamp", "document_id", "model", "image_path", "visual_token_budget",
        "max_new_tokens", "latency_ms", "status", "error", "output_chars",
    ]
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemma 4 OCR/transcription smoke test.")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--output-dir", default="outputs/ocr_transcriptions/gemma4_12b_smoke")
    parser.add_argument("--raw-response-dir", default="outputs/raw_responses/gemma4_12b_smoke")
    parser.add_argument("--usage-log", default="logs/gemma4_12b_usage.csv")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    raw_dir = PROJECT_ROOT / args.raw_response_dir
    usage_log = PROJECT_ROOT / args.usage_log
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_start = time.time()
    try:
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if not torch.cuda.is_available():
            model.to(device)
    except Exception as exc:
        raw_dir.mkdir(parents=True, exist_ok=True)
        error = str(exc)
        (raw_dir / "model_load_error.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "stage": "model_load",
                    "local_files_only": args.local_files_only,
                    "error": error,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        append_usage(
            usage_log,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "document_id": "model_load",
                "model": args.model,
                "image_path": "",
                "visual_token_budget": "",
                "max_new_tokens": args.max_new_tokens,
                "latency_ms": f"{(time.time() - load_start) * 1000:.2f}",
                "status": "failed_model_load",
                "error": error[:500],
                "output_chars": 0,
            },
        )
        print(f"model_load: failed ({(time.time() - load_start) * 1000:.1f} ms)")
        print(f"  error: {error}")
        return 1

    for doc_id, rel_image_path, visual_token_budget in DOCS:
        image_path = PROJECT_ROOT / rel_image_path
        start = time.time()
        status = "success"
        error = ""
        transcription = ""
        try:
            image = Image.open(image_path).convert("RGB")
            image = resize_for_budget(image, visual_token_budget)
            inputs = build_inputs(processor, image, OCR_PROMPT)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "enable_thinking": False,
            }
            try:
                generated = model.generate(**inputs, **generate_kwargs)
            except TypeError:
                generate_kwargs.pop("enable_thinking", None)
                generated = model.generate(**inputs, **generate_kwargs)
            transcription = decode_new_tokens(processor, generated, inputs)
            (output_dir / f"{doc_id}.txt").write_text(transcription + "\n", encoding="utf-8")
            (raw_dir / f"{doc_id}.json").write_text(
                json.dumps(
                    {
                        "document_id": doc_id,
                        "model": args.model,
                        "image_path": rel_image_path,
                        "visual_token_budget": visual_token_budget,
                        "prompt": OCR_PROMPT,
                        "transcription": transcription,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            (raw_dir / f"{doc_id}_error.json").write_text(
                json.dumps(
                    {
                        "document_id": doc_id,
                        "model": args.model,
                        "image_path": rel_image_path,
                        "visual_token_budget": visual_token_budget,
                        "error": error,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        latency_ms = (time.time() - start) * 1000
        append_usage(
            usage_log,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "document_id": doc_id,
                "model": args.model,
                "image_path": rel_image_path,
                "visual_token_budget": visual_token_budget,
                "max_new_tokens": args.max_new_tokens,
                "latency_ms": f"{latency_ms:.2f}",
                "status": status,
                "error": error[:500],
                "output_chars": len(transcription),
            },
        )
        print(f"{doc_id}: {status} ({latency_ms:.1f} ms)")
        if status != "success":
            print(f"  error: {error}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
