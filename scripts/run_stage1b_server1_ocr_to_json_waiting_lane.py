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

"""Stage 1B Server 1 OCR-to-JSON waiting lane.

This script intentionally does not run unless a Server 2 OCR handoff CSV is
provided. It exists to document the approved future lanes and prevent accidental
launches before OCR text is available.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


APPROVED_TEXT_BACKENDS = {
    "ollama_qwen25_14b_ocr_to_json": "qwen2.5:14b",
    "ollama_qwen3_8b_ocr_to_json": "qwen3:8b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/validate OCR-to-JSON handoff lane")
    parser.add_argument("--handoff-csv", required=True, help="Server 2 OCR handoff CSV with document_id and ocr_text_path columns")
    parser.add_argument("--backend", choices=sorted(APPROVED_TEXT_BACKENDS), required=True)
    parser.add_argument("--dry-run", action="store_true", help="Only validate handoff shape; do not call Ollama")
    args = parser.parse_args()

    handoff = Path(args.handoff_csv)
    if not handoff.exists():
        raise SystemExit(f"OCR handoff CSV not found: {handoff}. Do not launch OCR-to-JSON yet.")
    with handoff.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"document_id", "ocr_text_path"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"OCR handoff CSV missing columns: {sorted(missing)}")
    if args.dry_run:
        print(f"handoff ok: {len(rows)} rows for {args.backend} ({APPROVED_TEXT_BACKENDS[args.backend]})")
        return
    raise SystemExit("Execution is intentionally disabled until Server 2 handoff is approved for launch.")


if __name__ == "__main__":
    main()
