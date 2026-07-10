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

"""Stage 1B Server 1 low-priority LLaVA diagnostic runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import run_stage1b_server1_full as base


base.APPROVED_BACKEND = "ollama_llava_13b_raw_structured"
base.OUTPUT_BACKEND = "ollama_llava_13b"

REPORTS_DIR = base.REPORTS_DIR
OUTPUT_BACKEND = "ollama_llava_13b"
SERVER_NAME = base.SERVER_NAME


class LLaVARunner(base.Runner):
    def __init__(self, args: argparse.Namespace):
        if args.backend != "ollama_llava_13b_raw_structured":
            raise SystemExit("Only approved LLaVA backend is ollama_llava_13b_raw_structured")
        if args.model != "llava:13b":
            raise SystemExit("Only approved LLaVA model is llava:13b")
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.docs = base.load_manifest(Path(args.manifest))
        self.prompt, self.prompt_hash = base.read_prompt()
        self.start_time = time.time()
        self.current_doc = ""
        self.skipped = 0
        self.pid = os.getpid()
        self.checkpoint_path = self.output_root / "checkpoints" / "stage1b_server1_llava.jsonl"
        self.failure_events = []

    def build_prompt(self, doc: Dict[str, str]) -> str:
        strict = (
            "STRICT OUTPUT REQUIREMENT: Return one valid JSON object only. "
            "Do not use Markdown fences, explanations, comments, or trailing commas. "
            "If text is uncertain, preserve it as visible/uncertain text in JSON fields.\n\n"
        )
        return strict + super().build_prompt(doc)

    def write_progress(self) -> None:
        counts = self.status_counts()
        done_for_eta = counts["completed"] + counts["failed"] + self.skipped
        progress = {
            "generated": base.now(),
            "server_name": SERVER_NAME,
            "backend": self.args.backend,
            "model": self.args.model,
            "total_records": len(self.docs),
            "completed": counts["completed"],
            "failed": counts["failed"],
            "skipped": self.skipped,
            "currently_running_document": self.current_doc if counts["running"] else "",
            "running_count": counts["running"],
            "eta": self.eta(done_for_eta),
            "last_5_failures": self.last_failures(),
            "output_directory": str(self.output_root),
            "pid": self.pid,
            "log_path": self.args.process_log_path or "",
        }
        base.write_json_atomic(REPORTS_DIR / "stage1b_server1_llava_progress.json", progress)
        lines = [
            "# Stage 1B Server 1 LLaVA Progress",
            "",
            f"Generated: {progress['generated']}",
            "",
            f"- Output directory: `{progress['output_directory']}`",
            f"- PID: `{progress['pid']}`",
            f"- Backend: `{self.args.backend}`",
            f"- Model: `{self.args.model}`",
            f"- Total records: {progress['total_records']}",
            f"- Completed: {progress['completed']}",
            f"- Failed: {progress['failed']}",
            f"- Skipped: {progress['skipped']}",
            f"- Currently running document: `{progress['currently_running_document']}`",
            f"- ETA: {progress['eta']}",
            f"- Process log: `{progress['log_path']}`",
            "",
            "## Last 5 Failures",
            "",
        ]
        lines.extend([f"- `{f.get('document_id')}`: {f.get('error')}" for f in progress["last_5_failures"]] or ["No failures recorded yet."])
        base.write_text(REPORTS_DIR / "stage1b_server1_llava_progress.md", "\n".join(lines) + "\n")

    def write_interim_summary(self) -> None:
        return None

    def write_final_reports(self) -> None:
        metrics = base.compute_all_metrics(self.docs, self.output_root)
        fields = [
            "document_id", "patient_id", "department_inferred", "is_multi_page", "is_same_page_multi_view",
            "json_parse_success", "schema_validity", "output_completeness", "field_coverage",
            "scalar_accuracy_exact", "scalar_accuracy_lenient", "entity_exact_f1", "entity_lenient_f1",
            "hallucination_rate", "missing_entity_rate", "runtime_seconds", "status", "notes",
        ]
        base.write_csv(REPORTS_DIR / "stage1b_server1_llava_metrics.csv", metrics, fields)
        base.write_text(REPORTS_DIR / "stage1b_server1_llava_failure_log.md", base.render_failure_log(self.output_root))
        base.write_text(REPORTS_DIR / "stage1b_server1_llava_summary.md", base.render_summary(metrics, self.output_root).replace("Qwen3-VL", "LLaVA"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1B Server 1 LLaVA diagnostic runner")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--track", default="raw_structured")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-image-dim", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--single-worker", action="store_true")
    parser.add_argument("--process-log-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LLaVARunner(args).run()


if __name__ == "__main__":
    main()
