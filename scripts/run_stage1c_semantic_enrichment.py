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

"""Run resume-safe Stage 1C evidence-backed semantic enrichment with Ollama."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas.semantic_extraction import SemanticExtractionDoc


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_json(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    if "```json" in text:
        candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in list(candidates):
        candidates.append(re.sub(r",\s*([}\]])", r"\1", candidate))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return None


def parse_source(spec: str) -> tuple[str, Path, str]:
    parts = spec.split("|", 2)
    if len(parts) != 3:
        raise ValueError("--source must be system|output_root|lane")
    return parts[0], Path(parts[1]), parts[2]


def prompt_for(document_id: str, system: str, source_path: Path, raw_doc: dict[str, Any]) -> str:
    skeleton = {
        "document_id": document_id,
        "source_system": system,
        "semantic_entities": [
            {
                "semantic_type": "medication",
                "normalized_name": "",
                "raw_evidence_text": "",
                "source_raw_field": "medications[0].raw_line_text",
                "source_page_or_image": "1",
                "confidence": 0.0,
                "normalization_method": "direct_text_normalization",
                "evidence_supported": True,
            }
        ],
        "semantic_relations": [],
        "unsupported_inferences": [],
        "warnings": [],
        "metadata": {
            "schema_version": "semantic_rx_v1",
            "model_name": "qwen3:8b",
            "backend_name": "ollama",
            "source_stage1b_path": str(source_path),
            "processing_time_ms": 0.0,
            "timestamp": now(),
            "prompt_version": "stage1c_evidence_v1",
            "extra": {},
        },
    }
    return (
        "Return exactly one valid JSON object and no commentary. Perform semantic enrichment only; do not rewrite the raw input.\n"
        "Every semantic entity MUST preserve an exact raw_evidence_text quote from the supplied Stage 1B JSON and identify source_raw_field.\n"
        "Do not invent unsupported facts. Put uncertain or unsupported claims in unsupported_inferences instead of semantic_entities.\n"
        "Do not assign ontology identifiers. Preserve ambiguity. Use confidence 0.0-1.0.\n"
        "Allowed semantic_type values only: medication, dosage, frequency, duration, diagnosis, complaint, observation, vital, lab_result, procedure, advice, follow_up.\n"
        "Use evidence_supported=false when evidence is indirect or insufficient. Do not infer a diagnosis from treatment alone.\n"
        "For abbreviations such as OD/BD/TDS, retain the abbreviation as raw evidence and state abbreviation_expansion in normalization_method.\n"
        "Required JSON skeleton:\n"
        + json.dumps(skeleton, ensure_ascii=False, indent=2)
        + "\n\nSTAGE 1B RAW STRUCTURED JSON:\n"
        + json.dumps(raw_doc, ensure_ascii=False)[:24000]
    )


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sources = [parse_source(spec) for spec in args.source]
        self.smoke_docs = args.smoke_docs.split(",") if args.smoke_docs else []
        self.pid = os.getpid()
        self.current = ""

    def paths(self, system: str, doc_id: str) -> dict[str, Path]:
        return {
            "output": self.output_root / "semantic_outputs" / system / f"{doc_id}.json",
            "raw": self.output_root / "raw_responses" / system / f"{doc_id}.txt",
            "log": self.output_root / "logs" / system / f"{doc_id}.json",
            "failed": self.output_root / "failed_cases" / system / f"{doc_id}.json",
            "missing": self.output_root / "missing_inputs" / system / f"{doc_id}.json",
        }

    def input_docs(self, root: Path, lane: str) -> list[str]:
        directory = root / "raw_structured" / lane
        if self.smoke_docs:
            return self.smoke_docs
        return sorted(path.stem for path in directory.glob("*.json"))

    def counts(self) -> dict[str, dict[str, int]]:
        result = {}
        for system, root, lane in self.sources:
            docs = self.input_docs(root, lane)
            valid = len(list((self.output_root / "semantic_outputs" / system).glob("*.json")))
            failed = len(list((self.output_root / "failed_cases" / system).glob("*.json")))
            missing = len(list((self.output_root / "missing_inputs" / system).glob("*.json")))
            result[system] = {"valid": valid, "failed": failed, "missing_input": missing, "target": len(docs)}
        return result

    def write_progress(self) -> None:
        payload = {"generated": now(), "pid": self.pid, "output_root": str(self.output_root), "current": self.current, "systems": self.counts()}
        write_json(ROOT / "reports" / f"{self.args.report_stem}_progress.json", payload)
        lines = [f"# {self.args.report_title} Progress", "", f"Generated: {payload['generated']}", "", f"- PID: `{self.pid}`", f"- Output root: `{self.output_root}`", f"- Current: `{self.current}`", ""]
        for system, counts in payload["systems"].items():
            lines.append(f"- `{system}`: valid={counts['valid']} failed={counts['failed']} missing_input={counts['missing_input']} target={counts['target']}")
        write_text(ROOT / "reports" / f"{self.args.report_stem}_progress.md", "\n".join(lines) + "\n")

    def run_one(self, system: str, source_path: Path, doc_id: str) -> None:
        paths = self.paths(system, doc_id)
        self.current = f"{system}:{doc_id}"
        if self.args.resume and paths["output"].exists():
            return
        if not source_path.exists():
            write_json(paths["missing"], {"document_id": doc_id, "system": system, "reason": "valid_stage1b_input_missing", "source_path": str(source_path)})
            self.write_progress()
            return
        raw_doc = json.loads(source_path.read_text(encoding="utf-8"))
        prompt = prompt_for(doc_id, system, source_path, raw_doc)
        self.write_progress()
        started = time.time()
        error = ""
        response_data: dict[str, Any] = {}
        content = ""
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": self.args.model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False, "options": {"temperature": 0.0, "num_ctx": 16384, "num_predict": 4096}},
                timeout=self.args.timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            content = response_data.get("message", {}).get("content", "")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        runtime = round(time.time() - started, 3)
        write_text(paths["raw"], content or json.dumps(response_data, ensure_ascii=False))
        parsed = extract_json(content)
        valid = False
        validation_error = ""
        if parsed is not None:
            try:
                parsed["document_id"] = doc_id
                parsed["source_system"] = system
                metadata = parsed.setdefault("metadata", {})
                metadata.update({"schema_version": "semantic_rx_v1", "model_name": self.args.model, "backend_name": "ollama", "source_stage1b_path": str(source_path), "processing_time_ms": runtime * 1000.0, "timestamp": now(), "prompt_version": "stage1c_evidence_v1"})
                validated = SemanticExtractionDoc(**parsed)
                write_json(paths["output"], validated.model_dump())
                valid = True
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
        if not valid:
            write_json(paths["failed"], {"document_id": doc_id, "system": system, "error": error or validation_error or "semantic_json_parse_failed", "parsed": parsed, "raw_response_path": str(paths["raw"])})
        log = {
            "timestamp": now(), "document_id": doc_id, "system": system, "model": self.args.model,
            "source_stage1b_path": str(source_path), "runtime_seconds": runtime,
            "parse_success": parsed is not None, "schema_valid": valid,
            "status": "success" if valid else "failed", "error": "" if valid else (error or validation_error or "semantic_json_parse_failed"),
            "semantic_output_path": str(paths["output"]) if valid else "", "raw_response_path": str(paths["raw"]),
            "usage": {"prompt_eval_count": response_data.get("prompt_eval_count"), "eval_count": response_data.get("eval_count")},
        }
        write_json(paths["log"], log)
        self.write_progress()

    def run(self) -> None:
        for system, root, lane in self.sources:
            for doc_id in self.input_docs(root, lane):
                source_path = root / "raw_structured" / lane / f"{doc_id}.json"
                self.run_one(system, source_path, doc_id)
        self.current = ""
        self.write_progress()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="system|output_root|lane")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--smoke-docs", default="")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-stem", required=True)
    parser.add_argument("--report-title", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    Runner(parse_args()).run()

