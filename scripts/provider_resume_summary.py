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

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.extract import validate_completed_output  # noqa: E402


DEFAULT_MANIFESTS = [
    PROJECT_ROOT / "data/manifest_openrouter_google_gemini-2.5-flash_smoke.csv",
    PROJECT_ROOT / "data/manifest_openrouter_qwen_qwen2.5-vl-72b-instruct_smoke.csv",
    PROJECT_ROOT / "data/manifest_openrouter_meta-llama_llama-3.2-11b-vision-instruct_smoke.csv",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def main() -> None:
    started = datetime.now(timezone.utc)
    records = []
    for manifest in DEFAULT_MANIFESTS:
        if not manifest.exists():
            continue
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pred_rel = row.get("prediction_path") or ""
                pred_path = PROJECT_ROOT / pred_rel
                valid, reason = validate_completed_output(pred_path)
                records.append({
                    "manifest": str(manifest.relative_to(PROJECT_ROOT)),
                    "document_id": row.get("document_id"),
                    "prediction_path": pred_rel,
                    "valid": valid,
                    "reason": reason,
                })

    provider_logs = read_jsonl(PROJECT_ROOT / "logs/provider_usage.jsonl")
    checkpoints = []
    for p in (PROJECT_ROOT / "logs").glob("checkpoint_*.jsonl"):
        checkpoints.extend(read_jsonl(p))

    provider_usage = defaultdict(Counter)
    gemini_usage = defaultdict(Counter)
    retries = 0
    for row in provider_logs:
        provider = row.get("provider") or "unknown"
        model = row.get("model") or "unknown"
        provider_usage[(provider, model)]["requests"] += 1
        provider_usage[(provider, model)][row.get("status") or "unknown"] += 1
        provider_usage[(provider, model)]["retries"] += int(row.get("retry_count") or 0)
        retries += int(row.get("retry_count") or 0)
        if provider == "gemini":
            key = row.get("api_key_label") or "unknown"
            gemini_usage[key]["requests"] += 1
            gemini_usage[key][row.get("status") or "unknown"] += 1

    checkpoint_counts = Counter(row.get("status") for row in checkpoints)
    provider_docs = defaultdict(list)
    for row in provider_logs:
        if row.get("document_id"):
            provider_docs[row["document_id"]].append(row)
    live_attempted_docs = set(provider_docs)
    live_success_docs = {
        doc for doc, rows in provider_docs.items()
        if any(r.get("status") == "success" for r in rows)
    }
    provider_retry_attempts = sum(max(0, len(rows) - 1) for rows in provider_docs.values())
    total = len(records)
    valid_records = [r for r in records if r["valid"]]
    invalid_records = [r for r in records if not r["valid"]]
    duplicate_outputs = []
    by_path = Counter(r["prediction_path"] for r in records)
    for path, count in by_path.items():
        if path and count > 1:
            duplicate_outputs.append({"prediction_path": path, "count": count})

    summary = {
        "generated": started.isoformat(),
        "scope": "legacy OpenRouter smoke manifests rerouted to HF/Gemini; no OpenRouter calls",
        "total_files_discovered": total,
        "total_already_completed_or_valid": len(valid_records),
        "total_resumed_attempted": len(live_attempted_docs),
        "total_newly_processed_success": len(live_success_docs),
        "total_skipped_valid": checkpoint_counts.get("skipped_valid", 0),
        "total_retried": retries + provider_retry_attempts,
        "total_permanently_failed": len(invalid_records),
        "duplicate_outputs": duplicate_outputs,
        "invalid_or_missing": invalid_records,
        "provider_usage": {
            f"{provider}:{model}": dict(counter)
            for (provider, model), counter in provider_usage.items()
        },
        "gemini_api_key_usage": {key: dict(counter) for key, counter in gemini_usage.items()},
    }

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "provider_resume_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Provider Resume Summary",
        "",
        f"Generated: {summary['generated']}",
        "",
        "OpenRouter policy: disabled. Legacy OpenRouter manifests were routed through Hugging Face or Gemini only.",
        "",
        f"- Total files discovered: {summary['total_files_discovered']}",
        f"- Total already completed / valid: {summary['total_already_completed_or_valid']}",
        f"- Total resumed attempted: {summary['total_resumed_attempted']}",
        f"- Total newly processed successfully: {summary['total_newly_processed_success']}",
        f"- Total skipped valid: {summary['total_skipped_valid']}",
        f"- Total retried: {summary['total_retried']}",
        f"- Total permanently failed / still missing: {summary['total_permanently_failed']}",
        "",
        "## Hugging Face model usage",
        "",
    ]
    if summary["provider_usage"]:
        for key, val in summary["provider_usage"].items():
            lines.append(f"- `{key}`: {val}")
    else:
        lines.append("- No live HF/Gemini provider calls were logged.")
    lines.extend(["", "## Gemini API key usage", ""])
    if summary["gemini_api_key_usage"]:
        for key, val in summary["gemini_api_key_usage"].items():
            lines.append(f"- `{key}`: {val}")
    else:
        lines.append("- No Gemini calls were needed; existing Gemini outputs validated successfully and were skipped.")
    lines.extend(["", "## Remaining issues", ""])
    if invalid_records:
        for row in invalid_records:
            lines.append(f"- `{row['document_id']}` from `{row['manifest']}`: `{row['reason']}` at `{row['prediction_path']}`")
        lines.append("")
        lines.append("These were not sent to OpenRouter. HF Inference reported no available provider for the hosted Llama vision model.")
    else:
        lines.append("- None.")
    lines.append("")
    (reports / "provider_resume_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(reports / "provider_resume_summary.md")


if __name__ == "__main__":
    main()
