import argparse
import csv
import io
import json
import re
import os
import sys
import yaml
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Any, Optional


# Force the project root directory onto the python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import dotenv
dotenv.load_dotenv(PROJECT_ROOT / ".env")
dotenv.load_dotenv(PROJECT_ROOT.parent / ".env")

from src.schemas.raw_extraction import CanonicalRawDoc, Metadata
from src.adapters.backend_adapter_qwen import QwenVLBackendAdapter
from src.adapters.backend_adapter_openrouter import OpenRouterBackendAdapter
from src.adapters.backend_adapter_openai_compatible_vlm import OpenAICompatibleVLMBackendAdapter
from src.utils.rate_limiter import init_global_limiter, get_global_limiter



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ExtractCLI")

def csv_value(value: Any) -> Any:
    return "" if value is None else value

def estimate_compressed_image_size_kb(images: list[Image.Image], max_image_dim: int, jpeg_quality: int) -> float:
    total_bytes = 0
    for image in images:
        encoded_image = image
        if max(encoded_image.size) > max_image_dim:
            w, h = encoded_image.size
            scale = max_image_dim / max(w, h)
            encoded_image = encoded_image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        encoded_image.save(buffered, format="JPEG", quality=jpeg_quality)
        total_bytes += len(buffered.getvalue())

    return total_bytes / 1024.0

def get_internal_qwen3_usage_tokens(response: dict) -> tuple[int, int, int]:
    usage = response.get("usage") or {}
    raw_usage = (response.get("raw_response") or {}).get("usage") or {}

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or raw_usage.get("total_tokens") or 0

    if not total_tokens and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    return int(prompt_tokens or 0), int(completion_tokens or 0), int(total_tokens or 0)

def record_internal_qwen3_rate_usage(
    document_id: str,
    response: dict,
    estimated_input_tokens: Optional[int],
    estimated_output_tokens: Optional[int],
    reason: str,
) -> tuple[int, int, int, bool]:
    api_prompt_tokens, api_completion_tokens, api_total_tokens = get_internal_qwen3_usage_tokens(response)
    limiter = get_global_limiter()

    if api_total_tokens > 0:
        limiter.record_usage(
            document_id=document_id,
            prompt_tokens=api_prompt_tokens,
            completion_tokens=api_completion_tokens,
            estimated=False,
            reason=reason,
        )
        return api_prompt_tokens, api_completion_tokens, api_total_tokens, True

    if estimated_input_tokens is not None and estimated_output_tokens is not None:
        limiter.record_usage(
            document_id=document_id,
            prompt_tokens=estimated_input_tokens,
            completion_tokens=estimated_output_tokens,
            estimated=True,
            reason=f"{reason}_estimated_fallback",
        )

    return api_prompt_tokens, api_completion_tokens, api_total_tokens, False

def get_cumulative_cost(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    total = 0.0
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cost_str = row.get("estimated_cost", "0.0")
                try:
                    total += float(cost_str)
                except ValueError:
                    pass
    except Exception as e:
        logger.warning(f"Error reading cumulative cost from {log_path}: {e}")
    return total

def log_openrouter_usage(
    log_path: Path,
    document_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    estimated_cost: float,
    latency_ms: float,
    status: str
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    is_empty = False
    if file_exists:
        is_empty = log_path.stat().st_size == 0

    fieldnames = ["timestamp", "document_id", "model", "input_tokens", "output_tokens", "total_tokens", "estimated_cost", "latency_ms", "status"]
    if file_exists and not is_empty:
        try:
            with open(log_path, "r", encoding="utf-8", newline="") as existing_f:
                reader = csv.DictReader(existing_f)
                existing_fieldnames = reader.fieldnames or []
                if any(name not in existing_fieldnames for name in fieldnames):
                    existing_rows = list(reader)
                    with open(log_path, "w", encoding="utf-8", newline="") as migrated_f:
                        writer = csv.DictWriter(migrated_f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in existing_rows:
                            row.pop(None, None)
                            writer.writerow({name: row.get(name, "") for name in fieldnames})
                    file_exists = True
                    is_empty = False
                    logger.info(f"Migrated internal Qwen3 usage log to include rate-limit columns: {log_path}")
        except Exception as e:
            logger.warning(f"Could not migrate internal Qwen3 usage log header: {e}")

    try:
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or is_empty:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "document_id": document_id,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": estimated_cost,
                "latency_ms": latency_ms,
                "status": status
            })
    except Exception as e:
        logger.error(f"Failed to log OpenRouter usage: {e}")

def log_internal_qwen3_usage(
    log_path: Path,
    document_id: str,
    model: str,
    num_images: int,
    max_tokens: int,
    latency_ms: float,
    status: str,
    error_type: Optional[str] = None,
    validation_status: Optional[str] = None,
    notes: Optional[str] = None,
    estimated_input_tokens: Optional[int] = None,
    estimated_output_tokens: Optional[int] = None,
    total_estimated_tokens: Optional[int] = None,
    api_reported_prompt_tokens: Optional[int] = None,
    api_reported_completion_tokens: Optional[int] = None,
    api_reported_total_tokens: Optional[int] = None,
    rolling_tokens_before_request: Optional[int] = None,
    sleep_seconds_before_request: Optional[float] = None,
    retry_after_rate_limit: Optional[bool] = None,
    rate_limit_reason: Optional[str] = None,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    is_empty = False
    if file_exists:
        is_empty = log_path.stat().st_size == 0

    fieldnames = [
        "timestamp", "document_id", "model", "num_images", "max_tokens",
        "latency_ms", "status", "error_type", "validation_status", "notes",
        "estimated_input_tokens", "estimated_output_tokens", "total_estimated_tokens",
        "api_reported_prompt_tokens", "api_reported_completion_tokens", "api_reported_total_tokens",
        "rolling_tokens_before_request", "sleep_seconds_before_request",
        "retry_after_rate_limit", "rate_limit_reason"
    ]
    if file_exists and not is_empty:
        try:
            with open(log_path, "r", encoding="utf-8", newline="") as existing_f:
                reader = csv.DictReader(existing_f)
                existing_fieldnames = reader.fieldnames or []
                if any(name not in existing_fieldnames for name in fieldnames):
                    existing_rows = list(reader)
                    with open(log_path, "w", encoding="utf-8", newline="") as migrated_f:
                        writer = csv.DictWriter(migrated_f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in existing_rows:
                            row.pop(None, None)
                            writer.writerow({name: row.get(name, "") for name in fieldnames})
                    file_exists = True
                    is_empty = False
                    logger.info(f"Migrated internal Qwen3 usage log to include rate-limit columns: {log_path}")
        except Exception as e:
            logger.warning(f"Could not migrate internal Qwen3 usage log header: {e}")

    try:
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or is_empty:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "document_id": document_id,
                "model": model,
                "num_images": num_images,
                "max_tokens": max_tokens,
                "latency_ms": latency_ms,
                "status": status,
                "error_type": error_type or "",
                "validation_status": validation_status or "",
                "notes": notes or "",
                "estimated_input_tokens": csv_value(estimated_input_tokens),
                "estimated_output_tokens": csv_value(estimated_output_tokens),
                "total_estimated_tokens": csv_value(total_estimated_tokens),
                "api_reported_prompt_tokens": csv_value(api_reported_prompt_tokens),
                "api_reported_completion_tokens": csv_value(api_reported_completion_tokens),
                "api_reported_total_tokens": csv_value(api_reported_total_tokens),
                "rolling_tokens_before_request": csv_value(rolling_tokens_before_request),
                "sleep_seconds_before_request": csv_value(sleep_seconds_before_request),
                "retry_after_rate_limit": csv_value(retry_after_rate_limit),
                "rate_limit_reason": csv_value(rate_limit_reason)
            })
    except Exception as e:
        logger.error(f"Failed to log internal Qwen3 usage: {e}")



def clean_and_repair_json(raw_text: str) -> str:
    """
    Conservative JSON repair helper:
    - Strips markdown fences
    - Extracts the outermost JSON block
    - Removes trailing commas in objects and arrays
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1. Strip markdown fences
    if text.startswith("```"):
        # Match ```json or ``` at beginning
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    # 2. Locate the outermost curly braces { ... }
    start_idx = text.find("{")
    end_idx = text.rfind("}")

    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        return text  # Return as is, let standard json parser throw exception

    json_block = text[start_idx:end_idx + 1]

    # 3. Clean trailing commas (e.g. [1, 2,] -> [1, 2] or {"a": 1,} -> {"a": 1})
    # Remove trailing commas before matching closing bracket/brace
    # Avoid complex regex that could break nested JSON, use a conservative replacement:
    json_block = re.sub(r",\s*([\]}])", r"\1", json_block)

    return json_block

def prune_placeholder_items(parsed_dict: dict, warnings: Optional[list] = None) -> dict:
    """
    Remove empty placeholder objects copied verbatim from prompt templates.
    """
    if not isinstance(parsed_dict, dict):
        return parsed_dict

    if warnings is None:
        warnings = []

    # Helper to check if a value is effectively null or a placeholder string
    def is_empty(val):
        if val is None:
            return True
        if isinstance(val, str):
            val_stripped = val.strip().lower()
            return val_stripped in ("", "null", "none", "[]", "{}")
        return False

    # 1. Prune Entity Items (complaints_or_diagnosis, observations, procedures, advice, allergy_mentions, other_notes)
    entity_keys = ["complaints_or_diagnosis", "observations", "procedures", "advice", "allergy_mentions", "other_notes"]
    for k in entity_keys:
        if k in parsed_dict and isinstance(parsed_dict[k], list):
            cleaned_list = []
            orig_len = len(parsed_dict[k])
            for idx, item in enumerate(parsed_dict[k]):
                if isinstance(item, dict):
                    # Check if raw_text is valid
                    if not is_empty(item.get("raw_text")):
                        cleaned_list.append(item)
                    elif not is_empty(item.get("evidence_text")):
                        # Fallback: if raw_text is empty but evidence_text is populated, use evidence_text
                        item["raw_text"] = item["evidence_text"]
                        cleaned_list.append(item)
                        warnings.append(f"{k}[{idx}]: raw_text was empty; populated from evidence_text")
                    else:
                        warnings.append(f"{k}[{idx}]: pruned empty RawEntityItem")
                elif isinstance(item, str) and not is_empty(item):
                    # Convert raw string to RawEntityItem dictionary
                    cleaned_list.append({
                        "raw_text": item,
                        "evidence_text": item,
                        "page_number": 1
                    })
                    warnings.append(f"{k}[{idx}]: converted raw string to RawEntityItem dict")
                else:
                    warnings.append(f"{k}[{idx}]: pruned invalid/empty element")
            parsed_dict[k] = cleaned_list

    # 2. Prune Medications
    if "medications" in parsed_dict and isinstance(parsed_dict["medications"], list):
        cleaned_meds = []
        for idx, item in enumerate(parsed_dict["medications"]):
            if isinstance(item, dict):
                # Check if raw_line_text or raw_name is present
                raw_line = item.get("raw_line_text")
                raw_name = item.get("raw_name")

                # If raw_line_text is empty but raw_name is present, use raw_name
                if is_empty(raw_line) and not is_empty(raw_name):
                    item["raw_line_text"] = raw_name
                    raw_line = raw_name
                    warnings.append(f"medications[{idx}]: raw_line_text was empty; populated from raw_name")

                if not is_empty(raw_line):
                    cleaned_meds.append(item)
                else:
                    warnings.append(f"medications[{idx}]: pruned empty RawMedicationItem")
            elif isinstance(item, str) and not is_empty(item):
                cleaned_meds.append({
                    "raw_line_text": item,
                    "raw_name": item,
                    "evidence_text": item,
                    "page_number": 1
                })
                warnings.append(f"medications[{idx}]: converted raw string to RawMedicationItem dict")
            else:
                warnings.append(f"medications[{idx}]: pruned invalid/empty medication element")
        parsed_dict["medications"] = cleaned_meds

    # 3. Prune Lab Observations
    if "lab_observations" in parsed_dict and isinstance(parsed_dict["lab_observations"], list):
        cleaned_labs = []
        for idx, item in enumerate(parsed_dict["lab_observations"]):
            if isinstance(item, dict):
                test_name = item.get("test_name")
                result = item.get("result")
                # Need at least test_name or result to be valid
                if not is_empty(test_name) or not is_empty(result):
                    if is_empty(item.get("raw_line_text")):
                        # Construct a basic raw_line_text if missing
                        parts = [x for x in [test_name, result, item.get("unit")] if x]
                        item["raw_line_text"] = " ".join(parts) if parts else "Lab Observation"
                        warnings.append(f"lab_observations[{idx}]: raw_line_text was empty; constructed from parts")
                    cleaned_labs.append(item)
                else:
                    warnings.append(f"lab_observations[{idx}]: pruned empty RawLabObservationItem")
            elif isinstance(item, str) and not is_empty(item):
                cleaned_labs.append({
                    "raw_line_text": item,
                    "test_name": item,
                    "evidence_text": item,
                    "page_number": 1
                })
                warnings.append(f"lab_observations[{idx}]: converted raw string to RawLabObservationItem dict")
            else:
                warnings.append(f"lab_observations[{idx}]: pruned invalid/empty lab element")
        parsed_dict["lab_observations"] = cleaned_labs

    # 4. Clean Follow-up
    if "follow_up" in parsed_dict:
        fup = parsed_dict["follow_up"]
        if isinstance(fup, dict):
            if is_empty(fup.get("raw_text")) and is_empty(fup.get("date")) and is_empty(fup.get("review_after")):
                parsed_dict["follow_up"] = None
                warnings.append("follow_up: pruned empty RawFollowUp")
        elif is_empty(fup):
            parsed_dict["follow_up"] = None
            warnings.append("follow_up: pruned empty follow_up element")

    return parsed_dict

def select_prompt_template(row: dict, oracle_mode: bool, project_root: Path) -> str:
    source_type = None
    speciality = None

    if oracle_mode:
        # Load Ground Truth if available to get source_type/speciality
        doc_id = row["document_id"]
        gt_path = None

        # 1. Try manifest_canonical.csv lookup
        manifest_canonical = project_root / "data" / "manifest_canonical.csv"
        if manifest_canonical.exists():
            try:
                import csv
                with open(manifest_canonical, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if r["document_id"] == doc_id:
                            gt_path = project_root / r["gt_path"]
                            break
            except Exception as e:
                logger.warning(f"Error reading manifest_canonical.csv: {e}")

        # 2. Heuristics fallback
        if not gt_path:
            possible_paths = [
                project_root / "raw_ground_truths" / f"{doc_id}.json",
            ]
            if "_" in doc_id:
                parts = doc_id.split("_")
                possible_paths.append(project_root / "raw_ground_truths" / parts[0] / f"{doc_id}.json")
            for p in possible_paths:
                if p.exists():
                    gt_path = p
                    break

        # Load GT source_type
        if gt_path and gt_path.exists():
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                source_type = gt_data.get("document_metadata", {}).get("source_type")
            except Exception as e:
                logger.warning(f"Failed to load GT file {gt_path} for oracle prompt selection: {e}")

        # Load speciality from manifest row
        speciality = row.get("speciality")
    else:
        # Production deployment: use manifest metadata or filename/heuristics
        speciality = row.get("speciality")
        source_type = row.get("source_type")

        # Heuristics based on document_id and image_path
        doc_id = row.get("document_id", "").lower()
        image_path = row.get("image_path", "").lower()

        if not source_type:
            if "lab" in doc_id or "lab" in image_path or "report" in doc_id or "report" in image_path:
                source_type = "lab_report"
            elif "discharge" in doc_id or "discharge" in image_path:
                source_type = "discharge_card"
            elif "radio" in doc_id or "radio" in image_path or "xray" in doc_id or "xray" in image_path or "scan" in doc_id or "scan" in image_path:
                if "image" in doc_id or "image" in image_path or "film" in doc_id or "film" in image_path:
                    source_type = "radiology_image"
                else:
                    source_type = "radiology_report"

        if not speciality:
            if "ophthal" in doc_id or "ophthal" in image_path or "eye" in doc_id or "eye" in image_path:
                speciality = "ophthalmology"

    # Map variables to corresponding prompt templates
    if speciality == "ophthalmology":
        return "ophthalmology_prompt_v2"
    elif source_type == "lab_report":
        return "lab_report_prompt_v2"
    elif source_type == "radiology_image":
        return "radiology_image_prompt_v2"
    elif source_type in ["radiology_report", "diagnostic_report", "diagnostic_reports"]:
        return "radiology_report_prompt_v2"
    elif source_type == "discharge_card":
        return "discharge_card_prompt_v2"
    else:
        return "prescription_prompt_v2"

async def extract_document(
    row: dict,
    adapter: Any,
    backend_config: dict,
    prompt_config: dict,
    output_dir: Path,
    failed_dir: Path,
    overwrite: bool,
    dry_run: bool,
    oracle_mode: bool = False,
    raw_responses_dir: Optional[Path] = None
) -> bool:
    doc_id = row["document_id"]
    image_paths_str = row["image_path"]

    # Parse semicolon-separated images in exact order
    rel_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    abs_paths = [PROJECT_ROOT / p for p in rel_paths]

    # Determine save path
    save_path = output_dir / f"{doc_id}.json"

    if save_path.exists() and not overwrite:
        logger.info(f"Document {doc_id} already successfully extracted. Skipping.")
        return True

    if dry_run:
        logger.info(f"[DRY-RUN] Would extract document {doc_id} with images: {rel_paths}")
        return True

    logger.info(f"Extracting document {doc_id} with {len(abs_paths)} pages...")

    # Load images
    pil_images = []
    for p in abs_paths:
        if not p.exists():
            logger.error(f"Image path does not exist: {p}")
            return False
        try:
            pil_images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            logger.error(f"Failed to open image {p}: {e}")
            return False

    # Dynamic prompt selection
    prompt_key = select_prompt_template(row, oracle_mode, PROJECT_ROOT)
    logger.info(f"Selected prompt template: '{prompt_key}' for document '{doc_id}' (oracle_mode={oracle_mode})")

    template = prompt_config.get(prompt_key)
    if not template:
        # Fallback to prescription template if v2 specific not found
        template = prompt_config.get("prescription_prompt_v2")
    if not template:
        # Final fallback to raw_extraction legacy key if any
        template = prompt_config.get("raw_extraction", {})

    raw_prompt_text = template.get("user_prompt", "")
    user_prompt = raw_prompt_text.replace("{{document_id}}", doc_id)
    system_prompt = template.get("system_prompt", "")

    # Combined prompt with system instructions
    combined_prompt = f"{system_prompt}\n\nUser request:\n{user_prompt}"


    is_openrouter = (backend_config.get("backend_name") == "openrouter")
    log_path = PROJECT_ROOT / "logs" / "openrouter_usage.csv"

    # Execute backend adapter
    try:
        # Determine max_tokens override dynamically based on document name
        if backend_config.get("backend_name") == "internal_qwen3_27b_vlm":
            max_tokens_val = backend_config.get("max_tokens", 100000)
        else:
            max_tokens_val = backend_config.get("max_tokens", 4096)
            if doc_id in ["p1", "p2"]:
                max_tokens_val = 4000
            elif doc_id in ["p45_1", "p45_3"]:
                max_tokens_val = 6000
            elif doc_id == "p45_4":
                max_tokens_val = 8192


        # Pass backend decoding params
        decoding_params = {
            "temperature": backend_config.get("temperature", 0.0),
            "max_tokens": max_tokens_val,
            "top_p": backend_config.get("top_p", 0.9)
        }

        # Rate limiting for internal qwen3-27b API
        sleep_seconds_before_request = 0.0
        estimated_input_tokens = None
        estimated_output_tokens = None
        total_estimated_tokens = None
        rolling_tokens_before_request = None

        if backend_config.get("backend_name") == "internal_qwen3_27b_vlm":
            limiter = get_global_limiter()

            compressed_image_size_kb = estimate_compressed_image_size_kb(
                pil_images,
                max_image_dim=backend_config.get("max_image_dim", 1024),
                jpeg_quality=backend_config.get("jpeg_quality", 85),
            )
            estimated_components = limiter.estimate_token_components(
                document_id=doc_id,
                max_tokens=decoding_params["max_tokens"],
                num_images=len(abs_paths),
                compressed_image_size_kb=compressed_image_size_kb,
                prompt_length_chars=len(combined_prompt),
                reserve_full_output_budget=True,
            )
            estimated_input_tokens = estimated_components["input_tokens"]
            estimated_output_tokens = estimated_components["output_budget_tokens"]
            total_estimated_tokens = estimated_components["total_tokens"]

            # Check rate limits and wait if needed
            rolling_tokens_before_request = limiter.get_rolling_tokens()
            logger.info(f"Rate limit check before {doc_id}: {limiter.format_summary_for_log()}")

            sleep_seconds_before_request = limiter.wait_if_needed(
                document_id=doc_id,
                estimated_tokens=total_estimated_tokens
            )

        # Run async inference
        response = await adapter.run(
            prompt=combined_prompt,
            image=pil_images if len(pil_images) > 1 else pil_images[0],
            **decoding_params
        )
    except Exception as e:
        logger.error(f"Inference threw an exception for {doc_id}: {e}")
        # Save exception details
        save_failed_outputs(doc_id, failed_dir, f"Inference exception: {str(e)}", {"exception": str(e)}, is_openrouter=is_openrouter)
        if is_openrouter:
            log_openrouter_usage(
                log_path=log_path,
                document_id=doc_id,
                model=backend_config.get("model_name"),
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost=0.0,
                latency_ms=0.0,
                status="failed"
            )
        if backend_config.get("backend_name") == "internal_qwen3_27b_vlm":
            log_internal_qwen3_usage(
                log_path=PROJECT_ROOT / "logs" / "internal_qwen3_usage.csv",
                document_id=doc_id,
                model=backend_config.get("model_name"),
                num_images=len(abs_paths),
                max_tokens=decoding_params["max_tokens"],
                latency_ms=0.0,
                status="failed",
                error_type="exception",
                validation_status="unattempted",
                notes=str(e)[:200],
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                total_estimated_tokens=total_estimated_tokens,
                rolling_tokens_before_request=rolling_tokens_before_request,
                sleep_seconds_before_request=sleep_seconds_before_request,
                retry_after_rate_limit=False,
                rate_limit_reason=None
            )
        return False

    if "error" in response:
        logger.error(f"Inference backend returned error for {doc_id}: {response['error']}")
        save_failed_outputs(doc_id, failed_dir, response.get("content") or "", {"backend_error": response["error"]}, is_openrouter=is_openrouter)
        if is_openrouter:
            log_openrouter_usage(
                log_path=log_path,
                document_id=doc_id,
                model=backend_config.get("model_name"),
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost=0.0,
                latency_ms=response.get("processing_time_ms", 0.0),
                status="failed"
            )
        if backend_config.get("backend_name") == "internal_qwen3_27b_vlm":
            # Check if error was rate-limit related
            is_rate_limit_error = "429" in str(response.get("error", "")) or "rate" in str(response.get("error", "")).lower()
            limiter = get_global_limiter() if backend_config.get("backend_name") == "internal_qwen3_27b_vlm" else None

            log_internal_qwen3_usage(
                log_path=PROJECT_ROOT / "logs" / "internal_qwen3_usage.csv",
                document_id=doc_id,
                model=backend_config.get("model_name"),
                num_images=len(abs_paths),
                max_tokens=decoding_params["max_tokens"],
                latency_ms=response.get("processing_time_ms", 0.0),
                status="failed",
                error_type="backend_error",
                validation_status="unattempted",
                notes=response["error"][:200],
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                total_estimated_tokens=total_estimated_tokens,
                rolling_tokens_before_request=rolling_tokens_before_request,
                sleep_seconds_before_request=sleep_seconds_before_request,
                retry_after_rate_limit=is_rate_limit_error,
                rate_limit_reason="429_or_rate_error_in_response" if is_rate_limit_error else None
            )
        return False

    warnings = []
    coercions = []

    # Conservative json repair
    raw_content = response.get("content") or ""
    repaired_json = clean_and_repair_json(raw_content)
    if raw_content.strip() != repaired_json.strip():

        warnings.append("JSON raw output required markdown or trailing comma cleanup")

    # Parse and validate
    parsed_dict = None
    parse_error = None
    try:
        parsed_dict = json.loads(repaired_json)
    except Exception as je:
        parse_error = f"JSON parse error: {str(je)}"
        logger.error(f"Failed to parse JSON for {doc_id}: {je}")

    if parsed_dict is None:
        save_failed_outputs(doc_id, failed_dir, raw_content, {"parse_error": parse_error, "repaired_text": repaired_json}, is_openrouter=is_openrouter)
        if is_openrouter:
            usage = response.get("usage", {})
            log_openrouter_usage(
                log_path=log_path,
                document_id=doc_id,
                model=backend_config.get("model_name"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                estimated_cost=usage.get("estimated_cost", 0.0),
                latency_ms=response.get("processing_time_ms", 0.0),
                status="failed"
            )
        if backend_config.get("backend_name") == "internal_qwen3_27b_vlm":
            api_prompt_tokens, api_completion_tokens, api_total_tokens, actual_usage_available = record_internal_qwen3_rate_usage(
                document_id=doc_id,
                response=response,
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                reason="json_parse_error",
            )
            log_internal_qwen3_usage(
                log_path=PROJECT_ROOT / "logs" / "internal_qwen3_usage.csv",
                document_id=doc_id,
                model=backend_config.get("model_name"),
                num_images=len(abs_paths),
                max_tokens=decoding_params["max_tokens"],
                latency_ms=response.get("processing_time_ms", 0.0),
                status="failed",
                error_type="json_parse_error",
                validation_status="invalid",
                notes=f"{parse_error[:160]}; Actual usage: {actual_usage_available}",
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                total_estimated_tokens=total_estimated_tokens,
                api_reported_prompt_tokens=api_prompt_tokens if api_prompt_tokens > 0 else None,
                api_reported_completion_tokens=api_completion_tokens if api_completion_tokens > 0 else None,
                api_reported_total_tokens=api_total_tokens if api_total_tokens > 0 else None,
                rolling_tokens_before_request=rolling_tokens_before_request,
                sleep_seconds_before_request=sleep_seconds_before_request,
                retry_after_rate_limit=False,
                rate_limit_reason=None
            )
        return False

    # Prune empty placeholder items
    parsed_dict = prune_placeholder_items(parsed_dict, warnings=warnings)

    # Coerce fields in patient_information and encounter_information to string if they are numeric
    if isinstance(parsed_dict, dict) and "patient_information" in parsed_dict and isinstance(parsed_dict["patient_information"], dict):
        pinfo = parsed_dict["patient_information"]
        for k in ["name", "age", "gender", "address", "phone", "patient_identifier", "abha_id"]:
            if k in pinfo and pinfo[k] is not None and not isinstance(pinfo[k], str):
                old_val = pinfo[k]
                new_val = str(pinfo[k])
                pinfo[k] = new_val
                coercions.append(f"patient_information.{k}: coerced {type(old_val).__name__} ({old_val}) to string ({new_val})")

    if isinstance(parsed_dict, dict) and "encounter_information" in parsed_dict and isinstance(parsed_dict["encounter_information"], dict):
        einfo = parsed_dict["encounter_information"]
        for k in ["date", "department", "hospital_name", "doctor_name", "visit_type", "fees", "room_or_queue_no"]:
            if k in einfo and einfo[k] is not None and not isinstance(einfo[k], str):
                old_val = einfo[k]
                new_val = str(einfo[k])
                einfo[k] = new_val
                coercions.append(f"encounter_information.{k}: coerced {type(old_val).__name__} ({old_val}) to string ({new_val})")

    if coercions:
        logger.info(f"[{doc_id}] Type coercions applied: {coercions}")
    if warnings:
        logger.info(f"[{doc_id}] Validation warnings: {warnings}")

    # Validate against Pydantic schema
    try:
        # Check and populate metadata
        metadata_dict = {
            "backend_name": backend_config.get("backend_name", "qwen25_vl_7b"),
            "model_name": backend_config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
            "model_version": backend_config.get("model_version", "2.5"),
            "prompt_version": "2.0",
            "schema_version": "raw_rx_v2",
            "processing_time_ms": response.get("processing_time_ms", 0.0),
            "decoding_parameters": response.get("decoding_parameters", {}),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "document_type": prompt_key.replace("_prompt_v2", ""),
            "validation_warnings": warnings,
            "type_coercions": coercions,
            "pages": [
                {"page_number": idx + 1, "image_path": rel_path}
                for idx, rel_path in enumerate(rel_paths)
            ]
        }

        # Merge or overwrite parsed doc metadata
        parsed_dict["document_id"] = doc_id
        parsed_dict["schema_version"] = "raw_rx_v2"

        # Enforce page_number mapping order based on lists
        # We can add helper logic to ensure each item's page_number is checked
        def clamp_page_numbers(items_list):
            for item in items_list:
                if isinstance(item, dict):
                    pg = item.get("page_number", 1)
                    # Clamp page numbers to available images
                    if pg < 1:
                        item["page_number"] = 1
                    elif pg > len(pil_images):
                        item["page_number"] = len(pil_images)

        for k in ["complaints_or_diagnosis", "observations", "medications", "procedures", "advice", "allergy_mentions", "other_notes", "lab_observations"]:
            if k in parsed_dict and isinstance(parsed_dict[k], list):
                clamp_page_numbers(parsed_dict[k])

        # Validate Pydantic
        canonical_doc = CanonicalRawDoc(**parsed_dict)

        # Add metadata block
        canonical_doc.metadata = Metadata(**metadata_dict)

        # Save successful output JSON
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as out_f:
            out_f.write(canonical_doc.model_dump_json(indent=2))

        if raw_responses_dir:
            raw_responses_dir.mkdir(parents=True, exist_ok=True)
            with open(raw_responses_dir / f"{doc_id}.txt", "w", encoding="utf-8") as rf:
                rf.write(raw_content)

        if is_openrouter:
            usage = response.get("usage", {})
            log_openrouter_usage(
                log_path=log_path,
                document_id=doc_id,
                model=backend_config.get("model_name"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                estimated_cost=usage.get("estimated_cost", 0.0),
                latency_ms=response.get("processing_time_ms", 0.0),
                status="success"
            )
        if backend_config.get("backend_name") == "internal_qwen3_27b_vlm":
            api_prompt_tokens, api_completion_tokens, api_total_tokens, actual_usage_available = record_internal_qwen3_rate_usage(
                document_id=doc_id,
                response=response,
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                reason="successful_extraction",
            )

            log_internal_qwen3_usage(
                log_path=PROJECT_ROOT / "logs" / "internal_qwen3_usage.csv",
                document_id=doc_id,
                model=backend_config.get("model_name"),
                num_images=len(abs_paths),
                max_tokens=decoding_params["max_tokens"],
                latency_ms=response.get("processing_time_ms", 0.0),
                status="success",
                error_type=None,
                validation_status="valid",
                notes=f"Warnings: {len(warnings)}, Coercions: {len(coercions)}, Actual usage: {actual_usage_available}",
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                total_estimated_tokens=total_estimated_tokens,
                api_reported_prompt_tokens=api_prompt_tokens if api_prompt_tokens > 0 else None,
                api_reported_completion_tokens=api_completion_tokens if api_completion_tokens > 0 else None,
                api_reported_total_tokens=api_total_tokens if api_total_tokens > 0 else None,
                rolling_tokens_before_request=rolling_tokens_before_request,
                sleep_seconds_before_request=sleep_seconds_before_request,
                retry_after_rate_limit=False,
                rate_limit_reason=None
            )
        logger.info(f"[SUCCESS] Document {doc_id} parsed and validated successfully.")
        return True

    except Exception as ve:
        validation_error = f"Pydantic validation error: {str(ve)}"
        logger.error(f"Validation failed for {doc_id}: {ve}")
        save_failed_outputs(doc_id, failed_dir, raw_content, {"validation_error": validation_error, "parsed_dict": parsed_dict}, is_openrouter=is_openrouter)
        if is_openrouter:
            usage = response.get("usage", {})
            log_openrouter_usage(
                log_path=log_path,
                document_id=doc_id,
                model=backend_config.get("model_name"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                estimated_cost=usage.get("estimated_cost", 0.0),
                latency_ms=response.get("processing_time_ms", 0.0),
                status="failed"
            )
        if backend_config.get("backend_name") == "internal_qwen3_27b_vlm":
            api_prompt_tokens, api_completion_tokens, api_total_tokens, actual_usage_available = record_internal_qwen3_rate_usage(
                document_id=doc_id,
                response=response,
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                reason="validation_error",
            )
            log_internal_qwen3_usage(
                log_path=PROJECT_ROOT / "logs" / "internal_qwen3_usage.csv",
                document_id=doc_id,
                model=backend_config.get("model_name"),
                num_images=len(abs_paths),
                max_tokens=decoding_params["max_tokens"],
                latency_ms=response.get("processing_time_ms", 0.0),
                status="failed",
                error_type="validation_error",
                validation_status="invalid",
                notes=f"{validation_error[:160]}; Actual usage: {actual_usage_available}",
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
                total_estimated_tokens=total_estimated_tokens,
                api_reported_prompt_tokens=api_prompt_tokens if api_prompt_tokens > 0 else None,
                api_reported_completion_tokens=api_completion_tokens if api_completion_tokens > 0 else None,
                api_reported_total_tokens=api_total_tokens if api_total_tokens > 0 else None,
                rolling_tokens_before_request=rolling_tokens_before_request,
                sleep_seconds_before_request=sleep_seconds_before_request,
                retry_after_rate_limit=False,
                rate_limit_reason=None
            )
        return False


def save_failed_outputs(doc_id: str, failed_dir: Path, raw_text: str, error_details: dict, is_openrouter: bool = False):
    failed_dir.mkdir(parents=True, exist_ok=True)

    if is_openrouter:
        txt_path = failed_dir / "raw_response.txt"
        err_path = failed_dir / "error.json"
    else:
        txt_path = failed_dir / f"{doc_id}.txt"
        err_path = failed_dir / f"{doc_id}_error.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(err_path, "w", encoding="utf-8") as f:
        json.dump({
            "document_id": doc_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **error_details
        }, f, indent=2)

    logger.info(f"[FAILED] Saved error information to {failed_dir}")


async def async_main():
    parser = argparse.ArgumentParser(description="Clinical Prescription Raw Extraction CLI")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the manifest CSV containing image lists."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="qwen25_vl_7b",
        help="Consistent name of the extraction backend to run."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/backends.yaml",
        help="Path to the backends YAML configuration file."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="configs/prompts.yaml",
        help="Path to the prompts YAML configuration file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory to save successfully parsed CanonicalRawDoc JSONs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit execution to the first N documents from the manifest."
    )
    parser.add_argument(
        "--document-id",
        type=str,
        default=None,
        help="Optionally restrict extraction to this single document_id."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If true, skip files that have already been extracted successfully."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If true, overwrite existing successful extraction outputs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the list of documents that would be extracted without invoking VLM inference."
    )
    parser.add_argument(
        "--oracle-mode",
        action="store_true",
        help="If true, use ground truth source_type/speciality to select prompt during benchmark."
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=1.00,
        help="Strict budget limit in USD for OpenRouter API usage."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override (especially useful for openrouter model paths)"
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=1024,
        help="Maximum image dimension for resizing"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for internal image compression"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max_tokens for the selected backend"
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=500000,
        help="Token-per-minute limit for internal qwen3-27b API (default 500000)"
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=120,
        help="Request-per-minute limit for internal qwen3-27b API (default 120)"
    )
    parser.add_argument(
        "--rate-limit-window-sec",
        type=int,
        default=60,
        help="Rolling window size for rate limiting in seconds (default 60)"
    )
    parser.add_argument(
        "--rate-limit-buffer-sec",
        type=int,
        default=15,
        help="Safety buffer before rate limit window expiry in seconds (default 15)"
    )
    parser.add_argument(
        "--max-retries-rate-limit",
        type=int,
        default=1,
        help="Max retries per document after rate-limit error (default 1)"
    )
    parser.add_argument(
        "--inter-document-sleep-sec",
        type=float,
        default=0.0,
        help="Sleep this many seconds after each document attempt to avoid request bursts"
    )
    parser.add_argument(
        "--retry-failed-once",
        action="store_true",
        help="Retry a failed document once after --retry-cooldown-sec"
    )
    parser.add_argument(
        "--retry-cooldown-sec",
        type=float,
        default=90.0,
        help="Cooldown before a single failed-document retry"
    )
    parser.add_argument(
        "--disable-streaming",
        action="store_true",
        help="Use non-streaming chat completions for endpoints that return empty streamed content"
    )

    args = parser.parse_args()

    # Resolve absolute paths
    manifest_path = PROJECT_ROOT / args.manifest
    config_path = PROJECT_ROOT / args.config
    prompts_path = PROJECT_ROOT / args.prompts

    # Read manifest
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        sys.exit(1)

    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Restrict to document_id if specified
    if args.document_id:
        rows = [r for r in rows if r["document_id"] == args.document_id]
        if not rows:
            logger.error(f"No document with id '{args.document_id}' found in the manifest.")
            sys.exit(1)

    # Apply limit
    if args.limit is not None:
        rows = rows[:args.limit]

    logger.info(f"Loaded {len(rows)} documents for extraction from manifest.")

    # Load backends configuration
    if not config_path.exists():
        logger.error(f"Backends config not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        backends_config = yaml.safe_load(f)

    backend_opts = backends_config.get("backends", {}).get(args.backend)
    if not backend_opts:
        logger.error(f"Backend '{args.backend}' config not defined in backends.yaml.")
        sys.exit(1)
    backend_opts = dict(backend_opts)
    backend_opts["max_image_dim"] = args.max_image_dim
    backend_opts["jpeg_quality"] = args.jpeg_quality
    if args.max_tokens is not None:
        backend_opts["max_tokens"] = args.max_tokens

    if args.model:
        backend_opts["model_name"] = args.model

    # Set up paths dynamically
    safe_model = backend_opts.get("model_name", "").replace("/", "_")
    if args.backend == "openrouter":
        manifest_name = Path(args.manifest).name.lower()
        suffix = "subset" if "subset" in manifest_name else "smoke"
        if args.output_dir:
            output_dir = PROJECT_ROOT / args.output_dir
            raw_responses_dir = PROJECT_ROOT / "outputs" / "raw_responses" / "openrouter" / Path(args.output_dir).name
        else:
            output_dir = PROJECT_ROOT / "outputs" / "raw_extractions" / "openrouter" / f"{safe_model}_prompt_v2_{suffix}"
            raw_responses_dir = PROJECT_ROOT / "outputs" / "raw_responses" / "openrouter" / f"{safe_model}_prompt_v2_{suffix}"
        failed_dir = PROJECT_ROOT / "outputs" / "raw_extractions_failed" / "openrouter" / safe_model
    elif args.backend == "internal_qwen3_27b_vlm":
        if not args.output_dir:
            logger.error("--output-dir is required when backend is 'internal_qwen3_27b_vlm'")
            sys.exit(1)
        output_dir = PROJECT_ROOT / args.output_dir
        raw_responses_dir = PROJECT_ROOT / "outputs" / "raw_responses" / Path(args.output_dir).name
        failed_dir = PROJECT_ROOT / "outputs" / "raw_extractions_failed" / Path(args.output_dir).name
    else:
        if not args.output_dir:
            logger.error("--output-dir is required when backend is not 'openrouter'")
            sys.exit(1)
        output_dir = PROJECT_ROOT / args.output_dir
        raw_responses_dir = None
        failed_dir = PROJECT_ROOT / "outputs" / "raw_extractions_failed" / args.backend

    # Load prompts configuration
    if not prompts_path.exists():
        logger.error(f"Prompts config not found: {prompts_path}")
        sys.exit(1)
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    # Initialize backend adapter
    if args.backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY environment variable is not set!")
            sys.exit(1)
        adapter = OpenRouterBackendAdapter(
            model_id=backend_opts.get("model_name"),
            api_key=api_key,
            max_image_dim=args.max_image_dim
        )
    elif args.backend == "internal_qwen3_27b_vlm":
        api_key = os.getenv("INTERNAL_QWEN3_API_KEY")
        base_url = os.getenv("INTERNAL_QWEN3_BASE_URL", "http://10.10.110.37:4000/v1")
        model = os.getenv("INTERNAL_QWEN3_MODEL", "qwen3-27b")
        if not api_key:
            logger.error("INTERNAL_QWEN3_API_KEY environment variable is not set!")
            sys.exit(1)
        adapter = OpenAICompatibleVLMBackendAdapter(
            base_url=base_url,
            api_key=api_key,
            model_id=model,
            max_image_dim=args.max_image_dim,
            jpeg_quality=args.jpeg_quality,
            timeout=900,
            stream=not args.disable_streaming
        )
    else:
        adapter = QwenVLBackendAdapter(
            endpoint_url=backend_opts.get("endpoint_url", "http://localhost:8090/v1"),
            model_id=backend_opts.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        )


    # Initialize rate limiter for internal qwen3-27b API
    if args.backend == "internal_qwen3_27b_vlm":
        init_global_limiter(
            tpm_limit=args.tpm_limit,
            rpm_limit=args.rpm_limit,
            window_seconds=args.rate_limit_window_sec,
            buffer_seconds=args.rate_limit_buffer_sec,
            max_retries_rate_limit=args.max_retries_rate_limit,
        )
        logger.info(
            f"Rate limiter initialized: "
            f"TPM={args.tpm_limit}, RPM={args.rpm_limit}, "
            f"window={args.rate_limit_window_sec}sec, buffer={args.rate_limit_buffer_sec}sec"
        )

    # Process sequentially to avoid throttling or GPU OOM
    success_count = 0
    failure_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT_ROOT / "logs" / "openrouter_usage.csv"

    for row in rows:
        doc_id = row["document_id"]
        save_path = output_dir / f"{doc_id}.json"

        if save_path.exists() and args.resume and not args.overwrite:
            logger.info(f"Skipping {doc_id} as resume flag is enabled and output file exists.")
            success_count += 1
            continue

        # Strict Budget check before invoking API
        if args.backend == "openrouter":
            current_cumulative = get_cumulative_cost(log_path)
            if current_cumulative >= args.max_cost_usd:
                logger.warning(f"Budget exceeded! Cumulative cost {current_cumulative:.4f} USD >= limit {args.max_cost_usd:.4f} USD. Stopping extraction.")
                break

        doc_failed_dir = failed_dir / doc_id if args.backend == "openrouter" else failed_dir

        success = await extract_document(
            row=row,
            adapter=adapter,
            backend_config=backend_opts,
            prompt_config=prompt_config,
            output_dir=output_dir,
            failed_dir=doc_failed_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            oracle_mode=args.oracle_mode,
            raw_responses_dir=raw_responses_dir
        )

        if not success and args.retry_failed_once:
            logger.warning(
                f"Document {doc_id} failed on first attempt. "
                f"Cooling down for {args.retry_cooldown_sec:.1f} sec before one retry."
            )
            time.sleep(args.retry_cooldown_sec)
            success = await extract_document(
                row=row,
                adapter=adapter,
                backend_config=backend_opts,
                prompt_config=prompt_config,
                output_dir=output_dir,
                failed_dir=doc_failed_dir,
                overwrite=True,
                dry_run=args.dry_run,
                oracle_mode=args.oracle_mode,
                raw_responses_dir=raw_responses_dir
            )

        if success:
            success_count += 1
        else:
            failure_count += 1

        if args.inter_document_sleep_sec > 0:
            logger.info(f"Sleeping {args.inter_document_sleep_sec:.1f} sec before next document.")
            time.sleep(args.inter_document_sleep_sec)

    logger.info(f"Extraction Stage Completed. Total Successful: {success_count}, Total Failed: {failure_count}")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
