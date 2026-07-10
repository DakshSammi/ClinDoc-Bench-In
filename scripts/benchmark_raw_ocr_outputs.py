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

"""Raw OCR benchmark evaluator for Stage 1B.

Input is an OCR handoff CSV with document_id and ocr_text_path. This evaluator
does not validate OCR text as CanonicalRawDoc. It computes text similarity,
edit-distance metrics, token overlap, field recall proxies, runtime, and grouped
summaries against manual GT text/entities.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokens(text: str) -> List[str]:
    return re.findall(r"[\w./:+-]+", norm(text))


def numeric_tokens(text: str) -> List[str]:
    return re.findall(r"\d+(?:[./:-]\d+)*", norm(text))


def edit_distance(a: List[Any], b: List[Any]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def overlap_metrics(pred_tokens: List[str], gt_tokens: List[str]) -> Tuple[float, float, float]:
    pc, gc = Counter(pred_tokens), Counter(gt_tokens)
    overlap = sum((pc & gc).values())
    precision = overlap / max(1, sum(pc.values()))
    recall = overlap / max(1, sum(gc.values()))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def gt_reference_text(gt_path: Path) -> Tuple[str, Dict[str, Any]]:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    parts: List[str] = []
    raw_text = data.get("raw_text", {})
    if isinstance(raw_text, dict):
        parts.append(str(raw_text.get("full_text", "")))
        for page in raw_text.get("pages", []) or []:
            if isinstance(page, dict):
                parts.append(str(page.get("text", "")))
    entities = data.get("raw_entities", {})
    parts.append(json.dumps(entities, ensure_ascii=False))
    return "\n".join(p for p in parts if p), data


def terms_from_gt(data: Dict[str, Any], bucket: str) -> List[str]:
    ent = data.get("raw_entities", {})
    if bucket == "patient_name":
        p = ent.get("patient_information", {})
        return [p.get("name", ""), p.get("patient_identifier", "")]
    if bucket == "date":
        e = ent.get("encounter_information", {})
        return [e.get("date", "")]
    if bucket == "age_gender":
        p = ent.get("patient_information", {})
        return [p.get("age", ""), p.get("gender", "")]
    if bucket == "medication":
        terms = []
        for m in ent.get("medications", []) or []:
            if isinstance(m, dict):
                terms.extend([m.get("raw_medication_text", ""), m.get("raw_name", ""), m.get("raw_line_text", "")])
            elif isinstance(m, str):
                terms.append(m)
        return terms
    if bucket == "vitals":
        v = ent.get("vitals", {})
        if isinstance(v, dict):
            return list(v.values())
        return []
    if bucket == "complaints":
        terms = []
        for item in ent.get("complaints_or_diagnosis", []) or []:
            if isinstance(item, dict):
                terms.append(item.get("raw_text", "") or item.get("text", ""))
            elif isinstance(item, str):
                terms.append(item)
        return terms
    return []


def recall_proxy(pred_text: str, gt_data: Dict[str, Any], bucket: str) -> float:
    pred = norm(pred_text)
    raw_terms = [norm(t) for t in terms_from_gt(gt_data, bucket) if norm(str(t))]
    term_tokens = []
    for term in raw_terms:
        toks = [t for t in tokens(term) if len(t) >= 2]
        if toks:
            term_tokens.extend(toks[:4])
    if not term_tokens:
        return 0.0
    found = sum(1 for t in term_tokens if t in pred)
    return round(found / len(term_tokens), 4)


def runtime_from_log(log_path: str) -> str:
    if not log_path:
        return ""
    p = Path(log_path)
    if not p.exists():
        return ""
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("runtime_seconds", "")
    except Exception:
        return ""


def score_doc(row: Dict[str, str], manifest_row: Dict[str, str], engine: str) -> Dict[str, Any]:
    ocr_path = Path(row["ocr_text_path"])
    pred_text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
    gt_text, gt_data = gt_reference_text(PROJECT_ROOT / manifest_row["ground_truth_json"])
    pred_chars, gt_chars = list(norm(pred_text)), list(norm(gt_text))
    pred_tok, gt_tok = tokens(pred_text), tokens(gt_text)
    cer = edit_distance(pred_chars, gt_chars) / max(1, len(gt_chars))
    wer = edit_distance(pred_tok, gt_tok) / max(1, len(gt_tok))
    precision, recall, f1 = overlap_metrics(pred_tok, gt_tok)
    gt_nums = numeric_tokens(gt_text)
    pred_nums = set(numeric_tokens(pred_text))
    numeric_recall = sum(1 for n in gt_nums if n in pred_nums) / max(1, len(gt_nums))
    return {
        "document_id": row["document_id"],
        "patient_id": manifest_row.get("patient_id", ""),
        "engine": engine,
        "department_inferred": manifest_row.get("department_inferred", ""),
        "hospital_name": manifest_row.get("hospital_name", ""),
        "is_multi_page": manifest_row.get("is_multi_page", ""),
        "is_same_page_multi_view": manifest_row.get("is_same_page_multi_view", ""),
        "non_empty_output": int(bool(pred_text.strip())),
        "ocr_chars": len(pred_text),
        "gt_chars": len(gt_text),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "normalized_edit_similarity": round(max(0.0, 1.0 - cer), 4),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "numeric_token_recall": round(numeric_recall, 4),
        "patient_name_recall_proxy": recall_proxy(pred_text, gt_data, "patient_name"),
        "date_recall_proxy": recall_proxy(pred_text, gt_data, "date"),
        "age_gender_recall_proxy": recall_proxy(pred_text, gt_data, "age_gender"),
        "medication_recall_proxy": recall_proxy(pred_text, gt_data, "medication"),
        "vitals_recall_proxy": recall_proxy(pred_text, gt_data, "vitals"),
        "complaints_diagnosis_recall_proxy": recall_proxy(pred_text, gt_data, "complaints"),
        "runtime_seconds": runtime_from_log(row.get("log_path", "")) or row.get("runtime", ""),
        "ocr_text_path": str(ocr_path),
    }


def avg(rows: List[Dict[str, Any]], key: str) -> Any:
    vals = []
    for r in rows:
        try:
            if r.get(key) not in ("", None):
                vals.append(float(r[key]))
        except Exception:
            pass
    return round(sum(vals) / len(vals), 4) if vals else ""


def summarize(rows: List[Dict[str, Any]], group_key: str, label_name: str = "group") -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r.get(group_key) or "Unknown"].append(r)
    out = []
    for group, items in sorted(groups.items()):
        out.append({
            label_name: group,
            "records": len(items),
            "non_empty_output_rate": avg(items, "non_empty_output"),
            "cer": avg(items, "cer"),
            "wer": avg(items, "wer"),
            "normalized_edit_similarity": avg(items, "normalized_edit_similarity"),
            "token_precision": avg(items, "token_precision"),
            "token_recall": avg(items, "token_recall"),
            "token_f1": avg(items, "token_f1"),
            "numeric_token_recall": avg(items, "numeric_token_recall"),
            "runtime_seconds": avg(items, "runtime_seconds"),
        })
    return out


def qualitative(rows: List[Dict[str, Any]]) -> str:
    sorted_rows = sorted(rows, key=lambda r: float(r.get("token_f1") or 0))
    picks = sorted_rows[:3] + sorted_rows[-3:]
    lines = ["# GLM-OCR Qualitative Examples", ""]
    seen = set()
    for r in picks:
        if r["document_id"] in seen:
            continue
        seen.add(r["document_id"])
        text = Path(r["ocr_text_path"]).read_text(encoding="utf-8", errors="ignore")[:1200]
        lines.extend([
            f"## {r['document_id']}",
            "",
            f"- Token F1: {r['token_f1']}",
            f"- CER: {r['cer']}",
            "",
            "```text",
            text,
            "```",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    handoff = read_csv(Path(args.handoff))
    manifest = {r["document_id"]: r for r in read_csv(Path(args.manifest))}
    rows = [
        score_doc(r, manifest[r["document_id"]], args.engine)
        for r in handoff
        if r["document_id"] in manifest
        and (r.get("ocr_engine") or r.get("engine") or args.engine) == args.engine
        and (r.get("status") or "").lower() in {"available", "ok", "success"}
    ]
    out_dir = Path(args.output_dir)
    fields = [
        "document_id", "patient_id", "engine", "department_inferred", "hospital_name", "is_multi_page", "is_same_page_multi_view",
        "non_empty_output", "ocr_chars", "gt_chars", "cer", "wer", "normalized_edit_similarity",
        "token_precision", "token_recall", "token_f1", "numeric_token_recall",
        "patient_name_recall_proxy", "date_recall_proxy", "age_gender_recall_proxy",
        "medication_recall_proxy", "vitals_recall_proxy", "complaints_diagnosis_recall_proxy",
        "runtime_seconds", "ocr_text_path",
    ]
    write_csv(out_dir / "per_document_ocr_scores.csv", rows, fields)
    summary = {
        "engine": args.engine,
        "records": len(rows),
        "non_empty_output_rate": avg(rows, "non_empty_output"),
        "cer": avg(rows, "cer"),
        "wer": avg(rows, "wer"),
        "normalized_edit_similarity": avg(rows, "normalized_edit_similarity"),
        "token_precision": avg(rows, "token_precision"),
        "token_recall": avg(rows, "token_recall"),
        "token_f1": avg(rows, "token_f1"),
        "numeric_token_recall": avg(rows, "numeric_token_recall"),
        "patient_name_recall_proxy": avg(rows, "patient_name_recall_proxy"),
        "date_recall_proxy": avg(rows, "date_recall_proxy"),
        "age_gender_recall_proxy": avg(rows, "age_gender_recall_proxy"),
        "medication_recall_proxy": avg(rows, "medication_recall_proxy"),
        "vitals_recall_proxy": avg(rows, "vitals_recall_proxy"),
        "complaints_diagnosis_recall_proxy": avg(rows, "complaints_diagnosis_recall_proxy"),
        "runtime_seconds": avg(rows, "runtime_seconds"),
    }
    write_json(out_dir / "summary_metrics.json", summary)
    write_csv(out_dir / "per_engine_summary.csv", [summary], list(summary.keys()))
    write_csv(out_dir / "per_department_ocr_scores.csv", summarize(rows, "department_inferred"), ["group", "records", "non_empty_output_rate", "cer", "wer", "normalized_edit_similarity", "token_precision", "token_recall", "token_f1", "numeric_token_recall", "runtime_seconds"])
    write_csv(out_dir / "per_hospital_ocr_scores.csv", summarize(rows, "hospital_name"), ["group", "records", "non_empty_output_rate", "cer", "wer", "normalized_edit_similarity", "token_precision", "token_recall", "token_f1", "numeric_token_recall", "runtime_seconds"])
    page_groups = []
    for label, pred in [
        ("single_page", lambda r: r.get("is_multi_page") == "false" and r.get("is_same_page_multi_view") == "false"),
        ("multi_page", lambda r: r.get("is_multi_page") == "true"),
        ("same_page_multi_view", lambda r: r.get("is_same_page_multi_view") == "true"),
    ]:
        items = [r for r in rows if pred(r)]
        d = {"group": label, "records": len(items), "non_empty_output_rate": avg(items, "non_empty_output"), "token_f1": avg(items, "token_f1"), "cer": avg(items, "cer"), "runtime_seconds": avg(items, "runtime_seconds")}
        page_groups.append(d)
    write_csv(out_dir / "per_page_type_ocr_scores.csv", page_groups, ["group", "records", "non_empty_output_rate", "token_f1", "cer", "runtime_seconds"])
    write_text(out_dir / "qualitative_examples.md", qualitative(rows))
    summary_slug = re.sub(r"[^a-z0-9]+", "_", args.engine.lower()).strip("_")
    write_text(PROJECT_ROOT / "reports" / f"stage1b_server1_{summary_slug}_raw_ocr_benchmark_summary.md", "\n".join([
        f"# Stage 1B Final release {args.engine} Raw OCR Benchmark Summary",
        "",
        f"- Records: {summary['records']}",
        f"- Non-empty output rate: {summary['non_empty_output_rate']}",
        f"- Token F1: {summary['token_f1']}",
        f"- CER: {summary['cer']}",
        f"- WER: {summary['wer']}",
        f"- Normalized edit similarity: {summary['normalized_edit_similarity']}",
        f"- Runtime seconds/doc: {summary['runtime_seconds']}",
        f"- Output directory: `{out_dir}`",
        "",
    ]))


if __name__ == "__main__":
    main()
