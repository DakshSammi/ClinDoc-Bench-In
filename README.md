# ClinDoc-Bench-IN

ClinDoc-Bench-IN is a benchmark codebase for evaluating clinical document understanding on Indian outpatient prescriptions. The current in-house benchmark uses 53 prescription records with private annotations; raw images, raw ground-truth annotations, OCR text, model outputs, logs, failed cases, and model caches are intentionally not included in this repository.

The benchmark separates three evaluation families:

1. Raw OCR/document parser baselines, scored with OCR/token F1 and text similarity.
2. Direct VLM structured extraction baselines, scored against a canonical JSON schema.
3. Hybrid OCR-plus-LLM pipelines, where OCR output is converted into canonical structured JSON.

Stage 1C adds a separate semantic-enhanced extraction benchmark. It evaluates evidence-backed semantic inference over valid Stage 1B structured outputs; it is not treated as gold semantic accuracy.

No paid API results are included as completed final benchmark results. Redacted examples only should be committed.

## Repository Layout

- `src/`: schemas, adapters, and benchmark metric modules.
- `scripts/`: reproducible runners and evaluators.
- `configs/`: local and server configuration templates.
- `prompts/`: prompt templates for extraction and semantic enrichment.
- `docs/`: protocol, schema, setup, reference audit, and GitHub safety notes.
- `paper_assets/`: paper-ready tables and non-PHI figure drafts.

## Current Paper Assets

Server 1 paper-facing assets are under `paper_assets/tables/server1/` and combined assets are under `paper_assets/tables/combined/`. Server 2 assets are expected under `paper_assets/tables/server2/` once imported from the private Server 2 environment.

## Statistical Tests

Run paired statistical testing from the repository root:

```bash
python scripts/run_stage1b_statistical_tests.py \
  --structured "Qwen3-VL 8B=paper_assets/tables/server1/per_document_structured/qwen3_vl_8b.csv" \
  --structured "LLaVA 13B=paper_assets/tables/server1/per_document_structured/llava_13b_diagnostic.csv" \
  --structured "GLM-OCR + qwen3:8b=paper_assets/tables/server1/per_document_structured/glm_ocr_qwen3_8b.csv" \
  --structured "docTR + qwen3:8b=paper_assets/tables/server1/per_document_structured/doctr_qwen3_8b.csv" \
  --structured "TrOCR + qwen3:8b=paper_assets/tables/server1/per_document_structured/trocr_qwen3_8b.csv" \
  --ocr "GLM-OCR=paper_assets/tables/server1/per_document_raw_ocr/glm_ocr.csv" \
  --ocr "docTR=paper_assets/tables/server1/per_document_raw_ocr/doctr.csv" \
  --ocr "TrOCR=paper_assets/tables/server1/per_document_raw_ocr/trocr.csv" \
  --ocr "Docling=paper_assets/tables/server1/per_document_raw_ocr/docling.csv" \
  --ocr "Surya=paper_assets/tables/server1/per_document_raw_ocr/surya.csv" \
  --output-dir paper_assets/tables/combined
```

Add the Server 2 internal Qwen3-27B compact CSV after importing the final Server 2 package.

## Data Safety

Do not commit:

- Prescription images.
- Raw ground-truth annotations.
- Raw OCR/model outputs.
- Failed cases, logs, compressed images, or benchmark output directories.
- `.env`, API keys, model weights, or archives.

See `docs/github_ready_checklist.md` before committing or pushing.
