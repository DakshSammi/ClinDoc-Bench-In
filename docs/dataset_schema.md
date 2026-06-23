# Dataset And Output Schema

The public repository does not include the private 53-record dataset. The private manifest maps each `document_id` to source images, ground-truth JSON, document metadata, and evaluation splits.

Structured predictions are validated against the canonical raw extraction schema in `src/schemas/raw_extraction.py`. Semantic enrichment outputs are validated against `src/schemas/semantic_extraction.py`.

Expected private manifest fields include:

- `document_id`
- `patient_id`
- `department_inferred`
- `hospital_name`
- `is_multi_page`
- `is_same_page_multi_view`
- `ground_truth_json`
- source image paths

Do not commit private manifest files containing PHI or raw annotation paths.
