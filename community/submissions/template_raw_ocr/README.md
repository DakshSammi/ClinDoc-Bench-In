# Template Raw OCR Submission

Replace this directory with your own OCR submission package.

Required files:

- `metadata.yaml`
- `predictions/<document_id>.txt`
- `runtime.csv`
- `README.md`

Validate with:

```bash
python scripts/validate_submission.py \
    --submission-dir community/submissions/template_raw_ocr
```
