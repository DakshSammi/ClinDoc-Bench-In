# Benchmark

The frozen benchmark compares model lanes across three tracks:

| Track | Description |
| --- | --- |
| Raw OCR | Image-to-text extraction |
| Direct VLM | Image-to-canonical-JSON extraction |
| Hybrid OCR+LLM | OCR text-to-canonical-JSON extraction |

## Frozen Outputs

Final outputs are under:

`benchmark_v2/final_day_freeze_20260709/reports/`

The freeze marker is:

`benchmark_v2/final_day_freeze_20260709/reports/FINAL_BENCHMARK_FROZEN.txt`

Do not rerun, overwrite, or regenerate frozen results. New research lanes should write to separate experiment directories.
