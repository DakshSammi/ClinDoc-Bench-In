# GitHub Ready Checklist

- [ ] No raw prescription images are staged.
- [ ] No raw ground-truth annotations or PHI-bearing manifests are staged.
- [ ] No raw OCR text, raw model responses, failed cases, logs, compressed images, or benchmark output directories are staged.
- [ ] No `.env`, API keys, SSH keys, model caches, or model weights are staged.
- [ ] OCR-only and structured metrics are kept in separate tables.
- [ ] qwen2.5 OCR-to-JSON wrong-schema lane is labelled excluded, not successful.
- [ ] Server 2 old streaming failures are not confused with recovered compact Qwen3 results.
- [ ] Missing entity rate and annotation-gap rate are reported for structured systems where available.
- [ ] Paid API results are not included as completed final benchmark rows.
- [ ] Partial/interim results are labelled.
- [ ] `python -m py_compile` passes for tracked Python files.
- [ ] Main scripts respond to `--help` where applicable.
- [ ] `git diff --cached --name-only` shows only safe code, docs, configs, prompts, redacted examples, and aggregate paper assets.
