# Reproduction Guide

This repository supports code-level reproduction. Dataset-level reproduction requires private access to the 53-record benchmark and redaction procedures.

1. Create a Python environment and install `requirements/base.txt`.
2. Install server-specific extras from `requirements/server1_ollama.txt`, `requirements/server2_ocr.txt`, or `requirements/server2_vllm.txt`.
3. Copy private data, manifests, and model outputs outside the repository.
4. Configure private paths in a local YAML copied from `configs/servers/`.
5. Run evaluators from `scripts/` with explicit input and output paths.
6. Generate paper tables into `paper_assets/tables/`.
7. Run `docs/github_ready_checklist.md` before committing any changes.

The checked-in paper assets are aggregate/redacted outputs only.
