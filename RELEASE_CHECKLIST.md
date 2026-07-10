# Release Checklist

## GitHub Release

- [ ] Confirm `benchmark/final/reports/final_model_registry.csv` exists.
- [ ] Confirm no benchmark outputs were modified after the freeze.
- [ ] Replace placeholder authors in `CITATION.cff`.
- [ ] Replace repository URLs in `CITATION.cff`.
- [ ] Replace security contact placeholders.
- [ ] Review `.env.example` for completeness and absence of secrets.
- [ ] Run `pytest`.
- [ ] Run `ruff check .`.
- [ ] Run `python scripts/generate_publication_assets.py`.
- [ ] Inspect anonymized examples under `paper_assets/examples/`.
- [ ] Confirm README badges and BDA 2026 status are correct.

## BDA Submission

- [ ] Verify final figure order.
- [ ] Verify final table order.
- [ ] Export SVG/PDF/PNG figure variants.
- [ ] Confirm all patient-facing examples are anonymized with opaque boxes.
- [ ] Confirm supplementary checklist and artifact description are complete.
- [ ] Confirm citation metadata is paper-ready.
