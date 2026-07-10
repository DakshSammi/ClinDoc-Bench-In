# Publication Readiness Report

Generated from frozen benchmark reports. No benchmark outputs were rerun or overwritten.

## Repository Completeness

- README, license, citation, environment, contribution, security, changelog, and release checklist files are present.
- Documentation exists under `docs/`.
- Publication figures, tables, appendix assets, and anonymized examples are generated under `paper_assets/`.

## GitHub Readiness

- Apache 2.0 license and NOTICE are present.
- Issue and pull request templates are present.
- CI workflows for lint, format, tests, and documentation smoke checks are present.
- `.env.example` contains placeholders only.

## Paper Readiness

- Primary table lanes: 14
- Appendix lanes: 12
- Excluded lanes: 2
- Figures 1-19 plus Figures 8B, 12B, and 12C are exported as SVG, PDF, and 600 dpi PNG.
- Figure 20 was intentionally removed.
- Tables 1-12 are exported as CSV, LaTeX, and Markdown.
- Figure captions are stored in `paper_assets/figures/captions.md`.

## Remaining TODOs

- Replace placeholder author names in `CITATION.cff`.
- Replace repository URL placeholders.
- Fill security contact email.
- Fill hardware details before camera-ready submission.
- Manually inspect anonymized examples before public release.
- Confirm BDA 2026 metadata once the submission status changes.

## Suggested Paper Figure Order

1. Overall benchmark architecture.
2. Dataset composition.
3. Benchmark workflow.
4. Evaluation pipeline.
5. OCR performance.
6. Direct VLM comparison.
7. Hybrid pipeline comparison.
8. Department x model heatmap.
8B. Department win count.
9. Runtime vs accuracy.
10. Models x metrics heatmap.
11. Bootstrap confidence intervals.
12. Wilcoxon and Holm significance.
12B. Friedman omnibus tests.
12C. McNemar exact test summary.
13. Benchmark leaderboard.
14. Qualitative examples.
15. Good vs bad prescription examples.
16. Example ground truth annotation.
17. Hybrid pipeline architecture.
18. Direct VLM pipeline.
19. OCR pipeline.

## Checklist Before GitHub Release

- Verify no secrets are present.
- Verify no unanonymized examples are published.
- Verify frozen benchmark marker remains unchanged.
- Run CI locally.
- Review README badges and links.
