# Statistics

The final benchmark statistics are computed over the expanded 90-patient, 125-document dataset.

## Tests

- Wilcoxon signed-rank tests for paired score comparisons.
- McNemar exact tests where binary success outcomes are available.
- Friedman tests across multiple systems.
- Bootstrap confidence intervals for primary scores.
- Holm correction for multiple comparisons.

## Files

- `statistical_tests.csv`
- `bootstrap_confidence_intervals.csv`
- `department_tables.csv`
- `department_heatmap.csv`

All files are read from the frozen reports directory when generating publication assets.
