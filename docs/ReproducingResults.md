# Reproducing Results

The final benchmark itself is frozen. Reproduction means regenerating paper assets and verifying that reports are present, not rerunning model inference.

## Regenerate Paper Assets

```bash
python scripts/generate_publication_assets.py
```

This command reads frozen CSVs and writes figures, tables, anonymized examples, and supplementary outlines under `paper_assets/`.

## Verify Freeze

```bash
test -f benchmark_v2/final_day_freeze_20260709/reports/FINAL_BENCHMARK_FROZEN.txt
```

## Compare a New Model

Run the new model in a separate output directory, evaluate it using the same metrics, and compare its CSV with the frozen leaderboard. Do not modify frozen CSVs.
