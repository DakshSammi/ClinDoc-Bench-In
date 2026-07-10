# Architecture

ClinDoc-Bench-IN separates data preparation, model inference, evaluation, statistics, and publication assets.

The frozen benchmark lives under `benchmark/final/` and is read-only. New experiments should write elsewhere and then be compared against the frozen reports.

## System Flow

```mermaid
flowchart TD
    A[Clinical document images] --> B[Benchmark manifest]
    B --> C1[Raw OCR adapters]
    B --> C2[Direct VLM adapters]
    C1 --> C3[Hybrid OCR-to-LLM adapters]
    C1 --> D[Raw OCR evaluation]
    C2 --> E[Structured evaluation]
    C3 --> E
    D --> F[Frozen reports]
    E --> F
    F --> G[Statistics]
    G --> H[Figures and tables]
```

## Key Principles

- The manifest is the contract between data, predictions, and evaluation.
- Each lane is evaluated independently before aggregation.
- Canonical JSON is validated before structured scoring.
- Provenance records preserve artifact location, timestamp, coverage, and publication status.
- Publication assets are regenerated from frozen CSVs, not from live model calls.
