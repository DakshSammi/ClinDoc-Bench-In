# Hybrid Pipelines

Hybrid lanes combine OCR output with a structured extraction model.

## Pipeline

```mermaid
flowchart LR
    A[Document image] --> B[OCR engine]
    B --> C[OCR text]
    C --> D[LLM structuring prompt]
    D --> E[Canonical JSON]
    E --> F[Schema validation]
    F --> G[Structured metrics]
```

## OCR Sources

Hybrid experiments may use GLM OCR, DocTR, EasyOCR, TrOCR, Surya, Docling, or any new OCR lane that emits one text output per benchmark document.

## Reliability

Production-grade hybrid lanes should checkpoint every document, save raw model responses, validate JSON immediately, and resume only missing or invalid files.
