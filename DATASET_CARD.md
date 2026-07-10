# Dataset Card

## Dataset Name

ClinDoc-Bench-IN Frozen Benchmark Dataset v1.0

## Motivation

This dataset exists to benchmark OCR, direct VLM, and hybrid OCR-to-LLM systems on Indian clinical prescription-style documents under a shared evaluation protocol.

## Composition

- 90 patients
- 125 documents
- 150 images

## Source Material

The benchmark contains handwritten prescriptions, OPD sheets, medical reports, laboratory pages, and other related clinical document formats represented in the frozen benchmark manifest.

## Annotation

- canonical JSON ground truth
- patient and encounter fields
- medication and advice entities
- scalar and entity-friendly structure for downstream evaluation

## Intended Use

- benchmarking OCR systems
- benchmarking direct VLM systems
- benchmarking hybrid OCR-plus-LLM systems
- reproducing the frozen BDA 2026 paper benchmark

## Out Of Scope

- clinical decision support
- medical diagnosis
- deployment without additional privacy review
- use as a demographic or epidemiological dataset

## Privacy And Ethics

- public paper examples are anonymized derivatives
- raw patient identifiers and unanonymized images should not be committed or redistributed
- frozen benchmark outputs are immutable once released

## Limitations

- relatively small benchmark compared with web-scale document datasets
- strong prescription emphasis
- institution-specific layout and handwriting biases
- multilingual and multi-document expansion remains future work

## Governance

See [GOVERNANCE.md](GOVERNANCE.md) for freeze policy and future versioning.
