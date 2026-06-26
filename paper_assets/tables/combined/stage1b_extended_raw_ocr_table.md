# Stage 1B Extended Raw OCR Table

Generated: 2026-06-26T18:48:23

## Full-53 systems (primary comparison)

| System | N | Non-empty | OCR/token F1 | Text similarity | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|
| GLM-OCR | 53/53 | 53 | 0.2464 | 0.1874 | 2.6471 | full_53_raw_ocr | Accepted Server 1 full benchmark raw OCR baseline. |
| docTR | 53/53 | 53 | 0.1980 | 0.1851 | 9.1378 | full_53_raw_ocr | Server 2 OCR handoff re-benchmarked locally with the Stage 1B raw OCR evaluator. |
| TrOCR | 53/53 | 53 | 0.0082 | 0.0098 | 6.6103 | full_53_raw_ocr | Server 2 OCR handoff re-benchmarked locally with the Stage 1B raw OCR evaluator. |
| Docling | 53/53 | 53 | 0.1536 | 0.1529 | 50.4308 | full_53_raw_ocr | Extended final pass using the imported Server 2 OCR handoff. |
| Surya | 53/53 | 53 | 0.2362 | 0.1873 | 138.0848 | full_53_raw_ocr | Extended final pass using the imported Server 2 OCR handoff. |
| EasyOCR | 53/53 | 53 | 0.1606 | 0.1728 | 15.0650 | full_53_raw_ocr | Extended final pass using the imported Server 2 OCR handoff. |

## Coverage-limited systems

| System | N | Non-empty | OCR/token F1 | Text similarity | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL raw OCR | 52/53 | 45 | 0.4449 | 0.1636 | 16.2 | coverage_limited_52_53_with_metrics | Qwen2.5-VL raw OCR: 52/53 documents recovered (p26 unrecovered); per-document scores computed via canonical raw OCR evaluator. Excluded from primary full-53 paired tests because coverage is 52/53. |

## Partial / Interim Raw OCR

| System | N | Non-empty | OCR/token F1 | Text similarity | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|
| Marker | 19/53 | 19 | 0.1221 | 0.1264 | 295.9645 | partial_interim_raw_ocr | Only 19 usable Marker rows available in the imported handoff (Server 2 had 39 but handoff was limited). Keep separate from full-53 comparisons. |

Qwen2.5-VL raw OCR is included with computed per-document metrics (52/53 coverage, 45/53 non-empty). It is excluded from primary full-53 paired tests because 1 document (p26) was not recovered. Marker remains partial/interim (19/53) and is excluded from full-53 statistical comparisons.
