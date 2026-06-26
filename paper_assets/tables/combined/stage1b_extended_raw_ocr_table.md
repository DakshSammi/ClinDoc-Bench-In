# Stage 1B Extended Raw OCR Table

Generated: 2026-06-26T15:07:43

| System | N | Non-empty | OCR/token F1 | Text similarity | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|
| GLM-OCR | 53/53 | 53 | 0.2464 | 0.1874 | 2.6471 | full_53_best_server1_raw_ocr | Accepted Server 1 full benchmark raw OCR baseline. |
| docTR | 53/53 | 53 | 0.1980 | 0.1851 | 9.1378 | full_53_server2_handoff_rebenchmarked | Server 2 OCR handoff re-benchmarked locally with the Stage 1B raw OCR evaluator. |
| TrOCR | 53/53 | 53 | 0.0082 | 0.0098 | 6.6103 | full_53_server2_handoff_rebenchmarked | Server 2 OCR handoff re-benchmarked locally with the Stage 1B raw OCR evaluator. |
| Docling | 53/53 | 53 | 0.1536 | 0.1529 | 50.4308 | full_53_server2_handoff_rebenchmarked | Extended final pass using the imported Server 2 OCR handoff. |
| Surya | 53/53 | 53 | 0.2362 | 0.1873 | 138.0848 | full_53_server2_handoff_rebenchmarked | Extended final pass using the imported Server 2 OCR handoff. |
| EasyOCR | 53/53 | 53 | 0.1606 | 0.1728 | 15.0650 | full_53_server2_handoff_rebenchmarked | Extended final pass using the imported Server 2 OCR handoff. |
| Qwen2.5-VL raw OCR | 48/53 | 48 |  |  | 14.5 | coverage_only_imported_status | 5 no_images cases are data-availability gaps, not model failures; OCR quality metrics were not included in the imported Server 2 package. |

## Partial / Interim Raw OCR

| System | N | Non-empty | OCR/token F1 | Text similarity | Avg runtime s | Label | Notes |
|---|---|---|---|---|---|---|---|
| Marker | 19/53 | 19 | 0.1221 | 0.1264 | 295.9645 | partial_interim_only | Only 19 usable Server 2 marker rows were imported; keep separate from full-53 comparisons. |

Qwen2.5-VL raw OCR is included as a coverage/runtime row only because the imported Server 2 package did not include per-document OCR benchmark scores. Marker remains partial/interim and is excluded from full-53 statistical comparisons.
