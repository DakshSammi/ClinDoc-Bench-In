# Stage 1B Paper Narrative Snippets

Generated: 2026-06-19T16:56:29

## Dataset and Annotation Setup

We evaluated extraction on a 53-record Indian outpatient document benchmark with manual raw-text and structured annotations. Full-coverage results use the same denominator; partial OCR systems are explicitly labelled by their available record count.

## Evaluation Metrics

Raw OCR was evaluated primarily using OCR/token F1 and normalized text similarity, with CER and WER retained as supplementary diagnostics. Structured systems were evaluated using schema parse success, scalar exact and lenient accuracy, entity exact and lenient F1, hallucination rate, missing entity rate, annotation-gap rate when available, and a configured overall extraction score.

## Raw OCR Baseline Result

GLM-OCR was the best-performing full-coverage raw OCR baseline in our benchmark, with OCR/token F1 of 0.2464 and average runtime of 2.6471 seconds per document. However, the modest text similarity and field-recall proxies show that raw OCR alone is insufficient for dependable structured extraction.

## Direct VLM Result

Among currently consolidated local direct VLM results, Qwen3-VL 8B was best-performing in our benchmark, producing schema-valid outputs for 48 of 53 records and an overall score of 0.3549. The recovered Internal Qwen3-27B compact result remains pending final Server 2 package import and is not numerically claimed here.

## OCR-to-JSON Pipeline Result

GLM-OCR followed by qwen3:8b was the best-performing OCR-to-JSON pipeline, with 50 of 53 schema-valid outputs and an overall score of 0.3628. This modest improvement over the direct local baseline did not remove the central limitation: structured extraction remains incomplete, and the missing entity rate remained high at 0.8851.

## Failure Analysis

Failures included malformed JSON, schema-shape violations, scalar type errors, and prolonged TrOCR pipeline requests. TrOCR illustrates why non-empty OCR output is not sufficient: its OCR/token F1 was 0.0082 and downstream entity lenient F1 was 0.0009. The qwen2.5 OCR-to-JSON lane was excluded because its parseable responses did not follow the canonical schema.

## Final Conclusion

The benchmark identifies useful relative differences between OCR, direct VLM, and OCR-to-JSON approaches, but it does not establish clinical reliability. High missing entity rates persist across structured systems. These findings support a benchmark-first framing, careful error analysis, and explicit separation of schema validity from extraction completeness.
