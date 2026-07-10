# Evaluation

Evaluation has two families: raw OCR scoring and structured extraction scoring.

## Raw OCR Metrics

- Character error rate.
- Word error rate.
- Edit similarity.
- Token precision, recall, and F1.
- Numeric token recall.
- Clinical proxy recalls for names, dates, demographics, medications, vitals, complaints, and diagnoses.

## Structured Metrics

- Schema parse success.
- Scalar exact accuracy.
- Scalar lenient accuracy.
- Entity exact F1.
- Entity lenient F1.
- Hallucination rate.
- Missing entity rate.
- Annotation gap rate.
- Overall extraction score.

The final benchmark uses token F1 as the raw OCR primary metric and overall extraction score as the structured primary metric.
