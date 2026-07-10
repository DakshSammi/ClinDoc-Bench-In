# OCR Models

Raw OCR lanes are evaluated as text extraction systems. OCR output is scored against canonical ground truth converted to text-compatible targets.

## Adding OCR

1. Produce one text file per `document_id`.
2. Preserve source document ordering.
3. Record runtime and OCR engine metadata.
4. Evaluate with the raw OCR scoring script.
5. Store outputs in a new experiment directory.

## Publication Metrics

The OCR paper table reports token F1, token precision, token recall, edit similarity, character error rate, word error rate, and runtime where available.
