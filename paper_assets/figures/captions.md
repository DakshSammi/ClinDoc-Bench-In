# Figure Captions

**Figure 1. Overall benchmark architecture.** The benchmark separates dataset construction, three inference tracks, canonical JSON evaluation, and statistical analysis. This organization makes model provenance and metric computation explicit while keeping the frozen benchmark reproducible.

**Figure 2. Dataset composition.** The figure summarizes department distribution, source-document categories, and page structure. Department variants such as General Medicine units are consolidated to avoid repeated labels for the same clinical service.

**Figure 3. Benchmark workflow.** The workflow moves left-to-right from a fixed dataset to inference, unified evaluation, statistical testing, rankings, and paper figures. No live inference is required for the publication asset stage.

**Figure 4. Evaluation pipeline.** Predictions and ground truth are aligned and scored before paired statistical analysis. The figure separates field-level matching from aggregate ranking so metric provenance is clear.

**Figure 5. OCR Performance.** OCR systems are compared using token F1, precision, and recall. This view highlights transcription quality before structured extraction.

**Figure 6. Direct VLM Comparison.** Direct VLM models are compared by overall extraction score on the frozen dataset. Complete primary-table lanes are displayed with their publication labels.

**Figure 7. Hybrid Pipeline Comparison.** Hybrid OCR-to-LLM lanes are compared by structured extraction score. The ranking reflects how OCR quality and language-model structuring interact.

**Figure 8. Department x model heatmap.** The heatmap compares top models across departments and outlines the best model for each department. Related service labels such as General Medicine unit variants are consolidated.

**Figure 8B. Department win count.** This companion chart counts how often each model is the top performer across departments. It gives a compact robustness view separate from the heatmap.

**Figure 9. Runtime versus accuracy.** The scatter plot compares runtime and primary score only for lanes that have both values in the frozen reports. Each color represents one full benchmark lane, so repeated OCR sources paired with different downstream models remain visually distinct.

**Figure 10. Models x Metrics Heatmap.** The heatmap summarizes primary score, schema success, hallucination rate, and missing entity rate across models. It provides a compact cross-metric diagnostic view.

**Figure 11. Bootstrap confidence intervals.** The forest plot shows bootstrap confidence intervals for each model's primary score. Intervals convey uncertainty across the 125-document benchmark.

**Figure 12. Wilcoxon and Holm significance.** Pairwise Wilcoxon signed-rank tests are shown for structured primary-table lanes using Holm-adjusted p-values from the frozen statistics file. Brighter cells indicate stronger adjusted evidence of paired performance differences.

**Figure 12B. Friedman omnibus tests.** The Friedman omnibus tests were run for raw OCR and structured model families over the 125-document benchmark. The plot reports Holm-adjusted omnibus evidence from the frozen statistics file.

**Figure 12C. McNemar exact test summary.** McNemar exact tests were computed for paired binary schema-success outcomes. All frozen comparisons have zero discordant-count statistic and no Holm-significant result, so this summary reports applicability without overstating significance.

**Figure 13. Benchmark Leaderboard.** The leaderboard ranks selected lanes by their primary metric within the frozen benchmark. Display labels are publication-facing and do not alter frozen provenance files.

**Figure 14. Qualitative example.** The panel shows an anonymized input, corresponding canonical JSON, representative prediction structure, and the color convention used for error analysis. Green denotes matched fields, orange denotes missing content, and red denotes hallucinated content.

**Figure 15. Good and poor quality examples.** The examples illustrate the visual range of the dataset after targeted anonymization. The good-quality example has clearer structure and handwriting, while the poor-quality example shows lower contrast and incomplete visibility.

**Figure 16. Ground truth annotation example.** The figure pairs an anonymized prescription with a canonical JSON excerpt. It shows how visual evidence is translated into structured fields before evaluation.

**Figure 17. Hybrid pipeline architecture.** Hybrid lanes first transcribe the prescription and then use a language model to convert OCR text into canonical JSON. The validation step ensures outputs remain compatible with the unified scoring engine.

**Figure 18. Direct VLM pipeline.** Direct VLM lanes process prescription images without an intermediate OCR file. The model response is parsed into structured JSON and validated before metric computation.

**Figure 19. OCR pipeline.** The OCR pipeline converts prescription images into text for raw OCR scoring and downstream hybrid extraction. The same text outputs can be reused by multiple structured extraction models.
