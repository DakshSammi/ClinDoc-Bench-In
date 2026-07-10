#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HANDOFF="/Computational5/daksh/_gnn_/benchmark_outputs/server2_handoff_imports/extended_latest/stage1b_server2_ocr_handoff_for_server1_extended_latest/stage1b_server2_ocr_outputs_for_server1_extended.csv"
MANIFEST="data/full_benchmark_manifest.csv"

echo "Running EasyOCR Full..."
python3 scripts/run_stage1b_ocr_to_json_canonical.py \
  --handoff "$HANDOFF" \
  --manifest "$MANIFEST" \
  --output-root "/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_extended_easyocr_qwen3_full_$TIMESTAMP" \
  --lane "easyocr_qwen3|qwen3:8b|easyocr" \
  --report-stem "stage1b_extended_easyocr_qwen3" \
  --report-title "Stage 1B Extended EasyOCR+qwen3"

echo "Running Surya Full..."
python3 scripts/run_stage1b_ocr_to_json_canonical.py \
  --handoff "$HANDOFF" \
  --manifest "$MANIFEST" \
  --output-root "/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_extended_surya_qwen3_full_$TIMESTAMP" \
  --lane "surya_qwen3|qwen3:8b|surya" \
  --report-stem "stage1b_extended_surya_qwen3" \
  --report-title "Stage 1B Extended Surya+qwen3"

echo "Running Docling Full..."
python3 scripts/run_stage1b_ocr_to_json_canonical.py \
  --handoff "$HANDOFF" \
  --manifest "$MANIFEST" \
  --output-root "/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_extended_docling_qwen3_full_$TIMESTAMP" \
  --lane "docling_qwen3|qwen3:8b|docling" \
  --report-stem "stage1b_extended_docling_qwen3" \
  --report-title "Stage 1B Extended Docling+qwen3"

echo "Running Marker Partial..."
python3 scripts/run_stage1b_ocr_to_json_canonical.py \
  --handoff "$HANDOFF" \
  --manifest "$MANIFEST" \
  --output-root "/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_extended_marker_qwen3_partial_$TIMESTAMP" \
  --lane "marker_qwen3|qwen3:8b|marker" \
  --report-stem "stage1b_extended_marker_qwen3_partial" \
  --report-title "Stage 1B Extended Marker+qwen3 Partial"

