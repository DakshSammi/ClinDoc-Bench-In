#!/bin/bash
set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HANDOFF="/Computational5/daksh/_gnn_/benchmark_outputs/server2_handoff_imports/extended_latest/stage1b_server2_ocr_handoff_for_server1_extended_latest/stage1b_server2_ocr_outputs_for_server1_extended.csv"
MANIFEST="data/full_benchmark_manifest.csv"

echo "Waiting for EasyOCR to finish..."
wait

echo "Running Surya..."
python3 scripts/run_stage1b_ocr_to_json_canonical.py \
  --handoff "$HANDOFF" \
  --manifest "$MANIFEST" \
  --output-root "/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_extended_surya_qwen3_full_$TIMESTAMP" \
  --lane "surya_qwen3|qwen3:8b|surya" \
  --smoke-docs "p4,p20,p25_1,p38_1,p42_1" \
  --report-stem "reports/stage1b_extended_surya_qwen3" \
  --report-title "Stage 1B Extended Surya+qwen3"

echo "Running Docling..."
python3 scripts/run_stage1b_ocr_to_json_canonical.py \
  --handoff "$HANDOFF" \
  --manifest "$MANIFEST" \
  --output-root "/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_extended_docling_qwen3_full_$TIMESTAMP" \
  --lane "docling_qwen3|qwen3:8b|docling" \
  --smoke-docs "p4,p20,p25_1,p38_1,p42_1" \
  --report-stem "reports/stage1b_extended_docling_qwen3" \
  --report-title "Stage 1B Extended Docling+qwen3"

echo "Running Marker (Partial)..."
python3 scripts/run_stage1b_ocr_to_json_canonical.py \
  --handoff "$HANDOFF" \
  --manifest "$MANIFEST" \
  --output-root "/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_extended_marker_qwen3_partial_$TIMESTAMP" \
  --lane "marker_qwen3|qwen3:8b|marker" \
  --smoke-docs "p4,p20,p25_1,p38_1,p42_1" \
  --report-stem "reports/stage1b_extended_marker_qwen3_partial" \
  --report-title "Stage 1B Extended Marker+qwen3"
