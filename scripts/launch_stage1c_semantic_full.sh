#!/usr/bin/env bash
set -euo pipefail

ROOT=/Computational5/daksh/_gnn_/Daksh/prescription_pipeline
TS=$(date +%Y%m%d_%H%M%S)
OUT=/Computational5/daksh/_gnn_/benchmark_outputs/stage1c_server1_semantic_${TS}
LOG=${ROOT}/logs/stage1c_semantic_full_${TS}.log
PIDFILE=${ROOT}/logs/stage1c_semantic_full_${TS}.pid

cd "$ROOT"
nohup bash -lc "
python scripts/run_stage1c_semantic_enrichment.py \
  --source 'glm_ocr_qwen3_8b|/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_server1_glm_ocr_qwen3_canonical_full_20260619_124659|ocr_glm_ocr_qwen3_8b_full' \
  --source 'trocr_qwen3_8b|/Computational5/daksh/_gnn_/benchmark_outputs/stage1b_server2_doctr_trocr_qwen3_canonical_full_20260619_125803|ocr_trocr_qwen3_8b_full' \
  --output-root '$OUT' \
  --model qwen3:8b \
  --timeout 240 \
  --resume \
  --report-stem stage1c_semantic_full \
  --report-title 'Stage 1C Semantic Full' && \
python scripts/benchmark_semantic_outputs.py \
  --output-root '$OUT' \
  --report-prefix stage1c_semantic
" > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
printf 'PID=%s\nOUT=%s\nLOG=%s\nPIDFILE=%s\n' "$PID" "$OUT" "$LOG" "$PIDFILE"

