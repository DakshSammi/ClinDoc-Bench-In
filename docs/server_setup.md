# Server Setup

## Server 1

Server 1 is the local Ollama/VLM benchmark machine. Use `requirements/server1_ollama.txt` and `configs/servers/server1_4090_ollama.yaml`.

Use local Ollama models for final Server 1 rows. Do not start paid APIs or OpenRouter for final benchmark runs.

## Server 2

Server 2 is the OCR/vLLM benchmark machine. Use `requirements/server2_ocr.txt`, `requirements/server2_vllm.txt`, and `configs/servers/server2_rtx6000_vllm.yaml`.

Import only cleaned code and aggregate reports. Do not import raw OCR handoff text, images, annotations, logs, benchmark outputs, archives, or model weights.
