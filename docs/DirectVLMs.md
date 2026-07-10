# Direct VLMs

Direct VLM lanes receive document images and emit canonical JSON directly.

## Requirements

- Preserve document order for multi-image encounters.
- Record provider, model version, prompt version, runtime, and failures.
- Validate canonical JSON before scoring.
- Use automatic retries and provider-specific quota handling for API models.

## Frozen Direct VLM Examples

The final registry includes Gemini, Qwen3, Qwen2.5-VL, and local/server-hosted direct VLM lanes. Use `final_model_registry.csv` for exact provenance and coverage.
