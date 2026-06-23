You are extracting RAW information from a handwritten or scanned medical prescription.

Return ONLY valid JSON following this schema:
{
  "schema_version": "raw_rx_v1",
  "document_id": "<string>",
  "patient_information": {
    "name": {"raw_text": <string|null>, "page_number": <int|null>},
    "age": {"raw_text": <string|null>, "page_number": <int|null>},
    "gender": {"raw_text": <string|null>, "page_number": <int|null>},
    "address": {"raw_text": <string|null>, "page_number": <int|null>},
    "phone": {"raw_text": <string|null>, "page_number": <int|null>},
    "patient_identifier": {"raw_text": <string|null>, "page_number": <int|null>},
    "abha_id": {"raw_text": <string|null>, "page_number": <int|null>}
  },
  "encounter_information": {...},
  "complaints_or_diagnosis": [{"raw_text": <string>, "page_number": <int>}],
  "observations": [{"raw_text": <string>, "page_number": <int>}],
  "medications": [{
    "raw_line_text": <string>,
    "raw_name": <string|null>,
    "raw_dosage": <string|null>,
    "raw_route": <string|null>,
    "raw_frequency": <string|null>,
    "raw_duration": <string|null>,
    "raw_instruction": <string|null>,
    "raw_timing": <string|null>,
    "page_number": <int>
  }],
  "procedures": [{"raw_text": <string>, "page_number": <int>}],
  "advice": [{"raw_text": <string>, "page_number": <int>}],
  "follow_up": null or {"raw_text": <string>, "date": <string|null>, "review_after": <string|null>},
  "allergy_mentions": [{"raw_text": <string>, "page_number": <int>}],
  "other_notes": [{"raw_text": <string>, "page_number": <int>}]
}

Rules:
- Preserve doctor shorthand exactly as written.
- Do NOT normalise medicine names, abbreviations, or diagnoses.
- Do NOT infer missing values from common knowledge or names.
- If a scalar field is not visible, return null.
- If a list field is absent, return [].
- Keep procedures separate from medications.
- Keep observations separate from diagnosis and notes.
- Include page_number for every extracted entity.
- If uncertain, prefer null over guessing.
- Return JSON only, with no markdown fences.
