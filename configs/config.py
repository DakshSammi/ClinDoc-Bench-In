# Copyright 2026 ClinDoc-Bench-IN contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from dotenv import load_dotenv

# Project Paths
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DAKSH_ROOT = PROJECT_ROOT.parent

# Load the project-local env first, then allow a pipeline-local env to fill gaps.
# This keeps API keys available even when commands are launched from the repo root.
load_dotenv(DAKSH_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=False)

PRESCRIPTIONS_DIR = PROJECT_ROOT / "prescriptions"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"

# Execution Mode
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "true").lower() == "true"
DEVICE = "cuda" if USE_LOCAL_MODELS else "cpu"

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
DATALAB_API_KEY = os.getenv("DATALAB_API_KEY")
ZAI_API_KEY = os.getenv("ZAI_API_KEY")
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY")
BIOPORTAL_API_URL = os.getenv("BIOPORTAL_API_URL", "https://data.bioontology.org/")
ABEROWL_API_URL = os.getenv("ABEROWL_API_URL", "https://aber-owl.net/aberowl/rest/labels?query=")

# Model Routing
MODELS_RAW_EXTRACTION = [
    "Qwen/Qwen2-VL-7B-Instruct",
    "microsoft/Florence-2-large",
    "vikhyatk/moondream2",
    "gemini-2.0-flash"
]

MODELS_SEMANTIC_NORMALIZATION = [
    "gemini-2.0-flash",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
]

MODELS_ONTOLOGY_MAPPING = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "gemini-1.5-flash"
]

# Logging
LOG_FILE = LOGS_DIR / "pipeline.log"
