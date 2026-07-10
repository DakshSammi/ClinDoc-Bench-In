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
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
from huggingface_hub import InferenceClient
from google import genai
import requests
from PIL import Image
import io
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline, AutoModelForImageTextToText
from accelerate import Accelerator

class BaseModelWrapper:
    def __init__(self, model_id: str):
        self.model_id = model_id

    async def generate(self, prompt: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        raise NotImplementedError

class LocalHuggingFaceWrapper(BaseModelWrapper):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine model class based on ID
        if "qwen2-vl" in model_id.lower() or "florence" in model_id.lower():
            model_class = AutoModelForImageTextToText
        else:
            model_class = AutoModelForCausalLM

        # Moondream2 and Phi-3 do not yet support SDPA (Flash Attention)
        is_moondream = "moondream" in model_id.lower()
        is_phi3 = "phi-3" in model_id.lower()
        attn_impl = "eager" if (is_moondream or is_phi3) else "sdpa"
        
        # Use device_map="auto" to automatically distribute across 4 GPUs
        self.model = model_class.from_pretrained(
            model_id, 
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Moondream fix: ensure pad token is set
        if is_moondream:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    async def generate(self, prompt: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        start_time = time.time()
        try:
            # Qwen2-VL specific handling for image tokens
            if "qwen2-vl" in self.model_id.lower() and image:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Note: Qwen2-VL processor.apply_chat_template handles the <|image_pad|> insertion
                inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            elif image:
                inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
            else:
                inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
            # For multimodal models, we need to extract only the generated part
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            content = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            return {
                "content": content,
                "latency": (time.time() - start_time) * 1000
            }
        except Exception as e:
            return {"error": str(e), "latency": (time.time() - start_time) * 1000}

class HuggingFaceWrapper(BaseModelWrapper):
    def __init__(self, model_id: str, token: str):
        super().__init__(model_id)
        self.client = InferenceClient(model=model_id, token=token)

    async def generate(self, prompt: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if image:
                # Use raw requests to bypass InferenceClient version issues
                url = f"https://api-inference.huggingface.co/models/{self.model_id}"
                headers = {"Authorization": f"Bearer {self.client.token}"}
                
                # Some models expect different payloads
                payload = {
                    "inputs": prompt,
                    "image": self._img_to_bytes(image).decode('latin1') # Simplified
                }
                
                # Fallback to direct task methods if request fails
                response = requests.post(url, headers=headers, data=self._img_to_bytes(image), params={"prompt": prompt})
                content = response.text
            else:
                response = self.client.text_generation(prompt, max_new_tokens=1024)
                content = response
            
            return {
                "content": content,
                "latency": (time.time() - start_time) * 1000
            }
        except Exception as e:
            return {"error": str(e), "latency": (time.time() - start_time) * 1000}

    def _img_to_bytes(self, image: Image.Image) -> bytes:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

class GeminiWrapper(BaseModelWrapper):
    def __init__(self, model_id: str, api_key: str):
        # The new SDK handles prefixes internally, but let's be explicit
        self.raw_model_id = model_id.split("/")[-1] if "/" in model_id else model_id
        super().__init__(self.raw_model_id)
        # Initialize client with generic config
        self.client = genai.Client(api_key=api_key)

    async def generate(self, prompt: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        start_time = time.time()
        try:
            # The new SDK often prefers [prompt, image] or [image, prompt]
            contents = [prompt]
            if image:
                contents.append(image)
            
            # Try to generate content
            response = self.client.models.generate_content(
                model=self.raw_model_id,
                contents=contents
            )
            
            return {
                "content": response.text,
                "latency": (time.time() - start_time) * 1000
            }
        except Exception as e:
            # Fallback: list models if 404 occurs to help debug
            if "404" in str(e):
                available_models = []
                try:
                    for m in self.client.models.list():
                        available_models.append(m.name)
                except:
                    pass
                return {"error": f"Model {self.raw_model_id} not found. Available: {available_models}. Error: {str(e)}", "latency": 0}
            return {"error": str(e), "latency": (time.time() - start_time) * 1000}

class SarvamWrapper(BaseModelWrapper):
    def __init__(self, api_key: str):
        super().__init__("sarvam-vision")
        self.api_key = api_key
        self.url = "https://api.sarvam.ai/v1/vision" # Placeholder URL

    async def generate(self, prompt: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        start_time = time.time()
        if not image:
            return {"error": "Sarvam Vision requires an image", "latency": 0}
        
        try:
            # Placeholder for Sarvam API call
            # headers = {"Authorization": f"Bearer {self.api_key}"}
            # files = {"image": self._img_to_bytes(image)}
            # response = requests.post(self.url, headers=headers, files=files, data={"prompt": prompt})
            
            # Simulated response for now
            return {
                "content": "{\"items\": []}", 
                "latency": (time.time() - start_time) * 1000,
                "note": "Sarvam API integration placeholder"
            }
        except Exception as e:
            return {"error": str(e), "latency": (time.time() - start_time) * 1000}

class ModelFactory:
    @staticmethod
    def get_model(model_id: str, configs: Any) -> BaseModelWrapper:
        if "gemini" in model_id.lower():
            return GeminiWrapper(model_id, configs.GOOGLE_API_KEY)
        elif "sarvam" in model_id.lower():
            return SarvamWrapper(configs.SARVAM_API_KEY)
        else:
            if configs.USE_LOCAL_MODELS:
                return LocalHuggingFaceWrapper(model_id)
            return HuggingFaceWrapper(model_id, configs.HF_TOKEN)
