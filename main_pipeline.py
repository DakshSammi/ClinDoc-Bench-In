import sys
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import torch

# Add project root to sys.path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from configs import config
from models.model_factory import ModelFactory
from agents.raw_extraction.raw_extraction import RawExtractionAgent

# Configure logging to be less noisy
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SpeedRun")

async def process_single(image_path: Path, rel_path: Path, agent: RawExtractionAgent):
    try:
        image = Image.open(image_path).convert("RGB")
        output = await agent.run(image)
        
        # Save output
        save_dir = config.OUTPUTS_DIR / "raw_extractions" / "Qwen_Final" / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{image_path.stem}.json"
        
        agent.save_output(str(output_path), output.model_dump())
        logger.info(f"SUCCESS: {rel_path}")
        return True
    except Exception as e:
        logger.error(f"FAILED: {rel_path} | Error: {str(e)[:100]}")
        return False

async def main():
    logger.info("Starting Qwen-Only Speed Run for Meeting Benchmarks...")
    
    # 1. Load Qwen Model ONLY
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    try:
        model_wrapper = ModelFactory.get_model(model_id, config)
    except Exception as e:
        logger.error(f"Critical error loading Qwen: {e}")
        return

    # Setup Agent with flexible schema prompt
    raw_prompt = """Extract all information from this medical prescription. 
Return a JSON object with:
- patient_info: {name, age, gender, weight}
- items: list of {medicine_name, dosage, frequency, duration, instructions}
- clinical_notes: any other findings or doctor's notes.
Strictly return ONLY valid JSON."""
    
    agent = RawExtractionAgent(model_wrapper, prompt_template=raw_prompt)

    # 3. Discover Images
    images_base = config.PRESCRIPTIONS_DIR
    pres_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
    prescriptions = []
    for ext in pres_extensions:
        prescriptions.extend(list(images_base.rglob(ext)))
    
    prescriptions.sort()
    logger.info(f"Found {len(prescriptions)} prescriptions. Starting processing...")

    success_count = 0
    for pres_path in prescriptions:
        rel_path = pres_path.relative_to(images_base)
        if await process_single(pres_path, rel_path, agent):
            success_count += 1
        
        # Prevent GPU memory buildup
        if success_count % 5 == 0:
            torch.cuda.empty_cache()

    logger.info(f"Speed Run Complete! Processed {success_count}/{len(prescriptions)} successfully.")
    logger.info(f"Results are in: {config.OUTPUTS_DIR}/raw_extractions/Qwen_Final")

if __name__ == "__main__":
    asyncio.run(main())
