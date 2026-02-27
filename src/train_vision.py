#!/usr/bin/env python3
"""
Production-Grade Gemma-3 Vision Fine-Tuning for UI-to-SQL
Includes Lazy-Loading to prevent RAM crashes on massive datasets.
"""

import argparse
import json
import torch
import os
from pathlib import Path
import logging

# Unsloth MUST be imported FIRST
from unsloth import FastVisionModel

from transformers import TrainingArguments
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer
from datasets import Dataset, Sequence, Image as DatasetsImage

from dotenv import load_dotenv
from huggingface_hub import login

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- AUTHENTICATION ---
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    logger.info("✅ HuggingFace authenticated")
else:
    raise ValueError("HUGGINGFACE_TOKEN not in .env")

# --- DATASET LAZY LOADING ---
def load_vision_dataset(dataset_path: Path) -> Dataset:
    logger.info(f"Loading dataset from {dataset_path}...")
    
    raw = dataset_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw)
    
    valid_rows = []
    skipped = 0
    
    for item in parsed:
        img_path = Path(item.get("image_path") or item.get("image", ""))
        
        if not img_path.exists():
            skipped += 1
            continue
        
        # Format for Gemma-3
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": item["instruction"]}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["output"]}]
            }
        ]
        
        valid_rows.append({
            "messages": messages,
            # CRITICAL FIX: Pass the string path, NOT the loaded PIL Image
            "images": [str(img_path)]
        })
    
    logger.info(f"✅ Loaded {len(valid_rows)} valid examples (skipped {skipped})")
    
    # Create the dataset
    ds = Dataset.from_list(valid_rows)
    
    # CRITICAL FIX: Cast the list column to Sequence(Image) for lazy per-batch decoding.
    # DatasetsImage() alone fails on list columns — Sequence wraps it correctly.
    # HuggingFace will store paths and decode to PIL only when a batch is accessed.
    ds = ds.cast_column("images", Sequence(DatasetsImage()))
    
    return ds

# --- TRINITY ARCHITECTURE ---
def create_vision_model():
    logger.info("Loading Gemma-3 vision model via Unsloth...")

    model, processor = FastVisionModel.from_pretrained(
        model_name="google/gemma-3-12b-it",
        load_in_4bit=True,
        use_bnb_4bit_compute_dtype="float16",
    )

    # FastVisionModel.get_peft_model (not peft.get_peft_model) — required so
    # Unsloth's custom CUDA kernels and memory optimizations apply correctly.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=64,           # High rank, stabilized by rsLoRA
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        random_state=42,
        use_rslora=True,    # Rank-stabilized (prevents gradient collapse at r=64)
        use_dora=True,      # Weight decomposition (magnitude + direction)
    )

    logger.info(f"✅ Model loaded with Trinity adapters")
    return model, processor

# --- TRAINING LOOP ---
def train_vision_model(dataset_path: Path, output_dir: str):
    dataset = load_vision_dataset(dataset_path)
    model, processor = create_vision_model()
    
    # is_bfloat16_supported picks fp16 vs bf16 automatically based on GPU
    from unsloth import is_bfloat16_supported

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=25,
        logging_steps=5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
        remove_unused_columns=False, # Required for vision models
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # UnslothVisionDataCollator handles chat template application + image
        # processing per-batch. Without this, SFTTrainer cannot handle vision inputs.
        data_collator=UnslothVisionDataCollator(model, processor),
        processing_class=processor,
    )
    
    logger.info("Starting production training...")
    trainer.train()
    
    logger.info(f"✅ Training complete! Saving to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
    parser.add_argument("--output", type=str, default="output/adapters/vision_trinity")
    args = parser.parse_args()
    
    train_vision_model(args.dataset, args.output)