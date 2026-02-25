#!/usr/bin/env python3
"""Phase 2 Multimodal Trinity training entrypoint for Ghost Architect."""

import argparse
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import TrainingArguments, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def load_multimodal_dataset(dataset_path: Path):
    """Loads dataset and verifies images exist."""
    from datasets import Dataset
    
    raw = dataset_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw)
    
    valid_rows = []
    for item in parsed:
        img_path = Path(item["image_path"])
        if img_path.exists():
            # Format specifically for Gemma-3 Vision Chat Template
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
                "images": [Image.open(img_path).convert("RGB")]
            })
            
    print(f"Loaded {len(valid_rows)} valid multimodal examples.")
    return Dataset.from_list(valid_rows)

def run_vision_training(dataset_path: Path):
    model_id = "google/gemma-3-12b-it" # Or whichever model variant you are using
    
    print("Loading Multimodal Processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    print("Loading Base Vision Model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    
    print("Applying Full Trinity Architecture (QLoRA + DoRA + rsLoRA)...")
    peft_config = LoraConfig(
        r=64, # High rank for maximum capability
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Target all layers
        use_dora=True,     # <--- TRINITY: Weight-Decomposed LoRA (Precision)
        use_rslora=True,   # <--- TRINITY: Rank-Stabilized LoRA (Stability)
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    dataset = load_multimodal_dataset(dataset_path)
    
    training_args = TrainingArguments(
        output_dir="output/vision_adapters",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True, # <--- TRINITY MEMORY SAVER (Required for DoRA on T4)
        learning_rate=2e-4,
        max_steps=100,
        logging_steps=5,
        save_steps=50,
        optim="adamw_8bit",
        fp16=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    
    print("Starting Multimodal Training on T4 GPU...")
    trainer.train()
    
    print("Saving Ghost Architect Vision Adapters...")
    trainer.save_model("output/vision_adapters")
    processor.save_pretrained("output/vision_adapters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    run_vision_training(args.dataset)