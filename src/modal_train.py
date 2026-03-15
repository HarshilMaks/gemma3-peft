#!/usr/bin/env python3
"""
Ghost Architect — Modal Serverless Training
Full Trinity stack: QLoRA + DoRA + rsLoRA on A10G (24GB VRAM).

DoRA works on A10G because:
  - A10G supports bfloat16 → Unsloth doesn't need its fp16 Gemma3 attention hack
  - We also include the DoRA dtype patch as a safety net

SETUP (one time):
  pip install modal
  modal setup                          # creates ~/.modal.toml, links to your account
  modal secret create ghost-architect-secrets HF_TOKEN=hf_xxxx

UPLOAD DATASET (one time, or when dataset changes):
  modal run src/modal_train.py::upload_dataset

RUN TRAINING (~1.5 hrs on A10G, ~$1.65 from your $30 credits):
  modal run src/modal_train.py

DOWNLOAD ADAPTER (after training):
  modal run src/modal_train.py::download_adapter
"""

import modal
from pathlib import Path

# ── Persistent Volumes ──────────────────────────────────────────────────────
# Volumes persist across Modal runs.
# model_cache_vol saves the 12GB Gemma weights — avoids re-downloading every run.
# dataset_vol holds merged training assets (dataset_merged.json + screenshots).
# output_vol holds the trained LoRA adapter.

dataset_vol     = modal.Volume.from_name("ghost-architect-dataset",    create_if_missing=True)
model_cache_vol = modal.Volume.from_name("ghost-architect-hf-cache",   create_if_missing=True)
output_vol      = modal.Volume.from_name("ghost-architect-output",     create_if_missing=True)

# Paths INSIDE the container (where volumes are mounted)
DATASET_PATH = Path("/dataset")
CACHE_PATH   = Path("/hf-cache")
OUTPUT_PATH  = Path("/output")

# ── Container Image ─────────────────────────────────────────────────────────
# Modal builds this once and caches it. Only rebuilds when pip_install changes.
# Using CUDA 12.1 base which matches Unsloth's requirements.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "curl")
    .pip_install(
        # Core ML stack — pin torch first so Unsloth picks the right CUDA variant
        "torch==2.4.1",
        "torchvision==0.19.1",
        # Unsloth from source — gets latest Gemma3 patches
        "unsloth @ git+https://github.com/unslothai/unsloth.git",
        "unsloth_zoo",
        # Dependencies
        "transformers>=4.46.0",
        "trl>=0.11.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.44.0",
        "xformers",
        "datasets>=3.0.0",
        "huggingface_hub>=0.25.0",
        "pillow>=10.0.0",
        "triton",
        "sentencepiece",
    )
)

app = modal.App("ghost-architect-training", image=image)


# ── DoRA Dtype Patch ─────────────────────────────────────────────────────────
# PEFT's dora.py passes x_eye in the input's dtype to lora_A (fp32) without
# casting first. Regular LoRA does `x = x.to(self.lora_A.weight.dtype)` first.
# This patch adds the missing cast. Applied at runtime inside the container.
def _patch_dora():
    import peft.tuners.lora.dora as _dora_mod
    _dora_path = _dora_mod.__file__
    with open(_dora_path, "r") as f:
        src = f.read()
    old = "        lora_weight = lora_B(lora_A(x_eye)).T"
    new = (
        "        x_eye = x_eye.to(next(lora_A.parameters()).dtype)"
        "  # cast to match lora_A weights\n"
        "        lora_weight = lora_B(lora_A(x_eye)).T"
    )
    if old in src:
        with open(_dora_path, "w") as f:
            f.write(src.replace(old, new, 1))
        print("✅ DoRA dtype patch applied")
    else:
        print("ℹ️  DoRA patch: already applied or PEFT version changed — skipping")


# ── Dataset Loader ────────────────────────────────────────────────────────────
def _load_dataset(dataset_json: Path):
    import json
    import logging
    from datasets import Dataset

    log = logging.getLogger(__name__)
    parsed = json.loads(dataset_json.read_text())
    valid, skipped = [], 0

    for item in parsed:
        # image_path in JSON was written as data/ui_screenshots/xxx.png (relative)
        # Inside the container it lives at /dataset/ui_screenshots/xxx.png
        # BUT: synthetic data has paths like data/synthetic_factory/screenshots/ui_0000.png
        # Fix: Extract filename and look in /dataset/ui_screenshots/ (where both are uploaded)
        raw_path = item.get("image_path") or item.get("image", "")
        filename = Path(raw_path).name
        img_path = DATASET_PATH / "ui_screenshots" / filename

        if not img_path.exists():
            skipped += 1
            continue

        # Image path string embedded in messages — no top-level 'images' column.
        # UnslothVisionDataCollator falls back to process_vision_info(messages)
        # which calls fetch_image() → Image.open(path) for local paths.
        # Consistent Arrow schema: all content items have {type, image, text}.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path), "text": ""},
                    {"type": "text",  "image": "",            "text": item["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "image": "", "text": item["output"]},
                ],
            },
        ]
        valid.append({"messages": messages})

    log.info(f"✅ Dataset: {len(valid)} valid, {skipped} skipped")
    return Dataset.from_list(valid)


# ── Main Training Function ────────────────────────────────────────────────────
@app.function(
    gpu="A10G",          # 24GB VRAM — enough for full Trinity including DoRA
    timeout=21600,       # 6 hour cap for full merged-dataset 3-epoch run
    volumes={
        DATASET_PATH: dataset_vol,
        CACHE_PATH:   model_cache_vol,
        OUTPUT_PATH:  output_vol,
    },
    secrets=[modal.Secret.from_name("ghost-architect-secrets")],
)
def train(dataset_filename: str = "dataset_merged.json"):
    import os
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Point HuggingFace cache at the persistent volume
    os.environ["HF_HOME"]             = str(CACHE_PATH)
    os.environ["TRANSFORMERS_CACHE"]  = str(CACHE_PATH / "transformers")
    os.environ["HF_DATASETS_CACHE"]   = str(CACHE_PATH / "datasets")

    # Auth
    from huggingface_hub import login
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token, add_to_git_credential=False)
    print("✅ HuggingFace authenticated")

    # Apply DoRA patch
    _patch_dora()

    from unsloth import FastVisionModel, is_bfloat16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    dataset_json = DATASET_PATH / dataset_filename
    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset {dataset_filename} not found in Modal volume. Run upload_dataset first.")
        
    dataset = _load_dataset(dataset_json)

    print("Loading Gemma-3-12B-IT vision model...")
    model, processor = FastVisionModel.from_pretrained(
        model_name="google/gemma-3-12b-it",
        load_in_4bit=True,   # QLoRA: 12B fits in 24GB with room for DoRA gradients
    )

    # Full Trinity on A10G:
    # - finetune_vision_layers=True  → A10G has 24GB; safe to adapt SigLIP too
    # - use_dora=True               → DoRA dtype bug patched above
    # - use_rslora=True             → stabilizes rank 64 training
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,      # A10G can handle vision adapter too
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=64,
        lora_alpha=32,
        lora_dropout=0,          # 0 = Unsloth fast-patches ALL layers
        bias="none",
        random_state=42,
        use_rslora=True,         # Stabilizes rank 64 — prevents gradient collapse
        use_dora=True,           # Weight decomposition: magnitude + direction
    )
    print("✅ Model loaded with full Trinity (QLoRA + DoRA + rsLoRA)")

    output_dir = str(OUTPUT_PATH / "adapters" / "trinity_a10g")

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,   # Effective batch = 8
            num_train_epochs=3,              # 3 passes over full merged dataset
            learning_rate=2e-4,
            lr_scheduler_type="cosine",      # Cosine decay (better than linear for 3 epochs)
            warmup_ratio=0.1,
            optim="adamw_8bit",
            save_strategy="epoch",
            logging_steps=10,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),    # A10G supports bf16 → avoids fp16 hacks
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            seed=42,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=4096,             # A10G can handle full 4096 context
        ),
    )

    print(f"🚀 Starting training — 3 epochs × {len(dataset)} examples...")
    trainer.train()

    print(f"✅ Training complete! Saving adapter to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Commit writes to the persistent volume — required for output_vol
    output_vol.commit()
    print(f"✅ Adapter committed to Modal Volume at {output_dir}")
    return output_dir


# ── Upload Dataset ────────────────────────────────────────────────────────────
@app.local_entrypoint()
def upload_dataset(dataset_filename: str = "dataset_merged.json"):
    """
    Upload dataset JSON + ui_screenshots/ + synthetic_factory/ to Modal.
    Run once (or when dataset changes):
      modal run src/modal_train.py::upload_dataset
    """
    import os

    local_json = Path("data") / dataset_filename
    
    # Check both root data/ and synthetic_factory/ subfolder
    if not local_json.exists():
        local_json = Path("data/synthetic_factory/synthetic_dataset.json")

    local_screenshots = Path("data/ui_screenshots")
    local_synthetic_screenshots = Path("data/synthetic_factory/screenshots")

    assert local_json.exists(), f"Missing {local_json}"

    print(f"Uploading {dataset_filename}...")
    with dataset_vol.batch_upload() as batch:
        batch.put_file(local_json, dataset_filename)
        
        # Upload real screenshots if they exist
        if local_screenshots.exists():
            batch.put_directory(local_screenshots, "ui_screenshots")
            
        # Upload synthetic screenshots if they exist
        if local_synthetic_screenshots.exists():
            batch.put_directory(local_synthetic_screenshots, "ui_screenshots")

    print(f"✅ Uploaded dataset JSON and all images to ghost-architect-dataset volume")


# ── Download Adapter ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def download_adapter(adapter_name: str = "trinity_a10g"):
    """
    Download the trained adapter from Modal output volume to local output/.
    Run after training:  modal run src/modal_train.py::download_adapter
    """
    local_out = Path("output/adapters") / adapter_name
    local_out.mkdir(parents=True, exist_ok=True)

    print(f"Listing adapter files in Modal output volume for {adapter_name}...")
    adapter_prefix = f"adapters/{adapter_name}"
    files = list(output_vol.listdir(adapter_prefix, recursive=True))

    if not files:
        print(f"❌ No adapter files found for {adapter_name}. Has training completed?")
        return

    print(f"Downloading {len(files)} files...")
    for entry in files:
        dest = local_out / Path(entry.path).relative_to(adapter_prefix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with output_vol.read_file(entry.path) as f:
            dest.write_bytes(f.read())

    print(f"✅ Adapter downloaded to {local_out}/")


# ── Run Training ─────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(dataset_filename: str = "dataset_merged.json"):
    """
    Default entrypoint: 
      modal run src/modal_train.py
    """
    result = train.remote(dataset_filename=dataset_filename)
    print(f"\n🎉 Training complete!")
    print(f"   Adapter is in Modal Volume at: {result}")
    print(f"   Download it: modal run src/modal_train.py::download_adapter --adapter-name trinity_a10g")
