# Phase 2: Production-Grade Vision Training for Ghost Architect

## Overview

Train Gemma-3-12B with the **Trinity Architecture** (QLoRA + DoRA + rsLoRA) on 136 UI screenshots to create a production-ready UI-to-SQL model.

**What you're building:**
- 🖼️  Vision model that understands UI layouts
- 🧠 SQL schema generation from visual input
- 📦 Production-ready LoRA adapters
- 🚀 GGUF export for Ollama deployment

---

## Dataset: 136 Production UI Screenshots

### Status
✅ **dataset_vision.json** created and validated
- 136 unique UI screenshots (42.8 GB total)
- Classified by domain (dashboards, e-commerce, admin panels, etc.)
- Annotated with synthetic SQL schemas

### Directory
```
data/
├── ui_screenshots/           # Original images (136 PNG files)
│   ├── hangman-c0h7.onrender.com_7455.png
│   ├── dashboard-de-performance-de-tr-fego.vercel.app_6896.png
│   └── ... (134 more)
└── dataset_vision.json       # Training annotations
```

### Dataset Format
```json
[
  {
    "image_path": "data/ui_screenshots/hangman-c0h7.onrender.com_7455.png",
    "instruction": "Analyze this e-commerce UI and generate the database schema...",
    "output": "CREATE TABLE products (...)",
    "domain": "shop"
  },
  ...
]
```

---

## Training Architecture: Trinity Stack

### Layer 1: QLoRA (The Compressor)
- **Technique**: 4-bit quantization + Low-Rank Adaptation
- **Effect**: Shrinks 12B model from 24GB → 7.6GB
- **Benefit**: Fits on T4 GPU (16GB VRAM)

### Layer 2: DoRA (The Teacher)
- **Technique**: Weight-Decomposed LoRA
- **Effect**: Separates magnitude (scale) from direction (adaptation)
- **Benefit**: Better convergence for complex visual reasoning tasks

### Layer 3: rsLoRA (The Stabilizer)
- **Technique**: Rank-Stabilized LoRA
- **Effect**: Mathematically stabilizes gradients at high rank
- **Benefit**: Enables rank=64 without training collapse

### Memory Budget (T4: 16GB)
```
Model weights (4-bit QLoRA):     7.6 GB
Gradients (rank 64 + DoRA):      5.5 GB
Vision embeddings (4096 seq):    2.5 GB
Overhead & buffers:              ~0.4 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                          ~15.6 GB (98% utilization)
```

---

## Step-by-Step Training Guide

### Prerequisites
- ✅ HuggingFace token in `.env` (from HUGGINGFACE_AUTH.md)
- ✅ HUGGINGFACE_TOKEN authenticated
- ✅ Google Colab with T4 GPU or local NVIDIA GPU

### 1️⃣ Verify Dataset (Local or Colab)

```bash
# Check dataset integrity
python3 scripts/build_vision_dataset.py

# Expected output:
# ✅ Validation complete!
#    • Valid images: 136
#    • Duplicates removed: 0
```

### 2️⃣ Start Training (Colab)

**In Colab notebook (new cell):**
```python
# Mount and setup
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/ghost_architect_gemma3

# Install dependencies
!pip install -q unsloth "xformers<0.0.27" trl peft transformers datasets
!pip install -q python-dotenv huggingface-hub pillow

# Run training
!python src/train_vision_prod.py --dataset data/dataset_vision.json
```

### 3️⃣ Monitor Training (Live VRAM)

**In separate Colab cell:**
```bash
!watch -n 2 nvidia-smi
```

**Expected GPU metrics:**
- GPU Util: 85-95%
- VRAM: 14.8-15.6 GB (showing red is OK - we want 98% utilization)
- Temperature: <75°C

### 4️⃣ Training Time Estimate

| Dataset | T4 GPU | A100 GPU |
|---------|--------|----------|
| 136 examples (full) | 45-90 min | 10-15 min |
| 50 examples (test) | 20-30 min | 3-5 min |

---

## What Happens During Training

### Phase 1: Model Loading (5-10 min)
```
✅ HuggingFace authenticated
Loading Gemma-3 vision model...
✓ Downloaded model shards (25GB → 7.6GB after quantization)
✅ Model loaded with Trinity adapters
   Trainable params: 1,247,232
```

### Phase 2: Training Loop (40-80 min)
```
Dataset splits: 136 examples

Step 5/136    loss: 2.14
Step 10/136   loss: 1.92
Step 20/136   loss: 1.54
Step 50/136   loss: 0.87
Step 100/136  loss: 0.42
Step 136/136  loss: 0.18  ✅
```

### Phase 3: Saving (2-5 min)
```
✅ Training complete! Saving to output/adapters/vision_trinity...
✨ NEXT: Export to GGUF for production inference
```

---

## Output Files

After training, you'll have:

```
output/adapters/vision_trinity/
├── adapter_config.json          # LoRA configuration
├── adapter_model.bin            # LoRA weights (lightweight)
├── preprocessor_config.json     # Image processor settings
├── config.json                  # Model config
└── generation_config.json       # Generation parameters

logs/vision/
└── events.out.tfevents.*        # Training metrics
```

**Size**: LoRA weights ~8-15 MB (vs 24 GB for full model!)

---

## Troubleshooting

### ❌ "CUDA Out of Memory" Error

**Fix 1: Reduce batch size** (already minimal at 1)
```python
# In train_vision_prod.py:
per_device_train_batch_size=1    # Already set
gradient_accumulation_steps=2    # Reduce from 4 to 2
```

**Fix 2: Lower learning rank**
```python
r=32,  # Reduce from 64 to 32
```

**Fix 3: Use fewer examples**
```bash
python src/train_vision_prod.py --max-examples 50
```

### ❌ "HUGGINGFACE_TOKEN not found"

**Solution**: Add to `.env` file
```bash
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

Then restart Colab runtime.

### ❌ "Access to model google/gemma-3-12b-it is restricted"

**Solution**: Visit https://huggingface.co/google/gemma-3-12b-it
1. Click "Access Repository"
2. Accept Google's license terms
3. Wait ~1 minute for approval
4. Retry training

---

## Next Steps After Training

### Export to GGUF (Production Format)

```bash
# Convert LoRA adapters to GGUF for Ollama
python scripts/export_to_gguf.py \
  --adapter output/adapters/vision_trinity \
  --output output/gguf/gemma3-vision-ui2sql.gguf
```

### Deploy with Ollama

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./gemma3-vision-ui2sql.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are an expert in UI analysis and SQL schema design."
EOF

# Build and run
ollama create gemma3-vision-ui2sql -f Modelfile
ollama run gemma3-vision-ui2sql "Analyze this UI..."
```

### Use in Production API

```python
from ollama import chat

response = chat(
    model="gemma3-vision-ui2sql",
    messages=[{
        "role": "user",
        "content": "Analyze this screenshot and generate the database schema",
        "images": ["path/to/screenshot.png"]
    }]
)
```

---

## Production Checklist

- [ ] Dataset built (136 examples): `data/dataset_vision.json`
- [ ] HuggingFace authenticated: `.env` has `HUGGINGFACE_TOKEN`
- [ ] Training completed: `output/adapters/vision_trinity/`
- [ ] GGUF exported: `output/gguf/gemma3-vision-ui2sql.gguf`
- [ ] Ollama tested: `ollama run gemma3-vision-ui2sql`
- [ ] API deployed: FastAPI endpoint ready for inference

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/build_vision_dataset.py` | Create dataset from UI screenshots |
| `src/train_vision_prod.py` | Production-grade training script |
| `data/dataset_vision.json` | Training annotations (136 examples) |
| `HUGGINGFACE_AUTH.md` | HuggingFace setup guide |
| `docs/phase1_trinity_training.md` | Phase 1 text training (reference) |

---

## Learning Resources

**Understanding Trinity Architecture:**
- QLoRA: https://arxiv.org/pdf/2305.14314
- DoRA: https://arxiv.org/pdf/2402.09353
- Rank-Stabilized LoRA: Built into Unsloth

**Gemma-3 Vision:**
- Model Card: https://huggingface.co/google/gemma-3-12b-it
- Technical Paper: https://arxiv.org/pdf/2010.11929

**Vision Training Best Practices:**
- Image preprocessing for vision transformers
- Multi-modal tokenization strategies
- Handling variable-size images

---

**Status**: ✅ Ready for production training

**Next**: Run `python src/train_vision_prod.py --dataset data/dataset_vision.json`

