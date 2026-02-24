# Ghost Architect: Gemma-3-12B Fine-Tuning Project

## Overview
Ghost Architect is a progressive Gemma-3 engineering project with two stages:

- **Phase 1 (Foundation):** Trinity fine-tuning on Colab T4 using **QLoRA + rsLoRA + DoRA**.
- **Phase 2 (Specialization):** Multimodal UI-to-SQL generation for database schema reverse engineering.

The goal is to build a reproducible training pipeline first, then extend it into a production-oriented Ghost Architect system.

## Trinity Architecture (Phase 1)
The Phase 1 training stack combines three methods for high quality under tight VRAM limits:

1. **QLoRA (4-bit NF4 quantization)**  
   Compresses model weights so Gemma-3-12B can train on 16GB T4 hardware.
2. **rsLoRA (rank-stabilized scaling)**  
   Stabilizes high-rank adaptation and enables rank 64 without typical divergence.
3. **DoRA (weight decomposition)**  
   Improves update precision by separating magnitude and direction.

### Default T4 Profile (`configs/training_config.yaml`)
- **Model:** `unsloth/gemma-3-12b-it-bnb-4bit`
- **Sequence length:** `4096`
- **LoRA rank:** `64`
- **LoRA alpha:** `32`
- **Target modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Batch size:** `1` (required on T4 for 12B)
- **Gradient accumulation:** `4`
- **Learning rate:** `2e-4`
- **Optimizer:** `adamw_8bit`
- **Max steps:** `60`

### Memory Budget (T4, 16GB)
- Model weights (4-bit): ~7.6 GB
- Gradients (rank 64 + DoRA): ~5.5 GB
- Context overhead (4096): ~2.5 GB
- System buffer: ~0.4 GB  
**Total:** ~15.6 GB (near-capacity training)

### OOM Recovery Protocol
Apply in order if CUDA OOM occurs:
1. Reduce `max_seq_length`: `4096 -> 2048`
2. Reduce LoRA rank: `64 -> 32`
3. Disable DoRA: `use_dora: false`
4. Reduce target modules to `["q_proj", "v_proj"]`

## Quick Start

### 1) Local setup (project/dev workflow)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
make help
```

### 2) Colab T4 setup (training workflow)
Use:
- **`notebooks/main.ipynb`** — single entry notebook for Colab T4 runtime checks, dependency install, config setup, training launch, and GGUF export.

> Note: CUDA-specific training dependencies (Unsloth/xformers) are installed inside the Colab notebook cells.

## Make Targets
```bash
make venv           # create .venv
make install        # install dependencies with uv
make validate       # environment/GPU validation
make dataset-check  # validate data/dataset.json
make train          # run src/train.py with config + dataset
make export         # run src/export.py
make test           # run pytest
make clean          # remove Python cache files
```

## Dataset Requirements
`data/dataset.json` should be a JSON array of instruction-tuning examples:

```json
[
  {
    "instruction": "Task description",
    "input": "Optional context",
    "output": "Expected response"
  }
]
```

Recommended minimum for first runs: **50+ high-quality examples**.

## Project Tree
```text
ghost_architect_gemma3/
├── docs/                     # Plan, architecture, PRD, AI rules, learning guide
├── notebooks/
│   └── main.ipynb            # Colab T4 main workflow
├── configs/
│   ├── training_config.yaml
│   ├── model_config.yaml
│   └── deployment_config.yaml
├── data/
│   ├── dataset.json
│   ├── ui_screenshots/
│   ├── synthetic_pairs/
│   └── validation_set/
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── data_processing.py
│   ├── export.py
│   ├── multimodal_model.py
│   ├── synthetic_generator.py
│   ├── models/
│   ├── training/
│   ├── data/
│   └── api/
├── scripts/                  # setup/export/deploy helpers
├── tests/
├── docker/
├── .github/workflows/
├── output/                   # adapters/checkpoints/gguf
├── requirements.txt
└── LICENSE
```

## Documentation Map
- `docs/plan.md` - execution phases and implementation sequence
- `docs/learning-guide.md` - conceptual deep dive and training intuition
- `docs/architecture.md` - end-to-end system architecture
- `docs/prd.md` - product boundaries and requirements
- `docs/ai_rules.md` - quality and development guardrails

## Current Repository Status
- Project structure and scaffolding are in place.
- Colab T4 main notebook is prepared as the primary training entrypoint.
- Trinity configuration is defined in `configs/training_config.yaml`.
- Core implementation work is ongoing for `src/train.py`, dataset population, and export wiring.

## License
MIT (see `LICENSE`).
