# Ghost Architect: Gemma-3-12B Fine-Tuning Project

## Overview
This is a multimodal Gemma-3 fine-tuning project that converts UI screenshots into PostgreSQL database schemas.

- **Phase 1 (Foundation):** Trinity text fine-tuning on Colab T4 using **QLoRA + rsLoRA + DoRA**.
- **Phase 2 (Vision):** Multimodal UI-to-SQL training on 287 screenshot–schema pairs, with two training paths (Modal A10G and Colab T4).

The pipeline goes: **train → test → export GGUF → run with Ollama**, plus a Streamlit app (`src/app.py`) for interactive demos.

## Trinity Architecture

The training stack combines three methods for high-quality under tight VRAM limits:

1. **QLoRA (4-bit NF4 quantization)** — Compresses model weights so Gemma-3-12B fits on consumer GPUs.
2. **rsLoRA (rank-stabilized scaling)** — Stabilizes high-rank adaptation and enables rank 64.
3. **DoRA (weight decomposition)** — Improves update precision by separating magnitude and direction.

### Two Training Paths

| | **Modal A10G** (`src/modal_train.py`) | **Colab T4** (`src/train_vision.py`) |
|---|---|---|
| **Trinity** | Full: QLoRA + DoRA + rsLoRA | QLoRA + rsLoRA only (no DoRA) |
| **Vision layers** | `finetune_vision_layers=True` | `finetune_vision_layers=False` |
| **Epochs / Context** | 3 epochs, 4096 ctx | 1 epoch, 2048 ctx |
| **Cost** | ~$1.65 (from free $30 credits) | Free |

> **DoRA bug on T4:** PEFT's `dora.py` passes fp16 `x_eye` to fp32 `lora_A` without casting. Unsloth's Gemma3 temporary fp16 patch triggers this. `modal_train.py` includes an inline monkey-patch; the Colab path simply disables DoRA.

### Default Config (`configs/training_config.yaml`)
- **Model:** `unsloth/gemma-3-12b-it-bnb-4bit`
- **Sequence length:** `4096`
- **LoRA rank:** `64`, **alpha:** `32`
- **Target modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Batch size:** `1`, **Gradient accumulation:** `4`
- **Learning rate:** `2e-4`, **Optimizer:** `adamw_8bit`
- **Max steps:** `60`

### OOM Recovery Protocol
Apply in order if CUDA OOM occurs:
1. Reduce `max_seq_length`: `4096 → 2048`
2. Reduce LoRA rank: `64 → 32`
3. Disable DoRA: `use_dora: false.`
4. Reduce target modules to `["q_proj", "v_proj"]`

## Quick Start

### 1) Local setup (project/dev workflow)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
make help
```

### 2) Colab T4 (free vision training)
Use **` notebooks/main.ipynb`** — single notebook for runtime checks, dependency install, config, training, and GGUF export.

> CUDA-specific training dependencies (Unsloth/xformers) are installed inside the Colab notebook cells.

### 3) Modal A10G (full Trinity vision training)
```bash
modal run src/modal_train.py
```

### 4) Interactive demo
```bash
streamlit run src/app.py          # Upload screenshot → see schema
python src/inference.py            # CLI testing with rich output
```

## Make Targets
```bash
make venv           # create .venv
make install        # install dependencies with uv
make validate       # environment/GPU validation
make dataset-check  # validate data/dataset.json
make train          # run src/train.py with config + dataset
make export         # run src/export.py (GGUF export for Ollama)
make test           # run pytest
make clean          # remove Python cache files
```

## Dataset

### Phase 1 (text)
`data/dataset.json` — JSON array of instruction-tuning examples:
```json
[
  {
    "instruction": "Task description",
    "input": "Optional context",
    "output": "Expected response"
  }
]
```

### Phase 2 (vision)
`data/dataset_vision.json` — 287 vision training examples. Image paths are embedded in messages as `{"type": "image", "image": "<path>", "text": ""}` (no top-level `images` column).

## Project Tree
```text
ghost_architect_gemma3/
├── configs/
│   └── training_config.yaml        # Phase 1 config (used by Makefile + src/train.py)
├── scripts/
│   ├── build_vision_dataset.py     # Builds dataset_vision.json from screenshots + Gemini
│   ├── download_datasets.py        # Playwright scraper for UI screenshots
│   ├── generate_training_data.py   # Generates Phase 1 starter data
│   ├── validate_dataset.py         # Validates dataset.json (make dataset-check)
│   └── validate_environment.py     # Validates GPU/deps (make validate)
├── src/
│   ├── __init__.py
│   ├── modal_train.py              # Modal A10G training (full Trinity)
│   ├── train_vision.py             # Colab T4 vision training (QLoRA+rsLoRA)
│   ├── train.py                    # Phase 1 text training
│   ├── inference.py                # CLI testing with rich terminal output
│   ├── app.py                      # Streamlit web app (upload screenshot → schema)
│   ├── export.py                   # GGUF export for Ollama
│   └── synthetic_generator.py      # Gemini API for SQL generation from screenshots
├── data/
│   ├── dataset.json                # Phase 1 training data
│   ├── dataset_vision.json         # 287 vision training examples
│   ├── ui_screenshots/             # 287 PNGs
│   ├── raw_csvs/                   # Source CSVs for scraper
│   ├── synthetic_pairs/            # (empty, for future use)
│   └── validation_set/             # (empty, for future use)
├── tests/
│   └── __init__.py                 # Test package (no tests yet)
├── notebooks/
│   └── main.ipynb                  # Colab T4 notebook
├── docs/                           # Documentation
├── docker/                         # Docker setup (future)
├── output/                         # Generated adapters + GGUF (gitignored)
├── Makefile
├── requirements.txt
├── README.md
├── DATASET_README.md
├── SECURITY.md
└── LICENSE
```

## Documentation Map
- `docs/plan.md` — execution phases and implementation sequence
- `docs/learning-guide.md` — conceptual deep dive and training intuition
- `docs/architecture.md` — system architecture
- `docs/phase1_trinity_training.md` — Phase 1 readiness checklist
- `docs/phase2_vision_training.md` — Phase 2 vision training guide
- `docs/phase3_deployment.md` — deployment via GGUF/Ollama + Streamlit
- `docs/prd.md` — product boundaries and requirements
- `docs/ai_rules.md` — quality and development guardrails

## Current Status
- **Phase 1** (text fine-tuning): Complete. `src/train.py` + `configs/training_config.yaml` + 30-example starter dataset.
- **Phase 2** (vision training): Built. Two training scripts (`src/modal_train.py` for Modal A10G, `src/train_vision.py` for Colab T4), 287-example vision dataset, Streamlit demo app, CLI inference.
- **Export**: `src/export.py` converts adapters to GGUF for Ollama.
- **No API layer** — project scope is train → test → export GGUF → run locally.

## License
MIT (see `LICENSE.md`).
