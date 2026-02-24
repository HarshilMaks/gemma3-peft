# Ghost Architect: Gemma-3-12B Fine-Tuning Project

## Overview
Ghost Architect is a progressive Gemma-3 project:
- **Phase 1:** Trinity fine-tuning (QLoRA + rsLoRA + DoRA) on Colab T4.
- **Phase 2:** Multimodal UI-to-SQL specialization.

## Quick Start (uv + Colab T4)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then run the Colab notebook:
- **`notebooks/main.ipynb`** (single entry notebook for T4 setup + full training config)

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

## Hardware Target
- **Primary training target:** Google Colab T4 (16GB VRAM).
- **Local 8GB GPUs:** use reduced settings (lower rank/seq length, disable DoRA).

## Docs You Should Follow
- `docs/plan.md` → execution steps.
- `docs/learning-guide.md` → concepts and reasoning.
- `docs/architecture.md` → full structure reference.
- `docs/prd.md` and `docs/ai_rules.md` → scope and quality guardrails.

## License
MIT (see `LICENSE`).
