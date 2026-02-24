#!/usr/bin/env python3
"""Phase 1 Trinity training entrypoint for Gemma-3 (Colab T4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping/object: {path}")
    return data


def _format_example(example: dict[str, Any]) -> str:
    """Convert one JSON example to SFT text format."""
    if "text" in example:
        text = str(example["text"]).strip()
        if text:
            return text

    instruction = str(example.get("instruction", "")).strip()
    user_input = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()

    if not instruction or not output:
        raise ValueError(
            "Each example must contain either non-empty 'text' OR "
            "both non-empty 'instruction' and 'output'."
        )

    if user_input:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{user_input}\n\n"
            "### Response:\n"
            f"{output}"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output}"
    )


def _load_training_dataset(dataset_path: Path):
    """Load and validate dataset.json into a HuggingFace Dataset."""
    from datasets import Dataset

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    raw = dataset_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset file is empty: {dataset_path}")

    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "conversations" in parsed:
        parsed = parsed["conversations"]
    if not isinstance(parsed, list):
        raise ValueError(
            "Dataset must be a JSON array, or an object with 'conversations' array."
        )
    if not parsed:
        raise ValueError("Dataset must contain at least 1 training example.")

    rows: list[dict[str, str]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset example at index {idx} is not a JSON object.")
        rows.append({"text": _format_example(item)})

    return Dataset.from_list(rows)


def _print_oom_fallbacks(config: dict[str, Any]) -> None:
    """Print configured OOM recovery steps from YAML config."""
    fallbacks = config.get("oom_fallbacks", [])
    if not isinstance(fallbacks, list) or not fallbacks:
        print("No OOM fallback steps found in config.")
        return

    print("\nOOM recovery steps from config:")
    for i, step in enumerate(fallbacks, start=1):
        if not isinstance(step, dict):
            continue
        action = step.get("action", "unknown")
        details = ", ".join(
            [f"{k}={v}" for k, v in step.items() if k != "action"]
        )
        print(f"{i}. {action}" + (f" ({details})" if details else ""))


def _resolve_output_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve checkpoint and adapter output directories from config."""
    output_cfg = config.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValueError("Config key 'output' must be a mapping/object.")

    adapters_dir = Path(
        output_cfg.get("adapters_dir")
        or output_cfg.get("output_dir")
        or "output/adapters"
    )
    checkpoints_dir = Path(
        output_cfg.get("checkpoints_dir")
        or output_cfg.get("output_dir")
        or str(adapters_dir)
    )
    adapters_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir, adapters_dir


def run_training(config_path: Path, dataset_path: Path) -> None:
    """Run Trinity fine-tuning with Unsloth + TRL."""
    config = _load_yaml(config_path)

    # Import Unsloth first to preserve patched optimization path.
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth is not installed. In Colab, install dependencies from "
            "notebooks/main.ipynb first."
        ) from exc

    from transformers import TrainingArguments
    from trl import SFTTrainer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Use Google Colab with T4 GPU for this training."
        )

    train_dataset = _load_training_dataset(dataset_path)

    model_name = str(config.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("Config key 'model_name' is required.")

    max_seq_length = int(config.get("max_seq_length", 4096))
    load_in_4bit = bool(config.get("load_in_4bit", True))
    lora_cfg = config.get("lora", {})
    if not isinstance(lora_cfg, dict):
        raise ValueError("Config key 'lora' must be a mapping/object.")
    train_cfg = config.get("training", {})
    if not isinstance(train_cfg, dict):
        raise ValueError("Config key 'training' must be a mapping/object.")
    memory_cfg = config.get("memory", {})
    if not isinstance(memory_cfg, dict):
        raise ValueError("Config key 'memory' must be a mapping/object.")

    checkpoints_dir, adapters_dir = _resolve_output_dirs(config)

    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    target_modules = lora_cfg.get(
        "target_modules",
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("lora.target_modules must be a non-empty list.")

    print("Applying Trinity LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg.get("r", 64)),
        target_modules=target_modules,
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.1)),
        bias=str(lora_cfg.get("bias", "none")),
        use_gradient_checkpointing=memory_cfg.get(
            "gradient_checkpointing", "unsloth"
        ),
        random_state=3407,
        use_rslora=bool(lora_cfg.get("use_rslora", True)),
        use_dora=bool(lora_cfg.get("use_dora", True)),
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=int(
            train_cfg.get("per_device_train_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 4)
        ),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        max_steps=int(train_cfg.get("max_steps", 60)),
        warmup_steps=int(train_cfg.get("warmup_steps", 10)),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_steps=int(train_cfg.get("save_steps", 20)),
        save_total_limit=int(config.get("output", {}).get("save_total_limit", 3)),
        optim=str(train_cfg.get("optimizer", "adamw_8bit")),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        fp16=bool(memory_cfg.get("fp16", True)),
        bf16=False,
        dataloader_num_workers=int(memory_cfg.get("dataloader_num_workers", 0)),
        report_to=[],
        remove_unused_columns=False,
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=training_args,
    )

    print(
        f"Starting training on {torch.cuda.get_device_name(0)} "
        f"with {len(train_dataset)} examples."
    )
    try:
        trainer.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("\nCUDA OOM encountered.")
            _print_oom_fallbacks(config)
        raise

    print(f"Saving trained adapter to: {adapters_dir}")
    trainer.save_model(str(adapters_dir))
    tokenizer.save_pretrained(str(adapters_dir))
    print("Training complete.")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train Gemma-3 Trinity adapter (QLoRA + rsLoRA + DoRA)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset JSON file.",
    )
    args = parser.parse_args()
    run_training(config_path=args.config, dataset_path=args.dataset)


if __name__ == "__main__":
    main()
