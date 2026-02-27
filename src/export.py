"""GGUF export utilities (to be implemented)."""
#!/usr/bin/env python3
"""
Export trained LoRA adapters to GGUF format for local inference via Ollama.

How GGUF export works:
  1. Load the base model again (same Gemma-3 12B, 4-bit)
  2. Merge the LoRA adapter weights INTO the base model
     (LoRA adds small delta matrices A and B; merged = base + A*B)
  3. Save merged model as GGUF with q4_k_m quantization
     (q4_k_m = 4-bit, k-quantized, medium accuracy — best for inference)

Output: A single .gguf file you can run with `ollama run ./ghost-architect.gguf`
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def export_to_gguf(adapter_dir: str, output_dir: str, quantization: str = "q4_k_m"):
    """
    Merge LoRA adapters into base model and export as GGUF.

    Args:
        adapter_dir:   Path to saved LoRA adapters (output/adapters/phase2)
        output_dir:    Where to write the .gguf file (output/gguf)
        quantization:  GGUF quantization level (q4_k_m recommended for T4)
    """
    from unsloth import FastVisionModel

    adapter_path = Path(adapter_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}\n"
            f"Run training first: python src/train_vision.py --dataset data/dataset_vision.json"
        )

    logger.info(f"Loading base model + adapters from {adapter_path}...")

    # Reload model with the saved LoRA adapters attached
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(adapter_path),   # Unsloth reads adapter_config.json here
        load_in_4bit=True,
    )

    logger.info(f"Exporting to GGUF ({quantization})...")
    logger.info(f"Output directory: {output_path}/")
    logger.info("This takes ~5-10 minutes — the LoRA weights are being merged and quantized...")

    # save_pretrained_gguf:
    #   - Merges adapter deltas (A*B) into full weight matrices
    #   - Quantizes to q4_k_m (each weight stored in ~4 bits)
    #   - Writes a single .gguf file consumable by llama.cpp / Ollama
    model.save_pretrained_gguf(
        str(output_path / "ghost-architect-v1"),
        processor,
        quantization_method=quantization,
    )

    gguf_files = list(output_path.glob("**/*.gguf"))
    if gguf_files:
        gguf_file = gguf_files[0]
        size_gb = gguf_file.stat().st_size / (1024 ** 3)
        logger.info(f"\n✅ GGUF export complete!")
        logger.info(f"   File: {gguf_file}")
        logger.info(f"   Size: {size_gb:.1f} GB")
        logger.info(f"\nTo run locally:")
        logger.info(f"   ollama run {gguf_file}")
    else:
        logger.warning("Export ran but no .gguf file found — check output directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LoRA adapters to GGUF for Ollama")
    parser.add_argument("--adapter_dir", type=str, default="output/adapters/phase2",
                        help="Directory containing saved LoRA adapter weights")
    parser.add_argument("--output_dir", type=str, default="output/gguf",
                        help="Where to save the .gguf file")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q8_0", "f16"],
                        help="q4_k_m=best balance, q8_0=higher quality, f16=no compression")
    args = parser.parse_args()
    export_to_gguf(args.adapter_dir, args.output_dir, args.quantization)

