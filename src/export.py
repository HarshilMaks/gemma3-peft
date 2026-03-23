#!/usr/bin/env python3
"""Export trained LoRA adapters to GGUF for local Ollama deployment.

The exporter:
1. resolves the best available adapter directory
2. loads the vision model + processor
3. writes a quantized GGUF bundle
4. writes a companion Modelfile and manifest

The GGUF bundle is the deployment artifact for local Ollama usage.
For vision-aware screenshot inference, keep using the Python pipeline
in `src/app.py` and `src/inference.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "ghost-architect-v1"
DEFAULT_ADAPTER_DIR = "output/adapters/trinity_a10g"
DEFAULT_OUTPUT_DIR = "output/gguf"
DEFAULT_QUANTIZATION = "q4_k_m"
DEFAULT_MIN_GGUF_SIZE_MB = 100
DEFAULT_NUM_CTX = 4096
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
DEFAULT_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are Ghost Architect — an expert in UI analysis and PostgreSQL schema design.
    When given UI context, output a complete, normalized PostgreSQL schema with
    proper data types, primary keys, foreign keys, and indexes.
    Output only valid SQL. No explanations unless asked.
    """
).strip()

LoadModelFn = Callable[[Path], tuple[object, object]]


def _is_adapter_directory(path: Path) -> bool:
    if not path.is_dir():
        return False

    marker_names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "processor_config.json",
    )
    if any((path / marker).exists() for marker in marker_names):
        return True

    return any(
        child.is_dir() and (child / "adapter_config.json").exists()
        for child in path.iterdir()
    )


def _candidate_adapter_paths(adapter_dir: str) -> list[Path]:
    candidates = [
        Path(adapter_dir),
        Path("output/adapters/trinity_a10g"),
        Path("output/adapters/vision_trinity"),
    ]

    adapters_root = Path("output/adapters")
    if adapters_root.exists():
        candidates.extend(child for child in sorted(adapters_root.iterdir()) if child.is_dir())

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _resolve_adapter_path(adapter_dir: str) -> Path:
    candidates = _candidate_adapter_paths(adapter_dir)
    for candidate in candidates:
        if _is_adapter_directory(candidate):
            return candidate

    candidate_list = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Adapter directory not found. Checked:\n  - {candidate_list}\n"
        "Run training first, or set --adapter-dir to a saved adapter directory."
    )


def _load_model_and_processor(adapter_path: Path) -> tuple[object, object]:
    from unsloth import FastVisionModel

    return FastVisionModel.from_pretrained(
        model_name=str(adapter_path),
        load_in_4bit=True,
    )


def _find_exported_gguf(staging_root: Path) -> Path:
    candidates = [path for path in staging_root.rglob("*.gguf") if path.is_file()]
    if not candidates:
        raise RuntimeError("Export ran but no .gguf file was created.")
    if len(candidates) > 1:
        logger.warning("Multiple GGUF files found; using the largest one.")
    return max(candidates, key=lambda path: path.stat().st_size)


def _build_modelfile(
    gguf_path: Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    if '"""' in system_prompt:
        raise ValueError('system_prompt cannot contain triple quotes')

    prompt = textwrap.dedent(system_prompt).strip()
    return textwrap.dedent(
        f'''\
        FROM {gguf_path.resolve().as_posix()}

        PARAMETER temperature {temperature}
        PARAMETER top_p {top_p}
        PARAMETER num_ctx {num_ctx}

        SYSTEM """{prompt}"""
        '''
    ).strip() + "\n"


def _write_modelfile(
    modelfile_path: Path,
    gguf_path: Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> Path:
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(
        _build_modelfile(
            gguf_path=gguf_path,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            num_ctx=num_ctx,
        ),
        encoding="utf-8",
    )
    return modelfile_path


def _write_manifest(
    manifest_path: Path,
    model_name: str,
    adapter_path: Path,
    gguf_path: Path,
    modelfile_path: Path,
    quantization: str,
    size_bytes: int,
) -> Path:
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "adapter_path": str(adapter_path.resolve()),
        "gguf_path": str(gguf_path.resolve()),
        "modelfile_path": str(modelfile_path.resolve()),
        "quantization": quantization,
        "size_bytes": size_bytes,
        "ollama_create_command": f"ollama create {model_name} -f {modelfile_path.as_posix()}",
        "ollama_run_command": f"ollama run {model_name}",
        "notes": (
            "Vision-aware screenshot inference still uses the Python pipeline; "
            "this GGUF bundle is the local Ollama deployment artifact."
        ),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _validate_gguf_file(gguf_path: Path, min_size_mb: int) -> int:
    if not gguf_path.exists():
        raise FileNotFoundError(f"Expected GGUF file not found: {gguf_path}")

    size_bytes = gguf_path.stat().st_size
    if min_size_mb > 0 and size_bytes < min_size_mb * 1024 * 1024:
        raise RuntimeError(
            f"GGUF file is suspiciously small: {size_bytes / (1024 * 1024):.1f} MB "
            f"(< {min_size_mb} MB)."
        )
    return size_bytes


def export_to_gguf(
    adapter_dir: str,
    output_dir: str,
    quantization: str = DEFAULT_QUANTIZATION,
    model_name: str = DEFAULT_MODEL_NAME,
    min_gguf_size_mb: int = DEFAULT_MIN_GGUF_SIZE_MB,
    load_model_fn: LoadModelFn = _load_model_and_processor,
) -> dict[str, Path]:
    """Merge LoRA adapters into the base model and export GGUF + Ollama bundle."""
    adapter_path = _resolve_adapter_path(adapter_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_gguf_path = output_path / f"{model_name}.gguf"
    modelfile_path = output_path / "Modelfile"
    manifest_path = output_path / "export-manifest.json"

    logger.info(f"Loading model and processor from {adapter_path}...")

    with tempfile.TemporaryDirectory(prefix=f"{model_name}-gguf-", dir=output_path) as temp_dir:
        staging_root = Path(temp_dir)
        export_root = staging_root / model_name

        model, processor = load_model_fn(adapter_path)
        logger.info(f"Exporting to GGUF ({quantization})...")
        logger.info("This can take a few minutes while the adapter is merged and quantized...")
        model.save_pretrained_gguf(
            str(export_root),
            processor,
            quantization_method=quantization,
        )

        exported_gguf = _find_exported_gguf(staging_root)
        if final_gguf_path.exists():
            final_gguf_path.unlink()
        shutil.move(str(exported_gguf), str(final_gguf_path))

    size_bytes = _validate_gguf_file(final_gguf_path, min_gguf_size_mb)
    _write_modelfile(modelfile_path, final_gguf_path)
    _write_manifest(
        manifest_path=manifest_path,
        model_name=model_name,
        adapter_path=adapter_path,
        gguf_path=final_gguf_path,
        modelfile_path=modelfile_path,
        quantization=quantization,
        size_bytes=size_bytes,
    )

    size_gb = size_bytes / (1024 ** 3)
    logger.info("\n✅ GGUF export complete!")
    logger.info(f"   GGUF:       {final_gguf_path}")
    logger.info(f"   Modelfile:  {modelfile_path}")
    logger.info(f"   Manifest:   {manifest_path}")
    logger.info(f"   Size:       {size_gb:.2f} GB")
    logger.info("\nTo register in Ollama:")
    logger.info(f"   ollama create {model_name} -f {modelfile_path}")
    logger.info("To test:")
    logger.info(f"   ollama run {model_name}")
    logger.info(
        "\nNote: vision-aware screenshot inference still uses the Python processor path "
        "in src/app.py and src/inference.py."
    )

    return {
        "adapter_path": adapter_path,
        "gguf_path": final_gguf_path,
        "modelfile_path": modelfile_path,
        "manifest_path": manifest_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LoRA adapters to GGUF for Ollama")
    parser.add_argument(
        "--adapter-dir",
        "--adapter_dir",
        dest="adapter_dir",
        type=str,
        default=DEFAULT_ADAPTER_DIR,
        help="Directory containing saved LoRA adapter weights",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the GGUF bundle",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base name for the exported GGUF file",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=DEFAULT_QUANTIZATION,
        choices=["q4_k_m", "q8_0", "f16"],
        help="q4_k_m=best balance, q8_0=higher quality, f16=no compression",
    )
    parser.add_argument(
        "--min-gguf-size-mb",
        dest="min_gguf_size_mb",
        type=int,
        default=DEFAULT_MIN_GGUF_SIZE_MB,
        help="Minimum acceptable GGUF size in MB",
    )
    args = parser.parse_args()
    export_to_gguf(
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        quantization=args.quantization,
        model_name=args.model_name,
        min_gguf_size_mb=args.min_gguf_size_mb,
    )
