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
        # Core ML stack — torch>=2.6 required by current transformers masking path
        "torch==2.6.0",
        "torchvision==0.21.0",
        # Unsloth + Zoo from matching sources to avoid API mismatch (e.g. device_synchronize)
        "unsloth @ git+https://github.com/unslothai/unsloth.git",
        "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git",
        # Let Unsloth stack resolve its own compatible transformers/trl/peft set
        "accelerate>=0.34.0",
        "bitsandbytes>=0.44.0",
        "xformers",
        "datasets>=3.0.0",
        "huggingface_hub>=0.25.0",
        "pillow>=10.0.0",
        "triton==3.2.0",
        "sentencepiece",
    )
    # torchao can still introduce incompatible code paths via transitive deps. Keep it out.
    .run_commands("python -m pip uninstall -y torchao || true")
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
def train(dataset_filename: str = "dataset_merged.json", dry_run_limit: int = 0):
    import os
    import logging
    import torch

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Point HuggingFace cache at the persistent volume
    os.environ["HF_HOME"]             = str(CACHE_PATH)
    os.environ["TRANSFORMERS_CACHE"]  = str(CACHE_PATH / "transformers")
    os.environ["HF_DATASETS_CACHE"]   = str(CACHE_PATH / "datasets")
    # Force safer attention backend to avoid kernel NYI errors in SigLIP/Gemma3 vision path.
    os.environ["XFORMERS_DISABLED"]   = "1"
    os.environ["DISABLE_FLEX_ATTENTION"] = "1"

    # Auth
    from huggingface_hub import login
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token, add_to_git_credential=False)
    print("✅ HuggingFace authenticated")

    # Apply DoRA patch
    _patch_dora()

    # Prefer math SDPA backend for maximum compatibility on this stack.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        print("✅ Attention backend forced to math SDPA (flash/mem-efficient disabled)")

    from unsloth import FastVisionModel, is_bfloat16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    dataset_json = DATASET_PATH / dataset_filename
    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset {dataset_filename} not found in Modal volume. Run upload_dataset first.")
        
    dataset = _load_dataset(dataset_json)
    if dry_run_limit > 0:
        limit = min(dry_run_limit, len(dataset))
        dataset = dataset.select(range(limit))
        print(f"🧪 DRY RUN ACTIVATED: Training on only {len(dataset)} examples")

    print("Loading Gemma-3-12B-IT vision model...")
    model, processor = FastVisionModel.from_pretrained(
        model_name="unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        load_in_4bit=True,   # QLoRA: 12B fits in 24GB with room for DoRA gradients
        attn_implementation="eager",
    )

    # Full Trinity stack on A10G: QLoRA + DoRA + rsLoRA
    # A10G supports bf16 natively, so no dtype edge cases.
    # Vision layers are fine-tuned to improve screenshot encoding quality.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,      # A10G can handle it + bf16 support
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
    print("✅ Model loaded with full Trinity (QLoRA + DoRA + rsLoRA + Vision Fine-tuning)")

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
            max_seq_length=2048,             # Gemma3 runtime currently clamps to 2048
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

    print(f"Downloading {len(files)} entries...")
    downloaded = 0
    skipped = 0
    for entry in files:
        # Skip directory markers if present in listing.
        if str(getattr(entry, "type", "")).lower() == "directory" or entry.path.endswith("/"):
            skipped += 1
            continue
        dest = local_out / Path(entry.path).relative_to(adapter_prefix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob = output_vol.read_file(entry.path)
            if isinstance(blob, (bytes, bytearray)):
                dest.write_bytes(bytes(blob))
            else:
                dest.write_bytes(b"".join(blob))
            downloaded += 1
        except Exception as e:
            # Modal may list non-regular entries (e.g. checkpoint dirs) as paths.
            # Skip those and continue downloading regular files.
            skipped += 1
            print(f"⚠️ Skipping non-file entry: {entry.path} ({e})")

    print(f"✅ Adapter downloaded to {local_out}/ (files: {downloaded}, skipped: {skipped})")


# ── Inference output helpers ──────────────────────────────────────────────────
def _strip_markdown_code_fence(text: str) -> str:
    import re

    stripped = text.strip()
    fenced = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n```$", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def _split_mermaid_and_sql(text: str) -> tuple[str | None, str]:
    marker_mermaid = "=== MERMAID ==="
    marker_sql = "=== SQL ==="
    if marker_mermaid in text:
        after_mermaid = text.split(marker_mermaid, 1)[1]
        if marker_sql in after_mermaid:
            mermaid_part, sql_part = after_mermaid.split(marker_sql, 1)
            mermaid_part = _strip_markdown_code_fence(mermaid_part)
            sql_part = _strip_markdown_code_fence(sql_part)
            return (mermaid_part.strip() or None), sql_part.strip()

        # Mermaid-only fallback: keep the diagram, leave SQL empty.
        mermaid_part = _strip_markdown_code_fence(after_mermaid)
        return mermaid_part.strip() or None, ""

    return None, _strip_markdown_code_fence(text)


def _sql_to_mermaid_fallback(sql_text: str) -> str | None:
    """
    Build a basic Mermaid ER diagram from SQL when model does not return one.
    """
    import re

    tables: list[tuple[str, list[tuple[str, str, str]]]] = []
    relations: list[tuple[str, str, str]] = []

    create_table_pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*?)\)\s*;",
        re.IGNORECASE | re.DOTALL,
    )

    for match in create_table_pattern.finditer(sql_text):
        table_name = match.group(1)
        body = match.group(2)
        columns: list[tuple[str, str, str]] = []

        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue

            upper = line.upper()
            if upper.startswith(("PRIMARY KEY", "UNIQUE", "CHECK", "CONSTRAINT")):
                continue

            if upper.startswith("FOREIGN KEY"):
                fk_match = re.search(
                    r"FOREIGN\s+KEY\s*\([`\"]?(\w+)[`\"]?\)\s*REFERENCES\s+[`\"]?(\w+)[`\"]?",
                    line,
                    re.IGNORECASE,
                )
                if fk_match:
                    relations.append((table_name, fk_match.group(1), fk_match.group(2)))
                continue

            parts = line.split()
            col_name = parts[0].strip('`"') if parts else "unknown_column"
            col_type = parts[1] if len(parts) > 1 else "TEXT"
            tags: list[str] = []

            if "PRIMARY KEY" in upper:
                tags.append("PK")

            if "REFERENCES" in upper:
                tags.append("FK")
                ref_match = re.search(r"REFERENCES\s+[`\"]?(\w+)[`\"]?", line, re.IGNORECASE)
                if ref_match:
                    relations.append((table_name, col_name, ref_match.group(1)))

            columns.append((col_type, col_name, ",".join(tags)))

        if columns:
            tables.append((table_name, columns))

    if not tables:
        return None

    lines = ["erDiagram"]
    for table_name, columns in tables:
        lines.append(f"    {table_name.upper()} {{")
        for col_type, col_name, tags in columns:
            if tags:
                lines.append(f'        {col_type} {col_name} "{tags}"')
            else:
                lines.append(f"        {col_type} {col_name}")
        lines.append("    }")

    for src, col, dst in relations:
        lines.append(f'    {src.upper()} }}o--|| {dst.upper()} : "{col}"')

    return "\n".join(lines)


def _sanitize_mermaid_erdiagram(mermaid_text: str) -> str | None:
    """
    Normalize Mermaid ER output so duplicate entity blocks collapse into one valid diagram.

    The model often repeats the same entity multiple times. Mermaid ER diagrams reject that,
    so we merge columns per entity and infer simple FK relationships from *_id FK fields.
    """
    import re

    text = _strip_markdown_code_fence(mermaid_text)
    if "=== MERMAID ===" in text:
        text = text.split("=== MERMAID ===", 1)[1]
    if "=== SQL ===" in text:
        text = text.split("=== SQL ===", 1)[0]

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if lines[0].strip() == "erDiagram":
        lines = lines[1:]

    entity_order: list[str] = []
    entities: dict[str, list[dict[str, object]]] = {}
    explicit_relations: list[tuple[str, str, str, str]] = []

    entity_header = re.compile(r"^([A-Za-z_][\w]*)\s*\{$")
    relation_line = re.compile(r"^([A-Za-z_][\w]*)\s+([|}{o\-.]+)\s+([A-Za-z_][\w]*)\s*:\s*(.+)$")

    current_entity: str | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("%%"):
            continue

        header = entity_header.match(line)
        if header:
            current_entity = header.group(1)
            if current_entity not in entities:
                entities[current_entity] = []
                entity_order.append(current_entity)
            continue

        if line == "}":
            current_entity = None
            continue

        rel = relation_line.match(line)
        if rel and current_entity is None:
            explicit_relations.append(rel.groups())
            continue

        if current_entity is None:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        col_type = parts[0]
        col_name = parts[1]
        rest = " ".join(parts[2:]).upper()
        tags: list[str] = []
        if "PK" in rest:
            tags.append("PK")
        if "FK" in rest:
            tags.append("FK")
        if "UNIQUE" in rest:
            tags.append("UNIQUE")
        if "NOT NULL" in rest or "NN" in rest:
            tags.append("NN")

        column_list = entities[current_entity]
        existing = next((col for col in column_list if col["name"] == col_name), None)
        if existing:
            existing_tags = set(existing["tags"])  # type: ignore[index]
            existing_tags.update(tags)
            existing["tags"] = sorted(existing_tags)  # type: ignore[index]
            if existing["type"] in {"string", "text", "?"} and col_type not in {"string", "text", "?"}:
                existing["type"] = col_type  # type: ignore[index]
        else:
            column_list.append({"type": col_type, "name": col_name, "tags": tags})

    if not entities:
        return None

    def to_entity_name(column_name: str) -> str:
        base = column_name[:-3] if column_name.endswith("_id") else column_name
        return "".join(part.capitalize() for part in base.split("_") if part)

    relations: list[tuple[str, str, str, str]] = []
    seen_relations: set[tuple[str, str, str, str]] = set()

    for left, connector, right, label in explicit_relations:
        key = (left, connector, right, label)
        if key not in seen_relations:
            relations.append((left, connector, right, label))
            seen_relations.add(key)

    for child_entity in entity_order:
        for col in entities[child_entity]:
            if "FK" not in col["tags"]:  # type: ignore[index]
                continue
            parent_candidate = to_entity_name(str(col["name"]))  # type: ignore[index]
            parent_entity = next(
                (entity for entity in entity_order if entity.lower() == parent_candidate.lower()),
                None,
            )
            if not parent_entity or parent_entity == child_entity:
                continue
            key = (parent_entity, "||--o{", child_entity, str(col["name"]))  # type: ignore[index]
            if key in seen_relations:
                continue
            relations.append(key)
            seen_relations.add(key)

    output: list[str] = ["erDiagram"]
    for entity_name in entity_order:
        output.append(f"    {entity_name} {{")
        for col in entities[entity_name]:
            tags = " ".join(col["tags"])  # type: ignore[index]
            line = f"        {col['type']} {col['name']}"  # type: ignore[index]
            if tags:
                line += f" {tags}"
            output.append(line)
        output.append("    }")

    for left, connector, right, label in relations:
        output.append(f"    {left} {connector} {right} : {label}")

    return "\n".join(output)



def _parse_mermaid_erdiagram(mermaid_text: str) -> tuple[list[dict], list[dict]] | None:
    """Parse a Mermaid ER diagram into structured entities and relationships."""
    import re

    text = _strip_markdown_code_fence(mermaid_text)
    if "=== MERMAID ===" in text:
        text = text.split("=== MERMAID ===", 1)[1]
    if "=== SQL ===" in text:
        text = text.split("=== SQL ===", 1)[0]

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if lines[0].strip() == "erDiagram":
        lines = lines[1:]

    entity_header = re.compile(r"^([A-Za-z_][\w]*)\s*\{$")
    relation_line = re.compile(r"^([A-Za-z_][\w]*)\s+([|}{o\-.]+)\s+([A-Za-z_][\w]*)\s*:\s*(.+)$")

    entities: list[dict] = []
    entity_map: dict[str, dict] = {}
    relations: list[dict] = []
    current_entity: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("%%"):
            continue

        header = entity_header.match(line)
        if header:
            current_entity = header.group(1)
            if current_entity not in entity_map:
                entity = {"name": current_entity, "columns": []}
                entity_map[current_entity] = entity
                entities.append(entity)
            continue

        if line == "}":
            current_entity = None
            continue

        rel = relation_line.match(line)
        if rel and current_entity is None:
            relations.append({
                "left": rel.group(1),
                "connector": rel.group(2),
                "right": rel.group(3),
                "label": rel.group(4).strip(),
            })
            continue

        if current_entity is None:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        col_type = parts[0]
        col_name = parts[1].strip('`"')
        rest = " ".join(parts[2:]).upper()
        tags: list[str] = []
        if "PK" in rest:
            tags.append("PK")
        if "FK" in rest:
            tags.append("FK")
        if "UNIQUE" in rest:
            tags.append("UNIQUE")
        if "NOT NULL" in rest or "NN" in rest:
            tags.append("NN")

        column_list = entity_map[current_entity]["columns"]
        existing = next((col for col in column_list if col["name"] == col_name), None)
        if existing:
            existing_tags = set(existing["tags"])
            existing_tags.update(tags)
            existing["tags"] = sorted(existing_tags)
            if existing["type"] in {"string", "text", "?"} and col_type not in {"string", "text", "?"}:
                existing["type"] = col_type
        else:
            column_list.append({"type": col_type, "name": col_name, "tags": tags})

    if not entities:
        return None

    return entities, relations


def _build_mermaid_html(mermaid_diagram: str, title: str, source_text: str = "") -> str:
    """Build a polished Mermaid preview with a rendered diagram and source blocks."""
    import html as html_lib

    safe_title = html_lib.escape(title)
    safe_source = html_lib.escape(source_text.strip())
    mermaid_source = _strip_markdown_code_fence(mermaid_diagram).strip()
    safe_mermaid_source = html_lib.escape(mermaid_source)
    diagram_source = safe_mermaid_source or "erDiagram\\n    %% No Mermaid diagram found."

    mermaid_block = f"""
        <details>
          <summary>Show Mermaid source</summary>
          <pre class="source-block">{safe_mermaid_source or "(empty)"}</pre>
        </details>
        """
    raw_block = f"""
        <details>
          <summary>Show raw model output</summary>
          <pre class="source-block">{safe_source or "(empty)"}</pre>
        </details>
        """ if safe_source else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --border: #dbe4ee;
      --text: #0f172a;
      --muted: #475569;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }}
    .header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .title {{
      margin: 0;
      font-size: 24px;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255,255,255,0.85);
      font-size: 13px;
      color: var(--muted);
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
      overflow: hidden;
    }}
    .card-head {{
      padding: 18px 20px 0;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .description {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .diagram-wrap {{
      margin: 18px 20px 0;
      padding: 20px;
      border: 1px solid #cbd5e1;
      border-radius: 18px;
      background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 52%, #e2e8f0 100%);
      overflow: auto;
    }}
    .diagram-wrap .mermaid {{
      min-width: fit-content;
      padding: 12px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.7) 0%, rgba(248,250,252,0.92) 100%);
    }}
    .diagram-wrap svg {{
      max-width: none !important;
      height: auto !important;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%) !important;
      border-radius: 16px;
      box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.18);
    }}
    .source-block {{
      margin: 0;
      padding: 16px;
      overflow-x: auto;
      white-space: pre;
      background: #0f172a;
      color: #e2e8f0;
      border: 1px solid #1e293b;
      border-radius: 14px;
      font-size: 13px;
      line-height: 1.55;
    }}
    details {{
      border-top: 1px solid var(--border);
      padding: 16px 20px 18px;
      background: #f8fafc;
    }}
    details summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 12px;
    }}
    details pre {{
      margin: 0;
      padding: 16px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0f172a;
      color: #e2e8f0;
      border: 1px solid #1e293b;
      border-radius: 14px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .footer-note {{
      padding: 0 20px 20px;
      color: var(--muted);
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <h1 class="title">{safe_title}</h1>
        <p class="subtitle">Premium Mermaid ER diagram with the Mermaid source and raw model output preserved below.</p>
      </div>
      <div class="badge">Mermaid Diagram Representation · local file</div>
    </div>

    <div class="card">
      <div class="card-head">
        <span class="eyebrow">Mermaid Diagram Representation</span>
        <p class="description">The Mermaid source and raw model output preserved below</p>
      </div>
      <div class="diagram-wrap">
        <div class="mermaid">
{diagram_source}
        </div>
      </div>
      {mermaid_block}
      {raw_block}
      <div class="footer-note">Use the Mermaid source above in any Mermaid renderer if you want to reuse or edit the diagram.</div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      mermaid.initialize({{
        startOnLoad: false,
        theme: 'base',
        securityLevel: 'loose',
        darkMode: false,
        er: {{
          useMaxWidth: true,
          layoutDirection: 'TB',
          diagramPadding: 28
        }},
        themeVariables: {{
          fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
          fontSize: '15px',
          background: '#eef2ff',
          primaryColor: '#eff6ff',
          primaryTextColor: '#0f172a',
          primaryBorderColor: '#60a5fa',
          secondaryColor: '#f8fafc',
          secondaryTextColor: '#0f172a',
          secondaryBorderColor: '#cbd5e1',
          tertiaryColor: '#ffffff',
          tertiaryTextColor: '#0f172a',
          noteBkgColor: '#ecfeff',
          noteTextColor: '#0f172a',
          noteBorderColor: '#67e8f9',
          lineColor: '#64748b',
          textColor: '#0f172a'
        }}
      }});
      mermaid.run({{ querySelector: '.mermaid' }});
    }});
  </script>
</body>
</html>
"""

# ── Inference output helpers ──────────────────────────────────────────────────
def _strip_markdown_code_fence(text: str) -> str:
    import re

    stripped = text.strip()
    fenced = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n```$", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def _split_mermaid_and_sql(text: str) -> tuple[str | None, str]:
    marker_mermaid = "=== MERMAID ==="
    marker_sql = "=== SQL ==="
    if marker_mermaid in text:
        after_mermaid = text.split(marker_mermaid, 1)[1]
        if marker_sql in after_mermaid:
            mermaid_part, sql_part = after_mermaid.split(marker_sql, 1)
            mermaid_part = _strip_markdown_code_fence(mermaid_part)
            sql_part = _strip_markdown_code_fence(sql_part)
            return (mermaid_part.strip() or None), sql_part.strip()

        # Mermaid-only fallback: keep the diagram, leave SQL empty.
        mermaid_part = _strip_markdown_code_fence(after_mermaid)
        return mermaid_part.strip() or None, ""

    return None, _strip_markdown_code_fence(text)


def _sql_to_mermaid_fallback(sql_text: str) -> str | None:
    """
    Build a basic Mermaid ER diagram from SQL when model does not return one.
    """
    import re

    tables: list[tuple[str, list[tuple[str, str, str]]]] = []
    relations: list[tuple[str, str, str]] = []

    create_table_pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*?)\)\s*;",
        re.IGNORECASE | re.DOTALL,
    )

    for match in create_table_pattern.finditer(sql_text):
        table_name = match.group(1)
        body = match.group(2)
        columns: list[tuple[str, str, str]] = []

        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue

            upper = line.upper()
            if upper.startswith(("PRIMARY KEY", "UNIQUE", "CHECK", "CONSTRAINT")):
                continue

            if upper.startswith("FOREIGN KEY"):
                fk_match = re.search(
                    r"FOREIGN\s+KEY\s*\([`\"]?(\w+)[`\"]?\)\s*REFERENCES\s+[`\"]?(\w+)[`\"]?",
                    line,
                    re.IGNORECASE,
                )
                if fk_match:
                    relations.append((table_name, fk_match.group(1), fk_match.group(2)))
                continue

            parts = line.split()
            col_name = parts[0].strip('`"') if parts else "unknown_column"
            col_type = parts[1] if len(parts) > 1 else "TEXT"
            tags: list[str] = []

            if "PRIMARY KEY" in upper:
                tags.append("PK")

            if "REFERENCES" in upper:
                tags.append("FK")
                ref_match = re.search(r"REFERENCES\s+[`\"]?(\w+)[`\"]?", line, re.IGNORECASE)
                if ref_match:
                    relations.append((table_name, col_name, ref_match.group(1)))

            columns.append((col_type, col_name, ",".join(tags)))

        if columns:
            tables.append((table_name, columns))

    if not tables:
        return None

    lines = ["erDiagram"]
    for table_name, columns in tables:
        lines.append(f"    {table_name.upper()} {{")
        for col_type, col_name, tags in columns:
            if tags:
                lines.append(f'        {col_type} {col_name} "{tags}"')
            else:
                lines.append(f"        {col_type} {col_name}")
        lines.append("    }")

    for src, col, dst in relations:
        lines.append(f'    {src.upper()} }}o--|| {dst.upper()} : "{col}"')

    return "\n".join(lines)


def _sanitize_mermaid_erdiagram(mermaid_text: str) -> str | None:
    """
    Normalize Mermaid ER output so duplicate entity blocks collapse into one valid diagram.

    The model often repeats the same entity multiple times. Mermaid ER diagrams reject that,
    so we merge columns per entity and infer simple FK relationships from *_id FK fields.
    """
    import re

    text = _strip_markdown_code_fence(mermaid_text)
    if "=== MERMAID ===" in text:
        text = text.split("=== MERMAID ===", 1)[1]
    if "=== SQL ===" in text:
        text = text.split("=== SQL ===", 1)[0]

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if lines[0].strip() == "erDiagram":
        lines = lines[1:]

    entity_order: list[str] = []
    entities: dict[str, list[dict[str, object]]] = {}
    explicit_relations: list[tuple[str, str, str, str]] = []

    entity_header = re.compile(r"^([A-Za-z_][\w]*)\s*\{$")
    relation_line = re.compile(r"^([A-Za-z_][\w]*)\s+([|}{o\-.]+)\s+([A-Za-z_][\w]*)\s*:\s*(.+)$")

    current_entity: str | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("%%"):
            continue

        header = entity_header.match(line)
        if header:
            current_entity = header.group(1)
            if current_entity not in entities:
                entities[current_entity] = []
                entity_order.append(current_entity)
            continue

        if line == "}":
            current_entity = None
            continue

        rel = relation_line.match(line)
        if rel and current_entity is None:
            explicit_relations.append(rel.groups())
            continue

        if current_entity is None:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        col_type = parts[0]
        col_name = parts[1]
        rest = " ".join(parts[2:]).upper()
        tags: list[str] = []
        if "PK" in rest:
            tags.append("PK")
        if "FK" in rest:
            tags.append("FK")
        if "UNIQUE" in rest:
            tags.append("UNIQUE")
        if "NOT NULL" in rest or "NN" in rest:
            tags.append("NN")

        column_list = entities[current_entity]
        existing = next((col for col in column_list if col["name"] == col_name), None)
        if existing:
            existing_tags = set(existing["tags"])  # type: ignore[index]
            existing_tags.update(tags)
            existing["tags"] = sorted(existing_tags)  # type: ignore[index]
            if existing["type"] in {"string", "text", "?"} and col_type not in {"string", "text", "?"}:
                existing["type"] = col_type  # type: ignore[index]
        else:
            column_list.append({"type": col_type, "name": col_name, "tags": tags})

    if not entities:
        return None

    def to_entity_name(column_name: str) -> str:
        base = column_name[:-3] if column_name.endswith("_id") else column_name
        return "".join(part.capitalize() for part in base.split("_") if part)

    relations: list[tuple[str, str, str, str]] = []
    seen_relations: set[tuple[str, str, str, str]] = set()

    for left, connector, right, label in explicit_relations:
        key = (left, connector, right, label)
        if key not in seen_relations:
            relations.append((left, connector, right, label))
            seen_relations.add(key)

    for child_entity in entity_order:
        for col in entities[child_entity]:
            if "FK" not in col["tags"]:  # type: ignore[index]
                continue
            parent_candidate = to_entity_name(str(col["name"]))  # type: ignore[index]
            parent_entity = next(
                (entity for entity in entity_order if entity.lower() == parent_candidate.lower()),
                None,
            )
            if not parent_entity or parent_entity == child_entity:
                continue
            key = (parent_entity, "||--o{", child_entity, str(col["name"]))  # type: ignore[index]
            if key in seen_relations:
                continue
            relations.append(key)
            seen_relations.add(key)

    output: list[str] = ["erDiagram"]
    for entity_name in entity_order:
        output.append(f"    {entity_name} {{")
        for col in entities[entity_name]:
            tags = " ".join(col["tags"])  # type: ignore[index]
            line = f"        {col['type']} {col['name']}"  # type: ignore[index]
            if tags:
                line += f" {tags}"
            output.append(line)
        output.append("    }")

    for left, connector, right, label in relations:
        output.append(f"    {left} {connector} {right} : {label}")

    return "\n".join(output)


# ── Inference (test the adapter) ──────────────────────────────────────────────
@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        DATASET_PATH: dataset_vol,
        CACHE_PATH:   model_cache_vol,
        OUTPUT_PATH:  output_vol,
    },
)
def run_inference(adapter_name: str, image_path: str):
    """Run inference on a test image using the trained adapter."""
    import os
    from pathlib import Path
    import torch
    from PIL import Image
    from unsloth import FastVisionModel

    # Match training stability settings for Gemma3 vision path.
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["DISABLE_FLEX_ATTENTION"] = "1"
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    
    output_dir = OUTPUT_PATH / "adapters" / adapter_name
    if not output_dir.exists():
        print(f"❌ Adapter not found at {output_dir}")
        return None
    
    print(f"Loading adapter from {output_dir}...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(output_dir),
        load_in_4bit=True,
        attn_implementation="eager",
    )
    FastVisionModel.for_inference(model)
    
    # Check that image exists locally in dataset
    local_image = Path(image_path)
    if not local_image.exists():
        # Try without data/ prefix
        local_image = DATASET_PATH / "ui_screenshots" / Path(image_path).name
    
    if not local_image.exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"Loading image: {local_image}")
    image = Image.open(local_image).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(local_image), "text": ""},
                {
                    "type": "text",
                    "image": "",
                    "text": (
                        "Analyze this UI screenshot and generate a production-grade database design.\n\n"
                        "Return output in this exact format:\n"
                        "=== MERMAID ===\n"
                        "A valid Mermaid erDiagram with entities, columns, PK/FK tags, and relationships.\n\n"
                        "=== SQL ===\n"
                        "Valid PostgreSQL CREATE TABLE statements only.\n\n"
                        "No explanation. No extra text outside these markers."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt, images=image, return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1200,
            use_cache=True,
            do_sample=False,
        )

    full_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return full_text.split("model\n")[-1].strip() or full_text


@app.local_entrypoint()
def run_inference_local(adapter_name: str = "trinity_a10g", image_path: str = ""):
    """
    Test inference on a UI screenshot using a trained adapter.
    Usage:
      modal run src/modal_train.py::run_inference_local --adapter-name trinity_a10g --image-path data/ui_screenshots/paystack.co_44228.png

    Saves outputs to:
      output/inference/<image_stem>.mmd
      output/inference/<image_stem>.mermaid.html
      output/inference/<image_stem>.sql
      output/inference/<image_stem>.raw.txt
    """
    if not image_path:
        import glob as glob_module
        images = glob_module.glob("data/ui_screenshots/*.png")
        if images:
            image_path = images[0]
            print(f"Using first image: {image_path}")
        else:
            print("❌ No image path provided and no images found in data/ui_screenshots/")
            return
    
    result = run_inference.remote(adapter_name=adapter_name, image_path=image_path)
    if not result:
        print("❌ Inference returned no output.")
        return

    mermaid_diagram, sql_text = _split_mermaid_and_sql(result)
    if mermaid_diagram:
        sanitized_mermaid = _sanitize_mermaid_erdiagram(mermaid_diagram)
        if sanitized_mermaid:
            mermaid_diagram = sanitized_mermaid
    if not mermaid_diagram:
        mermaid_diagram = _sql_to_mermaid_fallback(sql_text)

    out_dir = Path("output/inference")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    raw_path = out_dir / f"{stem}.raw.txt"
    sql_path = out_dir / f"{stem}.sql"
    raw_path.write_text(result.strip() + "\n", encoding="utf-8")
    sql_output = sql_text.strip() or "-- SQL section was not generated by the model output."
    sql_path.write_text(sql_output + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    if mermaid_diagram:
        mermaid_path = out_dir / f"{stem}.mmd"
        mermaid_path.write_text(mermaid_diagram.strip() + "\n", encoding="utf-8")

        html_path = out_dir / f"{stem}.mermaid.html"
        html_path.write_text(
            _build_mermaid_html(
                mermaid_diagram,
                title="Ghost Architect Mermaid Diagram",
                source_text=result,
            ),
            encoding="utf-8",
        )

        print("=== MERMAID ===")
        print(mermaid_diagram.strip())
        print("\n=== SQL ===")
        print(sql_output)
        print("=" * 80)
        print(f"Saved Mermaid text: {mermaid_path}")
        print(f"Saved Mermaid HTML: {html_path}")
    else:
        print("=== SQL ===")
        print(sql_output)
        print("=" * 80)
        print("⚠️ Mermaid diagram was not generated.")

    print(f"Saved SQL: {sql_path}")
    print(f"Saved raw output: {raw_path}")


# ── Run Training ─────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(dataset_filename: str = "dataset_merged.json", dry_run_limit: int = 0):
    """
    Default entrypoint: 
      modal run src/modal_train.py
    """
    result = train.remote(dataset_filename=dataset_filename, dry_run_limit=dry_run_limit)
    print(f"\n🎉 Training complete!")
    print(f"   Adapter is in Modal Volume at: {result}")
    print(f"   Download it: modal run src/modal_train.py::download_adapter --adapter-name trinity_a10g")
