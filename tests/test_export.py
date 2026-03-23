from __future__ import annotations

import json
from pathlib import Path

from src.export import export_to_gguf, _resolve_adapter_path


class _FakeModel:
    def save_pretrained_gguf(self, output_dir, processor, quantization_method):
        root = Path(output_dir)
        nested = root / "nested"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / f"ghost-architect-{quantization_method}.gguf").write_bytes(b"gguf" + b"0" * 1024)


def test_resolve_adapter_path_falls_back_to_modal_adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "output" / "adapters" / "trinity_a10g"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    resolved = _resolve_adapter_path("missing/adapter")

    assert resolved == adapter


def test_export_to_gguf_writes_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "output" / "adapters" / "vision_trinity"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    def fake_load_model(adapter_path: Path):
        assert adapter_path == adapter
        return _FakeModel(), object()

    output_dir = tmp_path / "output" / "gguf"
    artifacts = export_to_gguf(
        adapter_dir=str(adapter),
        output_dir=str(output_dir),
        quantization="q4_k_m",
        model_name="ghost-architect-v1",
        min_gguf_size_mb=0,
        load_model_fn=fake_load_model,
    )

    gguf_path = output_dir / "ghost-architect-v1.gguf"
    modelfile_path = output_dir / "Modelfile"
    manifest_path = output_dir / "export-manifest.json"

    assert artifacts["gguf_path"] == gguf_path
    assert gguf_path.exists()
    assert modelfile_path.exists()
    assert manifest_path.exists()

    modelfile_text = modelfile_path.read_text(encoding="utf-8")
    assert f"FROM {gguf_path.resolve().as_posix()}" in modelfile_text
    assert "PARAMETER temperature 0.3" in modelfile_text
    assert "PARAMETER top_p 0.9" in modelfile_text
    assert 'SYSTEM """' in modelfile_text

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "ghost-architect-v1"
    assert manifest["quantization"] == "q4_k_m"
    assert manifest["gguf_path"] == str(gguf_path.resolve())
    assert manifest["modelfile_path"] == str(modelfile_path.resolve())
