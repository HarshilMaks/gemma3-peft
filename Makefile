.PHONY: help venv install validate dataset-check train export test clean modal-dry-1 modal-dry-10 modal-train

VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv
CONFIG := configs/training_config.yaml
DATASET := data/dataset.json

help:
	@echo "Ghost Architect Make targets:"
	@echo "  make venv          - Create local virtual environment with uv"
	@echo "  make install       - Install project dependencies with uv"
	@echo "  make validate      - Validate environment and GPU readiness"
	@echo "  make dataset-check - Validate dataset JSON file is present and valid"
	@echo "  make train         - Run training entrypoint with config + dataset"
	@echo "  make modal-dry-1   - Modal training smoke test on 1 sample"
	@echo "  make modal-dry-10  - Modal dry run on 10 samples"
	@echo "  make modal-train   - Modal full training run"
	@echo "  make export        - Run export entrypoint (GGUF + Modelfile)"
	@echo "  make test          - Run project tests"
	@echo "  make clean         - Remove Python cache files"

venv:
	@test -x $(PYTHON) || $(UV) venv

install: venv
	$(UV) pip install --python $(PYTHON) -r requirements.txt

validate:
	$(PYTHON) scripts/validate_environment.py

dataset-check:
	@$(PYTHON) scripts/validate_dataset.py

train:
	$(PYTHON) src/train.py --config $(CONFIG) --dataset $(DATASET)

modal-dry-1:
	$(UV) tool run modal run src/modal_train.py::main --dry-run-limit 1

modal-dry-10:
	$(UV) tool run modal run src/modal_train.py::main --dry-run-limit 10

modal-train:
	$(UV) tool run modal run src/modal_train.py::main

export:
	$(PYTHON) src/export.py --adapter-dir "$${ADAPTER_DIR:-output/adapters/trinity_a10g}" --output-dir output/gguf --model-name ghost-architect-v1

test:
	$(PYTHON) -m pytest -q

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
