.PHONY: help venv install validate dataset-check train export test clean

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
	@echo "  make export        - Run export entrypoint"
	@echo "  make test          - Run project tests"
	@echo "  make clean         - Remove Python cache files"

venv:
	$(UV) venv $(VENV)

install: venv
	$(UV) pip install --python $(PYTHON) -r requirements.txt

validate:
	$(PYTHON) scripts/validate_environment.py

dataset-check:
	@test -f $(DATASET) || (echo "Missing dataset: $(DATASET)" && exit 1)
	@$(PYTHON) - <<'PY'
import json
from pathlib import Path
p = Path("data/dataset.json")
raw = p.read_text().strip()
if not raw:
    raise SystemExit("data/dataset.json is empty")
json.loads(raw)
print("dataset.json is valid JSON")
PY

train:
	$(PYTHON) src/train.py --config $(CONFIG) --dataset $(DATASET)

export:
	$(PYTHON) src/export.py

test:
	$(PYTHON) -m pytest -q

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
