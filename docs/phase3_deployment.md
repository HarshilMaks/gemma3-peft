# Phase 3: Production Deployment — Ghost Architect API

## Overview

Deploy the trained Ghost Architect model as a production-ready web API.
After Phase 2, you have a `.gguf` file. Phase 3 turns it into a service
anyone can call with a screenshot and get a SQL schema back.

**What you're building:**
- 🌐 FastAPI server exposing `POST /generate-schema`
- 🐳 Docker container for reproducible deployment
- ⚡ Ollama-backed inference (runs GGUF locally or on server)
- 🔒 Auth middleware + rate limiting

**Prerequisites:**
- ✅ Phase 1 complete: `output/adapters/phase1/` exists
- ✅ Phase 2 complete: `output/gguf/ghost-architect-v1.gguf` exists
- ✅ Ollama installed locally (`curl -fsSL https://ollama.com/install.sh | sh`)

---

## Architecture: How Phase 3 Connects Everything

```
User sends request
        │
        ▼
FastAPI  (src/api/main.py)
  └─ POST /generate-schema  (src/api/endpoints.py)
        │ validates request (src/api/models.py)
        │ checks auth header (src/api/middleware.py)
        ▼
Inference engine  (src/inference.py)
  └─ loads ghost-architect-v1.gguf via Ollama
  └─ sends image + prompt
        │
        ▼
Ghost Architect model
  └─ vision encoder reads screenshot pixel_values
  └─ language model generates SQL schema
        │
        ▼
SQL Validator  (src/data/sql_validator.py)
  └─ checks CREATE TABLE syntax is valid
        │
        ▼
Response back to user
```

---

## Step 1: Register Model with Ollama

```bash
# Create the Modelfile
cat > Modelfile << 'EOF'
FROM ./output/gguf/ghost-architect-v1.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """You are Ghost Architect — an expert in UI analysis and PostgreSQL schema design.
When given a screenshot of a UI (dashboard, e-commerce, admin panel, etc.),
you output a complete, normalized PostgreSQL schema with proper data types,
primary keys, foreign keys, and indexes.
Output only valid SQL. No explanations unless asked."""
EOF

# Register and test
ollama create ghost-architect -f Modelfile
ollama run ghost-architect "Hello — are you ready?"
```

---

## Step 2: Implement the API Files

These are currently 1-line stubs. Implement them in this order:

### 2a. `src/api/models.py` — Request/Response Shapes

```python
from pydantic import BaseModel
from typing import Optional

class SchemaRequest(BaseModel):
    image_path: str           # Path to UI screenshot
    instruction: Optional[str] = "Analyze this UI and generate the PostgreSQL schema."
    domain: Optional[str] = None   # hint: "ecommerce", "dashboard", "admin"

class SchemaResponse(BaseModel):
    sql_schema: str           # Generated CREATE TABLE statements
    tables_detected: int      # How many tables were generated
    domain: str               # What type of UI was detected
    model_version: str = "ghost-architect-v1"
```

### 2b. `src/inference.py` — Model Inference

```python
from ollama import chat
from pathlib import Path

def generate_schema(image_path: str, instruction: str) -> str:
    response = chat(
        model="ghost-architect",
        messages=[{
            "role": "user",
            "content": instruction,
            "images": [image_path]
        }]
    )
    return response["message"]["content"]
```

### 2c. `src/api/endpoints.py` — Route Handler

```python
from fastapi import APIRouter
from src.api.models import SchemaRequest, SchemaResponse
from src.inference import generate_schema

router = APIRouter()

@router.post("/generate-schema", response_model=SchemaResponse)
async def generate_schema_endpoint(request: SchemaRequest):
    sql = generate_schema(request.image_path, request.instruction)
    tables = sql.count("CREATE TABLE")
    return SchemaResponse(
        sql_schema=sql,
        tables_detected=tables,
        domain=request.domain or "unknown"
    )
```

### 2d. `src/api/main.py` — FastAPI App

```python
from fastapi import FastAPI
from src.api.endpoints import router

app = FastAPI(title="Ghost Architect API", version="1.0.0")
app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health():
    return {"status": "ok", "model": "ghost-architect-v1"}
```

### 2e. `src/api/middleware.py` — Auth (Optional)

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api"):
            key = request.headers.get("X-API-Key")
            if not key or key != os.getenv("API_KEY"):
                raise HTTPException(status_code=401)
        return await call_next(request)
```

---

## Step 3: Run the API Server

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn ollama

# Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Test it
curl -X POST http://localhost:8000/api/v1/generate-schema \
  -H "Content-Type: application/json" \
  -d '{"image_path": "data/ui_screenshots/stripe.com_67324.png"}'
```

---

## Step 4: Docker Deployment

The `docker/` folder has empty stubs. Implement:

### `docker/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Copy the GGUF model
COPY output/gguf/ghost-architect-v1.gguf /models/

# Register with Ollama at startup
COPY Modelfile /models/Modelfile
RUN ollama serve & sleep 5 && ollama create ghost-architect -f /models/Modelfile

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker/docker-compose.yml`

```yaml
version: "3.9"

services:
  ghost-architect:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${API_KEY}
    volumes:
      - ./data/ui_screenshots:/app/data/ui_screenshots
    restart: unless-stopped
```

```bash
# Build and run
docker-compose up --build

# Test
curl http://localhost:8000/health
```

---

## Step 5: Configs

### `configs/model_config.yaml`
```yaml
model_name: ghost-architect-v1
gguf_path: output/gguf/ghost-architect-v1.gguf
ollama_model: ghost-architect

inference:
  temperature: 0.3
  top_p: 0.9
  max_tokens: 2048
  context_length: 4096
```

### `configs/deployment_config.yaml`
```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 2
  reload: false

auth:
  enabled: true
  header: X-API-Key

rate_limiting:
  requests_per_minute: 30
  burst: 10
```

---

## Output: What Phase 3 Produces

```
A running API server that:
  • Accepts a UI screenshot (image path or base64)
  • Returns a PostgreSQL CREATE TABLE schema
  • Runs the fine-tuned Ghost Architect GGUF model
  • Validates generated SQL before returning it

POST /api/v1/generate-schema
  Input:  { "image_path": "data/ui_screenshots/stripe.png" }
  Output: {
    "sql_schema": "CREATE TABLE payments (...);",
    "tables_detected": 3,
    "domain": "ecommerce",
    "model_version": "ghost-architect-v1"
  }
```

---

## Files to Implement (in order)

| File | Status | Priority |
|------|--------|----------|
| `src/api/models.py` | Stub → implement | 1st |
| `src/inference.py` | Stub → implement | 2nd |
| `src/api/endpoints.py` | Stub → implement | 3rd |
| `src/api/main.py` | Stub → implement | 4th |
| `src/data/sql_validator.py` | Stub → implement | 5th |
| `docker/Dockerfile` | Empty → implement | 6th |
| `docker/docker-compose.yml` | Empty → implement | 7th |
| `src/api/middleware.py` | Stub → implement | Optional |
| `configs/model_config.yaml` | 1-line → fill in | With step 4 |
| `configs/deployment_config.yaml` | 1-line → fill in | With step 4 |

---

## Phase Dependency Map

```
Phase 1  (Text Trinity training)
    │
    ▼
Phase 2  (Vision training on 136 UI screenshots)
    │
    ▼
Phase 3  (Deploy as API) ← YOU ARE PLANNING THIS
```

**Phase 3 cannot start until Phase 2 produces `output/gguf/ghost-architect-v1.gguf`.**

---

## References

| Resource | Link |
|----------|------|
| Phase 1 training guide | `docs/phase1_trinity_training.md` |
| Phase 2 vision training guide | `docs/phase2_vision_training.md` |
| Full architecture | `docs/architecture.md` |
| Product requirements | `docs/prd.md` |
| FastAPI docs | https://fastapi.tiangolo.com |
| Ollama Python client | https://github.com/ollama/ollama-python |
| GGUF format | https://github.com/ggerganov/llama.cpp |
