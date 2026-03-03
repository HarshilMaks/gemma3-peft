# Data Factory - Generate 5,000 Synthetic UI/SQL Pairs

## Quick Start

### 1. Setup
```bash
# Install dependencies
uv pip install -r requirements.txt
uv run playwright install  # Download browsers (~500MB)

# Set API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Run
```bash
# Test with 10 images (2-3 minutes)
uv run python scripts/data_factory.py 10

# Generate 100 images (30-40 minutes)
uv run python scripts/data_factory.py 100

# Generate 5000 images (8-10 hours)
uv run python scripts/data_factory.py 5000
```

## What It Does

The script generates 4-step synthetic dataset:

1. **UI Descriptions** (Gemini API)
   - Generates JSON descriptions of different UI types
   - Caches to avoid re-generation

2. **HTML Rendering** (Jinja2)
   - Converts descriptions to interactive Tailwind CSS pages
   - Saves as HTML files

3. **Screenshots** (Playwright)
   - Takes 1280x720 PNG screenshots of each page
   - Uses headless Chrome

4. **SQL Generation** (Gemini Vision API)
   - Analyzes each screenshot
   - Generates PostgreSQL CREATE TABLE statements
   - Saves final dataset

## Output

```
data/synthetic_factory/
├── ui_descriptions.json      # All generated UI descriptions
├── html_pages/              # Rendered HTML files
│   ├── ui_0000.html
│   └── ...
├── screenshots/             # PNG screenshots
│   ├── ui_0000.png
│   └── ...
└── synthetic_dataset.json   # Final dataset (ready for training)
```

The `synthetic_dataset.json` format:
```json
[
  {
    "image_path": "data/synthetic_factory/screenshots/ui_0000.png",
    "instruction": "Generate PostgreSQL schema from this UI",
    "output": "CREATE TABLE users (...);",
    "domain": "synthetic",
    "size_kb": 285.4
  }
]
```

## Cost & Limits

**Free Tier Gemini API:**
- Rate limit: 15 requests/minute
- Daily quota: 1M tokens/day
- **Sufficient for 5,000 images** ✅

**Storage:**
- ~300-400 KB per screenshot
- 5,000 images ≈ 1.5-2 GB

## Next Steps

After generation:
1. Verify dataset: `python -c "import json; data = json.load(open('data/synthetic_factory/synthetic_dataset.json')); print(f'{len(data)} items')`
2. Use for training: Update `src/train_vision.py` to use this dataset
3. Merge with existing data if needed

## Troubleshooting

**"GEMINI_API_KEY not found"**
- Check `.env` file exists and has your key
- Run: `echo $GEMINI_API_KEY` to verify environment

**"Rate limit exceeded"**
- Free tier has 15 req/min limit
- Script waits 1 second between requests
- Consider paid tier for faster generation

**"Playwright not found"**
- Run: `uv run playwright install`

**"jinja2 not found"**
- Run: `uv pip install jinja2` (auto-installed by script anyway)
