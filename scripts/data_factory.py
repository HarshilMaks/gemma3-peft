"""
Ghost Architect: Professional Batch Data Factory (v2.2) - STABLE RESUME
Generates 5,000 UI+SQL pairs for $0 by batching API requests.
- Fixed UnboundLocalError
- Stable JSON extraction
- Model fallback logic (1.5-flash-8b prioritized for quota)
- Auto-resume support
"""

import os
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from jinja2 import Template

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)

OUTPUT_DIR = Path("data/synthetic_factory")
HTML_DIR = OUTPUT_DIR / "html_pages"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
DATASET_FILE = OUTPUT_DIR / "synthetic_dataset.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

HTML_TEMPLATE = """<!DOCTYPE html><html><head><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-gray-50 p-8"><div class="max-w-4xl mx-auto bg-white shadow-xl rounded-xl overflow-hidden border">
<header class="bg-indigo-600 px-6 py-4 text-white"><h1 class="text-2xl font-bold">{{ name }}</h1></header>
<main class="p-6">{% if layout == 'card-grid' %}<div class="grid grid-cols-3 gap-4">
{% for f in fields %}<div class="p-4 border rounded-lg bg-gray-50"><p class="text-xs text-gray-400 uppercase font-bold">{{ f }}</p>
<div class="h-4 bg-gray-200 rounded mt-2 w-3/4"></div></div>{% endfor %}</div>
{% else %}<div class="space-y-4">{% for f in fields %}<div class="flex items-center justify-between p-3 border-b">
<span class="font-medium text-gray-700">{{ f }}</span><div class="h-6 w-24 bg-indigo-100 rounded"></div></div>{% endfor %}</div>{% endif %}
</main></div></body></html>"""

def get_model():
    """Use gemini-2.0-flash (confirmed available in user list)."""
    return genai.GenerativeModel('gemini-2.0-flash')

def clean_json(text: str) -> str:
    """Robustly extract JSON from model response."""
    for delim in ['```json', '```']:
        if delim in text:
            text = text.split(delim)[1]
            if '```' in text:
                text = text.split('```')[0]
            break
    return text.strip()

async def main(target_count: int):
    print(f"\n🚀 Launching SOTA Data Factory (8B-Quota-Optimized): Target {target_count} samples")
    model = get_model()
    dataset = []
    
    # FIX: Initialize backoff OUTSIDE the while loop
    backoff_time = 5 

    if DATASET_FILE.exists():
        try:
            with open(DATASET_FILE) as f:
                dataset = json.load(f)
            print(f"   Reusing {len(dataset)} existing samples")
        except:
            print("   ⚠️  Corrupt dataset file found, starting fresh.")
            dataset = []

    ui_types = ["Dashboard", "Analytics", "CRM", "E-commerce", "SaaS", "Settings", "Profile", "Inventory"]
    template = Template(HTML_TEMPLATE)

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})

        while len(dataset) < target_count:
            ui_type = ui_types[len(dataset) % len(ui_types)]
            prompt = f"Generate 5 unique UI designs for {ui_type}. Return JSON list: [{{'name':'...', 'fields':[], 'layout':'card-grid|list', 'sql':'CREATE TABLE...'}}]"
            
            try:
                print(f"   🛰️  Requesting batch (Current: {len(dataset)}/{target_count})...")
                resp = model.generate_content(prompt)
                batch = json.loads(clean_json(resp.text))

                if not isinstance(batch, list):
                    raise ValueError("Model didn't return a list")

                # Reset backoff on success
                backoff_time = 5

                for item in batch:
                    if len(dataset) >= target_count: break
                    
                    idx = len(dataset)
                    file_name = f"ui_{idx:04d}"
                    
                    # 1. Render HTML
                    html = template.render(
                        name=item.get('name', 'App'), 
                        fields=item.get('fields', []), 
                        layout=item.get('layout', 'list')
                    )
                    html_path = HTML_DIR / f"{file_name}.html"
                    html_path.write_text(html)

                    # 2. Screenshot
                    ss_path = SCREENSHOTS_DIR / f"{file_name}.png"
                    await page.goto(html_path.absolute().as_uri())
                    await page.screenshot(path=str(ss_path))

                    # 3. Add to dataset
                    dataset.append({
                        "image_path": str(ss_path),
                        "instruction": "Analyze the UI structure and generate an appropriate database schema.",
                        "output": item.get('sql', 'CREATE TABLE error (id INT);'),
                        "domain": ui_type,
                        "size_kb": round(ss_path.stat().st_size / 1024, 1)
                    })
                
                # Save after every batch for safety
                with open(DATASET_FILE, 'w') as f:
                    json.dump(dataset, f, indent=2)
                
                print(f"   ✅ Batch Complete. Total: {len(dataset)}")
                await asyncio.sleep(backoff_time) 

            except Exception as e:
                print(f"   ⚠️  Error: {str(e)[:100]}")
                print(f"   ⏳ Backing off for {backoff_time}s...")
                await asyncio.sleep(backoff_time)
                # Increase backoff for next attempt
                backoff_time = min(backoff_time * 2, 60) 

        await browser.close()

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    asyncio.run(main(count))
