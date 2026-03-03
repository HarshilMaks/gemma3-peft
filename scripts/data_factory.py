"""
Data Factory: Generate synthetic UI/SQL pairs using Gemini API + Playwright

Simple, focused pipeline:
  1. Generate UI descriptions using Gemini (free tier)
  2. Render to HTML using Jinja2 + Tailwind
  3. Screenshot with Playwright
  4. Generate SQL schemas from screenshots
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import time

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)

OUTPUT_DIR = Path("data/synthetic_factory")
UI_DESCRIPTIONS = OUTPUT_DIR / "ui_descriptions.json"
HTML_DIR = OUTPUT_DIR / "html_pages"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
DATASET_FILE = OUTPUT_DIR / "synthetic_dataset.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_model():
    """Get best available Gemini model."""
    for model_name in ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']:
        try:
            return genai.GenerativeModel(model_name)
        except:
            continue
    return genai.GenerativeModel('gemini-1.5-pro')


def generate_ui_descriptions(count: int = 100) -> List[Dict]:
    """Generate diverse UI descriptions using Gemini."""
    print(f"\n🎨 Generating {count} UI descriptions...")
    
    if UI_DESCRIPTIONS.exists():
        with open(UI_DESCRIPTIONS) as f:
            existing = json.load(f)
        if len(existing) >= count:
            print(f"   Reusing {len(existing)} existing descriptions")
            return existing
    
    model = get_model()
    descriptions = []
    
    ui_types = [
        "User Profile Dashboard", "Product Listing with Filters",
        "Admin Analytics Panel", "Task Management App",
        "Social Media Feed", "Support Ticket System",
        "E-commerce Checkout", "Calendar Scheduler",
        "Invoice System", "Chat Application",
        "File Storage", "Settings Panel",
    ]
    
    for i in range(count):
        ui_type = ui_types[i % len(ui_types)]
        prompt = f"""Generate a JSON UI description for: {ui_type} (variant {i//len(ui_types) + 1})

Return ONLY valid JSON with fields: name, type, description, fields (list), layout, has_charts, has_forms
Make it realistic and different from previous ones.
"""
        
        try:
            resp = model.generate_content(prompt)
            json_str = resp.text
            
            for delim in ['```json', '```']:
                if delim in json_str:
                    json_str = json_str.split(delim)[1]
                    if '```' in json_str:
                        json_str = json_str.split('```')[0]
                    break
            
            ui_desc = json.loads(json_str.strip())
            descriptions.append(ui_desc)
            print(f"   ✅ {i+1}: {ui_desc.get('name', 'Unknown')}")
            time.sleep(1)
            
        except Exception as e:
            print(f"   ⚠️  {i+1}: {str(e)[:60]}")
    
    with open(UI_DESCRIPTIONS, 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    return descriptions


def render_html_pages(descriptions: List[Dict]) -> List[str]:
    """Convert descriptions to HTML files."""
    print(f"\n🌐 Rendering {len(descriptions)} HTML pages...")
    
    try:
        from jinja2 import Template
    except ImportError:
        print("   ⚠️  Installing jinja2...")
        os.system("uv pip install jinja2")
        from jinja2 import Template
    
    template_html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{{ ui.name }}</title>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-gray-50"><header class="bg-white border-b px-6 py-4">
<h1 class="text-2xl font-bold">{{ ui.name }}</h1></header>
<main class="max-w-6xl mx-auto px-6 py-8">
{% if ui.layout == 'card-grid' %}
<div class="grid grid-cols-3 gap-6">
{% for f in ui.fields[:9] %}<div class="bg-white rounded-lg border p-4">
<h3 class="font-semibold">{{ f }}</h3><div class="mt-3 h-8 bg-blue-100 rounded"></div>
</div>{% endfor %}</div>
{% elif ui.layout == 'two-column' %}
<div class="grid grid-cols-2 gap-6"><div class="bg-white rounded-lg border p-6">
<h2 class="text-lg font-semibold mb-4">Sidebar</h2>
{% for f in ui.fields[:5] %}<div class="mb-3"><label class="text-sm">{{ f }}</label>
<input type="text" class="w-full px-3 py-2 border rounded-md text-sm"></div>{% endfor %}
</div><div class="bg-white rounded-lg border p-6"><h2 class="text-lg font-semibold mb-4">Content</h2>
{% for i in range(5) %}<div class="mb-3 h-4 bg-gray-200 rounded"></div>{% endfor %}
</div></div>
{% else %}
<div class="bg-white rounded-lg border p-6"><h2 class="text-lg font-semibold mb-4">{{ ui.name }}</h2>
{% if ui.has_forms %}<form class="space-y-4">
{% for f in ui.fields[:6] %}<div><label class="block text-sm">{{ f }}</label>
<input type="text" class="w-full px-3 py-2 border rounded-md"></div>{% endfor %}
<button type="submit" class="w-full bg-blue-600 text-white py-2 rounded-md">Submit</button>
</form>{% else %}<div class="space-y-3">
{% for f in ui.fields[:10] %}<div class="flex justify-between py-3 border-b">
<span>{{ f }}</span><span class="text-gray-500">Value</span></div>{% endfor %}
</div>{% endif %}</div>{% endif %}</main></body></html>"""
    
    template = Template(template_html)
    files = []
    
    for i, ui in enumerate(descriptions):
        try:
            html = template.render(ui=ui, range=range)
            html_file = HTML_DIR / f"ui_{i:04d}.html"
            html_file.write_text(html)
            files.append(str(html_file))
            
            if (i + 1) % 20 == 0:
                print(f"   ✅ Generated {i+1}/{len(descriptions)}")
        except Exception as e:
            print(f"   ⚠️  Failed {i}: {str(e)[:60]}")
    
    return files


async def screenshot_pages(html_files: List[str]) -> List[str]:
    """Screenshot HTML pages with Playwright."""
    print(f"\n📸 Screenshotting {len(html_files)} pages...")
    
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("   ⚠️  Installing playwright...")
        os.system("uv pip install playwright && uv run playwright install")
        from playwright.async_api import async_playwright
    
    screenshots = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        
        for i, html_file in enumerate(html_files):
            try:
                url = Path(html_file).absolute().as_uri()
                await page.goto(url, wait_until="networkidle")
                
                ss_file = SCREENSHOTS_DIR / f"ui_{i:04d}.png"
                await page.screenshot(path=str(ss_file), full_page=False)
                screenshots.append(str(ss_file))
                
                if (i + 1) % 50 == 0:
                    print(f"   ✅ Screenshotted {i+1}/{len(html_files)}")
            except Exception as e:
                print(f"   ⚠️  Failed {i}: {str(e)[:60]}")
        
        await browser.close()
    
    return screenshots


def generate_sql_schemas(screenshots: List[str]) -> List[Dict]:
    """Generate SQL schemas from screenshots."""
    print(f"\n🗄️  Generating SQL for {len(screenshots)} screenshots...")
    
    model = get_model()
    prompt_sys = """Analyze this UI screenshot and generate a PostgreSQL schema.
Output ONLY valid CREATE TABLE statements. No explanations."""
    
    dataset = []
    
    for i, ss in enumerate(screenshots):
        try:
            img = Image.open(ss)
            resp = model.generate_content([prompt_sys, img])
            
            sql = resp.text.replace('```sql', '').replace('```', '').strip()
            
            dataset.append({
                "image_path": ss,
                "instruction": "Generate PostgreSQL schema from this UI",
                "output": sql,
                "domain": "synthetic",
                "size_kb": round(Path(ss).stat().st_size / 1024, 1)
            })
            
            if (i + 1) % 50 == 0:
                print(f"   ✅ Generated SQL for {i+1}/{len(screenshots)}")
            
            time.sleep(1)
        except Exception as e:
            print(f"   ⚠️  Failed {i}: {str(e)[:60]}")
    
    with open(DATASET_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"   📁 Saved to {DATASET_FILE}")
    return dataset


async def main(count: int = 100):
    """Run the complete pipeline."""
    print("\n" + "="*60)
    print(f"🏭 DATA FACTORY: Generating {count} synthetic UI/SQL pairs")
    print("="*60)
    
    descs = generate_ui_descriptions(count)
    if not descs:
        return
    
    htmls = render_html_pages(descs)
    if not htmls:
        return
    
    screenshots = await screenshot_pages(htmls)
    if not screenshots:
        return
    
    dataset = generate_sql_schemas(screenshots)
    
    print("\n" + "="*60)
    print(f"✅ COMPLETE: Generated {len(dataset)} synthetic UI/SQL pairs")
    print(f"   📁 {DATASET_FILE}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    asyncio.run(main(count))
