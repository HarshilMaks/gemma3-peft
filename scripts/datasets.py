"""Automated UI Screenshot Generator using Kaggle CSVs, upgraded with robust rendering."""
import os
import re
import requests
import pandas as pd
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from tqdm import tqdm

# --- Configuration ---
CSV_FILES = [
    "data/raw_csvs/saas_companies.csv",
    "data/raw_csvs/ycombinator.csv",
    "data/raw_csvs/producthunt.csv"
]
GITHUB_URL = "https://raw.githubusercontent.com/obazoud/awesome-dashboard/master/README.md"

# Changed to screenshots2 so it doesn't mix with your good 136 images
OUTPUT_DIR = "data/ui_screenshots2"
MAX_URLS = 1000 # You can increase this!
MIN_FILE_SIZE_KB = 40  # Kills the blank/black/white screens

def clean_url(url):
    """Ensure the URL has http/https and is valid. Skip malformed/private URLs."""
    if pd.isna(url) or not isinstance(url, str):
        return None
    url = url.strip()
    if any(x in url.lower() for x in [',', '\n', '\r', ' mobile', 'javascript:', 'data:', '<', '>']):
        return None
    if not url.startswith('http'):
        url = 'https://' + url
    if any(x in url.lower() for x in ['localhost', '127.0.0.1', '192.168.', '10.0.', ':8080']):
        return None
    if any(x in url.lower() for x in ['utm_', 'r/[a-f0-9]{16}', 'ref=']):
        return None
    parsed = urlparse(url)
    if not parsed.netloc or '.' not in parsed.netloc:
        return None
    return url

def extract_github_urls():
    print(f"Fetching dashboard links directly from GitHub...")
    try:
        response = requests.get(GITHUB_URL)
        if response.status_code != 200:
            return []
        urls = re.findall(r'\[.*?\]\((https?://[^\)]+)\)', response.text)
        valid_ui_urls = [u for u in urls if "github.com" not in u and "twitter.com" not in u and "wikipedia.org" not in u]
        return valid_ui_urls
    except Exception:
        return []

def extract_all_urls():
    all_urls = set()
    all_urls.update(extract_github_urls())
    
    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            continue
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip', low_memory=False)
            url_col = next((col for col in df.columns if any(x in str(col).lower() for x in ['url', 'website', 'domain', 'link'])), None)
            if url_col:
                urls = df[url_col].apply(clean_url).dropna().tolist()
                all_urls.update(urls)
        except Exception:
            pass
            
    final_urls = list(all_urls)[:MAX_URLS]
    print(f"\nTotal unique URLs to attempt: {len(final_urls)}")
    return final_urls

def capture_screenshots(urls):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nStarting Robust Playwright to capture screenshots...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        success_count = 0
        for url in tqdm(urls, desc="Snapping UIs"):
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                safe_name = "".join(c for c in domain if c.isalnum() or c in ".-_")
                output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{abs(hash(url)) % 100000}.png")
                
                if os.path.exists(output_path):
                    continue
                
                # THE MODERN FIX: Wait for all API calls to finish
                response = page.goto(url, timeout=25000, wait_until="networkidle")
                
                if not response or not response.ok:
                    continue

                # THE MODERN FIX: Wait 3 seconds for animations
                page.wait_for_timeout(3000)
                
                page.evaluate("""
                    document.querySelectorAll('[id*="cookie"], [class*="cookie"], [id*="popup"], [class*="popup"], [id*="banner"], [class*="banner"]').forEach(el => el.remove());
                """)
                
                page.screenshot(path=output_path)
                
                # THE MODERN FIX: Delete blank white/black screens
                file_size_kb = os.path.getsize(output_path) / 1024
                if file_size_kb < MIN_FILE_SIZE_KB:
                    os.remove(output_path)
                    continue
                    
                success_count += 1
                
            except Exception:
                pass
                
        browser.close()
    print(f"\n✅ Finished! Captured {success_count} perfect screenshots to {OUTPUT_DIR}/")

if __name__ == "__main__":
    target_urls = extract_all_urls()
    if target_urls:
        capture_screenshots(target_urls)