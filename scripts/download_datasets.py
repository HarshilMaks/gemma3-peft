"""
The Omni-Scraper: Automated UI Screenshot Generator.
Combines 2024+ GitHub API, Awesome-Dashboards, and Kaggle CSVs.
"""
import os
import re
import time
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
GITHUB_MD_URL = "https://raw.githubusercontent.com/obazoud/awesome-dashboard/master/README.md"
OUTPUT_DIR = "data/ui_screenshots" # Unified folder for ALL images
MAX_URLS = 2000
MIN_FILE_SIZE_KB = 40

def clean_url(url):
    """Filters out bad, local, or malformed URLs."""
    if pd.isna(url) or not isinstance(url, str): return None
    url = url.strip()
    if any(x in url.lower() for x in [',', '\n', '\r', 'javascript:', 'data:', '<', '>']): return None
    if not url.startswith('http'): url = 'https://' + url
    if any(x in url.lower() for x in ['localhost', '127.0.0.1', '192.168.', '10.0.', ':8080']): return None
    if not urlparse(url).netloc or '.' not in urlparse(url).netloc: return None
    return url

def get_api_urls():
    """Fetches modern 2024+ URLs from GitHub API."""
    print("Fetching fresh 2024+ URLs from GitHub API...")
    urls = set()
    queries = ["topic:dashboard pushed:>2024-01-01", "topic:admin-panel pushed:>2024-01-01", "topic:saas pushed:>2024-01-01"]
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "GhostArchitect"}
    
    for query in queries:
        for page in range(1, 6):
            try:
                resp = requests.get(f"https://api.github.com/search/repositories?q={query}&sort=updated&order=desc&per_page=100&page={page}", headers=headers)
                if resp.status_code == 200:
                    for item in resp.json().get("items", []):
                        hp = item.get("homepage")
                        if hp and hp.startswith("http") and "github.com" not in hp:
                            urls.add(hp.strip())
                elif resp.status_code == 403: time.sleep(10)
                time.sleep(3)
            except Exception: pass
    return urls

def get_legacy_urls():
    """Fetches URLs from CSVs and Markdown."""
    print("Fetching legacy URLs from CSVs and Markdown...")
    urls = set()
    try:
        resp = requests.get(GITHUB_MD_URL)
        if resp.status_code == 200:
            found = re.findall(r'\[.*?\]\((https?://[^\)]+)\)', resp.text)
            urls.update([u for u in found if "github.com" not in u and "twitter.com" not in u])
    except Exception: pass

    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file): continue
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip', low_memory=False)
            url_col = next((col for col in df.columns if any(x in str(col).lower() for x in ['url', 'website', 'domain'])), None)
            if url_col:
                urls.update(df[url_col].apply(clean_url).dropna().tolist())
        except Exception: pass
    return urls

def capture_screenshots(urls):
    """Bulletproof Playwright rendering."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nStarting Playwright for {len(urls)} URLs...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()
        
        success = 0
        for url in tqdm(urls, desc="Snapping UIs"):
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                safe_name = "".join(c for c in domain if c.isalnum() or c in ".-_")
                output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{abs(hash(url)) % 100000}.png")
                
                if os.path.exists(output_path): continue
                
                resp = page.goto(url, timeout=25000, wait_until="networkidle")
                if not resp or not resp.ok: continue
                page.wait_for_timeout(3000)
                
                page.evaluate("""document.querySelectorAll('[id*="cookie"], [id*="popup"]').forEach(el => el.remove());""")
                page.screenshot(path=output_path)
                
                if os.path.getsize(output_path) / 1024 < MIN_FILE_SIZE_KB:
                    os.remove(output_path)
                    continue
                success += 1
            except Exception: pass
        browser.close()
    print(f"\n✅ Finished! Captured {success} new screenshots.")

if __name__ == "__main__":
    final_urls = list(get_api_urls().union(get_legacy_urls()))[:MAX_URLS]
    if final_urls: capture_screenshots(final_urls)