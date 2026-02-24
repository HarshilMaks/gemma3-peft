"""Dataset download helpers for phase 1/2."""
"""Dataset download helpers for phase 1/2.
Automated UI Screenshot Generator using Playwright, Kaggle CSVs, and GitHub.
"""
import os
import re
import requests
import pandas as pd
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from tqdm import tqdm

# --- Configuration ---
# 1. Your downloaded CSVs inside your data folder
CSV_FILES = [
    "data/raw_csvs/saas_companies.csv",
    "data/raw_csvs/ycombinator.csv",
    "data/raw_csvs/producthunt.csv"
]

# 2. The GitHub "Awesome Dashboards" raw Markdown URL (THIS IS THE GITHUB EXTRACTOR)
GITHUB_URL = "https://raw.githubusercontent.com/obazoud/awesome-dashboard/master/README.md"

# Output directory for your screenshots
OUTPUT_DIR = "data/ui_screenshots"
MAX_URLS = 10000  # Total number of screenshots you want

def clean_url(url):
    """Ensure the URL has http/https and is valid."""
    if pd.isna(url) or not isinstance(url, str):
        return None
    url = url.strip()
    if not url.startswith('http'):
        url = 'https://' + url
    return url

def extract_github_urls():
    """Fetch and extract URLs from the GitHub Awesome Dashboards repository."""
    print(f"Fetching dashboard links directly from GitHub...")
    try:
        response = requests.get(GITHUB_URL)
        if response.status_code != 200:
            print("  -> Failed to fetch GitHub list.")
            return []
            
        # Regex to find standard Markdown links: [Name](https://link.com)
        urls = re.findall(r'\[.*?\]\((https?://[^\)]+)\)', response.text)
        
        # Filter out boring non-UI links (like wikipedia, twitter, github source code)
        valid_ui_urls = [
            u for u in urls 
            if "github.com" not in u and "twitter.com" not in u and "wikipedia.org" not in u
        ]
        
        print(f"  -> Found {len(valid_ui_urls)} valid UI links from GitHub Awesome Dashboards")
        return valid_ui_urls
    except Exception as e:
        print(f"  -> Error fetching GitHub links: {e}")
        return []

def extract_all_urls():
    """Extract URLs intelligently from CSVs AND GitHub."""
    all_urls = set()
    
    # 1. Execute the GitHub Extractor first
    github_urls = extract_github_urls()
    all_urls.update(github_urls)
    
    # 2. Extract from your local Kaggle CSVs
    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            print(f"Skipping {csv_file} (Not found in {csv_file})")
            continue
            
        print(f"Reading {csv_file}...")
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip', low_memory=False)
            
            # Auto-detect the website column
            url_col = None
            for col in df.columns:
                lower_col = str(col).lower()
                if 'url' in lower_col or 'website' in lower_col or 'domain' in lower_col or 'link' in lower_col:
                    url_col = col
                    break
            
            if url_col:
                urls = df[url_col].apply(clean_url).dropna().tolist()
                all_urls.update(urls)
                print(f"  -> Found {len(urls)} URLs in '{url_col}'")
            else:
                print(f"  -> Could not find a website column in {csv_file}")
                
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            
    # Convert set back to list (removes duplicates) and limit to MAX_URLS
    final_urls = list(all_urls)[:MAX_URLS]
    print(f"\nTotal unique URLs gathered (GitHub + CSVs): {len(final_urls)}")
    return final_urls

def capture_screenshots(urls):
    """Use Playwright to visit URLs and take screenshots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nStarting Playwright to capture screenshots...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()
        
        for url in tqdm(urls, desc="Snapping UIs"):
            try:
                # Clean up filename
                domain = urlparse(url).netloc.replace("www.", "")
                safe_name = "".join(c for c in domain if c.isalnum() or c in ".-_")
                output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
                
                # Skip if already downloaded
                if os.path.exists(output_path):
                    continue
                
                # Visit page and snap
                page.goto(url, timeout=15000, wait_until="networkidle")
                page.evaluate("""
                    document.querySelectorAll('[id*="cookie"], [class*="cookie"], [id*="popup"], [class*="popup"], [id*="banner"], [class*="banner"]').forEach(el => el.remove());
                """)
                page.screenshot(path=output_path)
                
            except Exception:
                # Silently skip websites that timeout or fail
                pass
                
        browser.close()
    print(f"\nFinished! Check your {OUTPUT_DIR} folder.")

if __name__ == "__main__":
    target_urls = extract_all_urls()
    if target_urls:
        capture_screenshots(target_urls)
    else:
        print("No URLs found. Check your sources.")