# 🔒 Quick Start: Secure API Setup

## Your .env file is ALREADY created and protected!

✅ **.env file exists** (not in Git)  
✅ **.env.example template exists** (shows what keys you need)  
✅ **python-dotenv installed** (loads .env automatically)  
✅ **src/synthetic_generator.py updated** (uses .env, not hardcoded keys)  

---

## Setup in 3 Steps

### Step 1: Get Your API Key
Go to: https://aistudio.google.com/app/apikey
- Click "Create API Key"
- Copy the key

### Step 2: Update Your .env File
```bash
nano .env
```

Find this line:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace with your actual key:
```
GEMINI_API_KEY=sk_abc123xyz789...
```

Save and exit (Ctrl+O, Enter, Ctrl+X)

### Step 3: Test It Works
```bash
uv run python src/synthetic_generator.py
```

You should see:
```
Found X screenshots. Starting extraction...
```

---

## Important Security Notes

✅ **Safe** — Files in Git:
- `.env.example` (template only, no real keys)
- `SECURITY.md` (how-to guide)
- All Python files (use `os.environ.get()`)

❌ **Protected** — Files NOT in Git:
- `.env` (your actual API keys)
- `.venv/` (local virtual environment)
- `data/dataset.json` (training data)

---

## If Something Goes Wrong

### "GEMINI_API_KEY not found"
→ Make sure .env file exists and has your key

### "Invalid API key"
→ Double-check you copied the key correctly (no spaces)

### "Accidentally committed .env?"
→ Don't panic! Rotate your key immediately in Google AI Studio

---

## Learn More
See `SECURITY.md` for complete security best practices.
