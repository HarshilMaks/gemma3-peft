# HuggingFace Authentication Setup for Phase 2

## Issue: Gated Model Access

The Gemini-3-12B model is **gated** on HuggingFace, meaning you need to:
1. Accept the license agreement from Google
2. Create an authentication token
3. Log in within Colab before training

---

## Step-by-Step Setup

### Step 1: Accept Model Access
1. Go to: https://huggingface.co/google/gemma-3-12b-it
2. Click the **"Access Repository"** button
3. Accept Google's license terms
4. Wait for approval (usually instant)

### Step 2: Create HuggingFace Token
1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Give it a name: "Ghost Architect Colab"
4. Select permission: **"repo"** (read-only is fine)
5. Click **"Create token"**
6. Copy the token (keep it secret!)

### Step 3: Update .env File
Add your HF token to `.env`:

```bash
# Google Gemini Vision API Key
GEMINI_API_KEY=AIzaSy...

# HuggingFace Token (for gated models like gemma-3-12b-it)
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
```

**Security:** 
- .env is in .gitignore (won't be committed)
- Keep token private, never share it
- You can rotate tokens anytime on HF website

### Step 4: Login in Colab

In your Colab notebook, add this cell **BEFORE** running training:

```python
# HuggingFace Authentication
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.environ.get("HUGGINGFACE_TOKEN")

if hf_token:
    login(token=hf_token)
    print("✅ Logged in to HuggingFace")
else:
    print("⚠️  HUGGINGFACE_TOKEN not found in .env")
    print("Please update .env with your HF token first")
```

Or use the interactive prompt:

```python
from huggingface_hub import notebook_login
notebook_login()  # Will show login dialog
```

### Step 5: Run Training

Now you can run Phase 2 training:

```python
!python src/train_vision.py --dataset data/dataset.json
```

---

## Testing Authentication

To verify your token works:

```python
from huggingface_hub import list_models, model_info

# This will fail if not authenticated
info = model_info("google/gemma-3-12b-it")
print(f"Model: {info.modelId}")
print(f"Downloads: {info.downloads}")
```

---

## Troubleshooting

### "401 Client Error: Unauthorized"
**Cause:** Token is invalid or expired
**Solution:** 
- Create a new token at https://huggingface.co/settings/tokens
- Update .env and redeploy

### "Access to model google/gemma-3-12b-it is restricted"
**Cause:** Haven't clicked "Access Repository" yet
**Solution:**
1. Go to https://huggingface.co/google/gemma-3-12b-it
2. Click "Access Repository"
3. Accept license terms
4. Wait ~1 minute for approval

### "GatedRepoError: You must have access to it"
**Cause:** Model access not granted yet
**Solution:** Same as above - click "Access Repository"

### Token works locally but not in Colab
**Cause:** Colab has cached old credentials
**Solution:**
```python
!rm -rf ~/.cache/huggingface/
from huggingface_hub import login
login(token="your_new_token")
```

---

## Security Best Practices

✅ **Do:**
- Keep token in .env (not in code)
- Use repo-level tokens with limited scope
- Rotate tokens periodically
- Regenerate token if accidentally exposed

❌ **Don't:**
- Share token in GitHub issues/discussions
- Commit .env file
- Hardcode token in notebooks
- Reuse tokens across projects

---

## Alternative: Use Quantized Version (No Auth)

If you want to skip authentication, use Phase 1 instead:

```python
# Phase 1: Uses unsloth quantized model (NO AUTH)
!python src/train.py --config configs/training_config.yaml --dataset data/dataset.json
```

This uses `unsloth/gemma-3-12b-it-bnb-4bit` which is freely available.

---

## More Info

- HuggingFace Docs: https://huggingface.co/docs/hub/security-tokens
- Gemini-3 Model Card: https://huggingface.co/google/gemma-3-12b-it
- Colab + HF Guide: https://huggingface.co/docs/hub/security-colab
