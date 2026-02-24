"""Dataset loading and preprocessing utilities (to be implemented)."""
from datasets import load_dataset
import json, os, re

   def clean(x):
       x = "" if x is None else str(x)
       x = re.sub(r"[ \t]+", " ", x).strip()
       return x

   # Good starter dataset for Phase 1
   ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

   rows = []
   for ex in ds:
       inst = clean(ex.get("instruction"))
       inp = clean(ex.get("input"))
       out = clean(ex.get("output"))
       if inst and out:  # keep only valid examples
           rows.append({
               "instruction": inst,
               "input": inp,
               "output": out
           })

   # First training run size (fast + stable)
   rows = rows[:300]

   os.makedirs("data", exist_ok=True)
   with open("data/dataset.json", "w", encoding="utf-8") as f:
       json.dump(rows, f, ensure_ascii=False, indent=2)

   print(f"Saved {len(rows)} examples to data/dataset.json")

  Then validate it with this cell (must pass before training):

   import json, random

   p = "data/dataset.json"
   with open(p, "r", encoding="utf-8") as f:
       data = json.load(f)

   assert isinstance(data, list) and len(data) >= 50, "Need at least 50 examples"
   for i, ex in enumerate(data):
       assert isinstance(ex, dict), f"Row {i} is not an object"
       assert ex.get("instruction", "").strip(), f"Row {i} instruction is empty"
       assert ex.get("output", "").strip(), f"Row {i} output is empty"
       ex.setdefault("input", "")

   print("dataset.json valid ✅")
   print("Count:", len(data))
   print("Sample:", random.choice(data))