#!/usr/bin/env python3
"""
Merge synthetic dataset with existing dataset_vision.json

Usage:
  python scripts/merge_datasets.py
"""

import json
from pathlib import Path

def merge_datasets():
    """Merge synthetic_dataset.json with existing dataset_vision.json"""
    
    synthetic_file = Path("data/synthetic_factory/synthetic_dataset.json")
    existing_file = Path("data/dataset_vision.json")
    output_file = Path("data/dataset_merged.json")
    
    print("\n📊 Merging datasets...")
    
    # Load synthetic
    if not synthetic_file.exists():
        print(f"   ❌ {synthetic_file} not found")
        print("   Run: python scripts/data_factory.py 5000")
        return
    
    with open(synthetic_file) as f:
        synthetic = json.load(f)
    print(f"   ✅ Loaded {len(synthetic)} synthetic items")
    
    # Load existing
    existing = []
    if existing_file.exists():
        with open(existing_file) as f:
            existing = json.load(f)
        print(f"   ✅ Loaded {len(existing)} existing items")
    
    # Merge (synthetic first = prioritize quality)
    merged = synthetic + existing
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n   📁 Saved {len(merged)} total items to {output_file}")
    print(f"\n   Next: Use {output_file} for training")
    print(f"   Example: python src/train_vision.py --dataset {output_file}")


if __name__ == "__main__":
    merge_datasets()
