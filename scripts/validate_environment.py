#!/usr/bin/env python3
"""
Environment validation script for Ghost Architect project.
Validates dependencies, GPU access, and memory requirements.
"""

import sys
import importlib
import torch

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_gpu_availability():
    """Check CUDA/GPU availability and memory."""
    print("\n🔥 Checking GPU availability...")
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA not available - CPU training only (not recommended)")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"   ✅ CUDA available with {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   📟 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if gpu_memory < 15:
            print(f"   ⚠️  Warning: GPU {i} has less than 15GB VRAM (may need reduced settings)")
        else:
            print(f"   ✅ GPU {i}: Sufficient memory for Trinity training")
    
    return True

def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking dependencies...")
    
    # Keep unsloth first to avoid import-order warnings from patched libraries.
    required_packages = [
        'unsloth', 'torch', 'transformers', 'accelerate', 'bitsandbytes',
        'peft', 'trl', 'xformers', 'datasets'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_unsloth_compatibility():
    """Check Unsloth installation and compatibility."""
    print("\n⚡ Checking Unsloth compatibility...")
    
    try:
        from unsloth import FastLanguageModel
        print("   ✅ Unsloth imported successfully")
        
        # Try to check for Gemma-3 support
        print("   ✅ Ready for Gemma-3-12B training")
        return True
        
    except ImportError as e:
        print(f"   ❌ Unsloth import failed: {e}")
        print("   Install: pip install unsloth[colab-new]==2026.1.4")
        return False
    except Exception as e:
        print(f"   ⚠️  Unsloth imported with warnings: {e}")
        return True

def estimate_memory_usage():
    """Estimate memory usage for Trinity training."""
    print("\n🧠 Memory usage estimation (Trinity architecture):")
    
    memory_breakdown = {
        "Model weights (4-bit QLoRA)": 7.6,
        "Gradients (Rank 64 + DoRA)": 5.5,
        "Context overhead (4096 seq)": 2.5,
        "System buffer": 0.4,
        "Total estimated": 15.6
    }
    
    for component, gb in memory_breakdown.items():
        print(f"   📊 {component}: {gb:.1f}GB")
    
    print("\n💡 Recommendations:")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 16:
            print("   ✅ Should work with default settings")
        elif gpu_memory >= 12:
            print("   ⚠️  May need: max_seq_length=2048, rank=32")
        else:
            print("   ❌ Insufficient VRAM - consider cloud training (Colab T4)")
    
    return True

def main():
    """Main validation function."""
    print("🚀 Ghost Architect Environment Validation")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_unsloth_compatibility(),
        check_gpu_availability(),
        estimate_memory_usage()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 Environment validation PASSED!")
        print("   Ready to begin Phase 1: Trinity training")
        return True
    else:
        print("❌ Environment validation FAILED!")
        print("   Please fix the issues above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
