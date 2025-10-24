#!/usr/bin/env python3
"""
Check if a model has bitsandbytes quantization.
"""

import json
import torch
from transformers import AutoModelForCausalLM
from pathlib import Path

def check_model_quantization(model_path):
    """Check if a model has quantization config and what type."""
    print(f"🔍 Checking quantization status for: {model_path}")

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print("❌ No config.json found")
        return False

    with open(config_path, 'r') as f:
        config = json.load(f)

    has_quantization = 'quantization_config' in config
    print(f"   Has quantization_config: {has_quantization}")

    if has_quantization:
        quant_config = config['quantization_config']
        print(f"   Quantization type: {quant_config.get('quant_method', 'unknown')}")
        print(f"   Load in 4bit: {quant_config.get('load_in_4bit', False)}")
        print(f"   Load in 8bit: {quant_config.get('load_in_8bit', False)}")

    # Try to load a small part to check actual weight types
    try:
        print("   Checking actual weight types...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load on CPU to avoid GPU issues
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Check first few parameters
        sample_params = list(model.named_parameters())[:3]
        for name, param in sample_params:
            print(f"   {name}: {param.dtype}, shape: {param.shape}")

        has_bnb_quantization = any('bitsandbytes' in str(type(param)) for _, param in sample_params)
        print(f"   BitsAndBytes quantization detected: {has_bnb_quantization}")

    except Exception as e:
        print(f"   Could not load model to check weights: {e}")

    return has_quantization

def main():
    """Main function."""
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "outputs/final_model_merged"

    if not Path(model_path).exists():
        print(f"❌ Model path not found: {model_path}")
        return 1

    try:
        check_model_quantization(model_path)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
