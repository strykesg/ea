#!/usr/bin/env python3
"""
Find the specific tensor causing the shape mismatch error.
"""

import os
import json
import torch
from pathlib import Path
import safetensors.torch
from transformers import AutoConfig

def find_problematic_tensor(model_path):
    """Find the tensor that's causing the shape mismatch."""
    print("=" * 60)
    print("🔍 Finding Problematic Tensor")
    print("=" * 60)

    # Load config to get expected shapes
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Expected shape for the failing tensor (from error message)
    # Error: copying a param with shape torch.Size([11206656, 1]) from checkpoint, the shape in current model is torch.Size([2048, 10944])
    problematic_shape = (11206656, 1)
    expected_shape = (2048, 10944)

    print(f"Looking for tensor with shape {problematic_shape} (should be {expected_shape})")

    # Find safetensors files
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensors_files:
        print("❌ No safetensors files found!")
        return

    found = False

    for safetensors_file in safetensors_files:
        print(f"\n📄 Checking {safetensors_file.name}...")
        try:
            state_dict = safetensors.torch.load_file(safetensors_file, device='cpu')

            for name, tensor in state_dict.items():
                if tuple(tensor.shape) == problematic_shape:
                    print(f"   🎯 FOUND: {name}")
                    print(f"   Shape: {tensor.shape}")
                    print(f"   Elements: {tensor.numel()}")
                    print(f"   Expected elements: {expected_shape[0] * expected_shape[1]}")
                    print(f"   File: {safetensors_file.name}")

                    # Check if this matches the expected layer pattern
                    if 'mlp.down_proj.weight' in name or 'down_proj' in name:
                        print("   📋 This appears to be an MLP down projection weight")
                        layer_match = name.split('layers.')[-1].split('.')[0] if 'layers.' in name else 'unknown'
                        print(f"   Layer: {layer_match}")

                    found = True
                    break

            if found:
                break

        except Exception as e:
            print(f"   ❌ Failed to load {safetensors_file}: {e}")

    if not found:
        print("❌ Problematic tensor not found in safetensors files")
        print("   It might be in a different format or location")

    return found

def main():
    """Main function."""
    model_path = "outputs/final_model_merged"

    if not Path(model_path).exists():
        print(f"❌ Model path not found: {model_path}")
        return 1

    try:
        found = find_problematic_tensor(model_path)
        return 0 if found else 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
