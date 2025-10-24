#!/usr/bin/env python3
"""
Inspect the tensor shapes and structure of a saved model.
This helps diagnose issues with corrupted or modified models.
"""

import os
import json
import torch
from pathlib import Path
import safetensors.torch

def inspect_model_structure(model_path):
    """Inspect the structure and shapes of tensors in a model."""
    print("=" * 70)
    print("🔍 Model Structure Inspection")
    print("=" * 70)
    print(f"📁 Inspecting model: {model_path}")

    # Check if config exists
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Config loaded: {config.get('model_type', 'unknown')}")
        print(f"   Hidden size: {config.get('hidden_size', 'N/A')}")
        print(f"   Intermediate size: {config.get('intermediate_size', 'N/A')}")
        print(f"   Num layers: {config.get('num_hidden_layers', 'N/A')}")
        print(f"   Vocab size: {config.get('vocab_size', 'N/A')}")

        # Check for quantization config
        if 'quantization_config' in config:
            print(f"   ⚠️  Has quantization_config: {config['quantization_config']}")
        else:
            print("   ℹ️  No quantization_config found")
    else:
        print("❌ No config.json found")
        return False

    # Find safetensors files
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensors_files:
        print("❌ No safetensors files found!")
        return False

    print(f"\n📋 Found {len(safetensors_files)} safetensors files:")
    for f in safetensors_files:
        print(f"   - {f.name}")

    # Inspect first few tensors from each file
    print(f"\n🔬 Inspecting tensor shapes:")

    total_tensors = 0
    suspicious_tensors = []

    for safetensors_file in safetensors_files:
        print(f"\n📄 {safetensors_file.name}:")
        try:
            state_dict = safetensors.torch.load_file(safetensors_file, device='cpu')
            file_tensors = len(state_dict)
            total_tensors += file_tensors
            print(f"   Contains {file_tensors} tensors")

            # Check ALL tensors for suspicious shapes
            for name, tensor in state_dict.items():
                # Check for suspicious shapes
                if len(tensor.shape) == 2:
                    rows, cols = tensor.shape
                    # Check for unusual aspect ratios or very large/small dimensions
                    if rows == 1 or cols == 1:
                        suspicious_tensors.append((name, tensor.shape, safetensors_file.name))
                    elif rows > 100000 or cols > 100000:
                        suspicious_tensors.append((name, tensor.shape, safetensors_file.name))
                    # Also check for element count mismatches with expected shapes
                    elif name.endswith('.weight') and tensor.numel() > 50000000:  # Very large tensors
                        suspicious_tensors.append((name, tensor.shape, safetensors_file.name))

            # Show first few tensors and their shapes
            count = 0
            for name, tensor in state_dict.items():
                if count >= 5:  # Show first 5 tensors per file
                    break

                shape_str = ' x '.join(str(s) for s in tensor.shape)
                size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                print(f"   {name}: {tensor.dtype}, shape [{shape_str}], size {size_mb:.1f}MB")

                count += 1

        except Exception as e:
            print(f"   ❌ Failed to load {safetensors_file}: {e}")

    print(f"\n📊 Summary:")
    print(f"   Total tensors: {total_tensors}")

    if suspicious_tensors:
        print(f"\n⚠️  Found {len(suspicious_tensors)} suspicious tensor shapes:")
        for name, shape, file in suspicious_tensors:
            print(f"   - {name} in {file}: {shape}")
    else:
        print("   ✅ No obviously suspicious tensor shapes found")

    # Try to load with transformers to see what happens
    print("\n🔄 Testing model loading...")

    # Skip loading test if we found corrupted tensors
    if suspicious_tensors:
        print("   ⏭️  Skipping model loading test due to suspicious tensors found")
        print("   ℹ️  Run remove_corrupted_tensor.py first to fix the model")
    else:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("   Loading model with transformers...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # Use CPU to avoid GPU memory issues
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("   ✅ Model loaded successfully!")
            print(f"   Model type: {type(model)}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Clean up
            del model

        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            print("   💡 Try running: python remove_corrupted_tensor.py outputs/final_model_merged")
            # Don't show full traceback unless requested

    return True

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
        success = inspect_model_structure(model_path)
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
