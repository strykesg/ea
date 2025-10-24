#!/usr/bin/env python3
"""
Thoroughly clean all bitsandbytes quantization artifacts from a model.
This removes quantization metadata tensors that cause issues with llama.cpp conversion.
"""

import os
import json
import torch
from pathlib import Path
import safetensors.torch
from transformers import AutoConfig
import gc

def clean_quantization_artifacts(model_path, output_path=None):
    """Remove all quantization artifacts from a model."""
    if output_path is None:
        output_path = model_path

    print("=" * 60)
    print("🧹 Cleaning Quantization Artifacts")
    print("=" * 60)
    print(f"📁 Processing model: {model_path}")
    print(f"💾 Output path: {output_path}")

    # Load config to understand the model
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded config: {config.get('model_type', 'unknown')}")

    # Find all safetensors files
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensors_files:
        print("❌ No safetensors files found!")
        return False

    print(f"📋 Found {len(safetensors_files)} safetensors files")

    # Process each safetensors file
    total_cleaned = 0
    total_tensors = 0

    for safetensors_file in safetensors_files:
        print(f"\n🔄 Processing: {safetensors_file.name}")

        # Load the state dict
        try:
            state_dict = safetensors.torch.load_file(safetensors_file)
            original_count = len(state_dict)
            total_tensors += original_count
            print(f"   Loaded {original_count} tensors")
        except Exception as e:
            print(f"   ❌ Failed to load {safetensors_file}: {e}")
            continue

        # Identify tensors to remove (quantization artifacts)
        tensors_to_remove = []

        # Common quantization artifact patterns
        quant_patterns = [
            '.absmax',           # Absolute maximum values
            '.quant_map',        # Quantization mapping
            '.quant_state',      # Quantization state
            '.nested_absmax',    # Nested quantization metadata
            '.nested_quant_map', # Nested quantization mapping
            '.nested_quant_state', # Nested quantization state
            '.quant_state.bitsandbytes__',  # BitsAndBytes specific
            '.SCB',              # Some quantization metadata
            '.weight_format',    # Weight format metadata
        ]

        for tensor_name in state_dict.keys():
            # Check if this tensor matches any quantization pattern
            is_quant_artifact = False
            for pattern in quant_patterns:
                if pattern in tensor_name:
                    is_quant_artifact = True
                    break

            # Also check for common quantization tensor names
            if any(keyword in tensor_name.lower() for keyword in [
                'quant', 'absmax', 'quant_map', 'quant_state',
                'nested', 'scb', 'weight_format'
            ]):
                is_quant_artifact = True

            if is_quant_artifact:
                tensors_to_remove.append(tensor_name)
                print(f"   🗑️  Removing: {tensor_name}")

        # Remove the quantization tensors
        for tensor_name in tensors_to_remove:
            del state_dict[tensor_name]

        cleaned_count = len(state_dict)
        removed_count = original_count - cleaned_count
        total_cleaned += removed_count

        print(f"   ✅ Cleaned: {removed_count} artifacts removed, {cleaned_count} tensors remaining")

        # Save the cleaned state dict
        if output_path != model_path:
            os.makedirs(output_path, exist_ok=True)
            output_file = Path(output_path) / safetensors_file.name
        else:
            # Overwrite in place (backup the original first)
            backup_file = safetensors_file.with_suffix('.safetensors.backup')
            if not backup_file.exists():
                safetensors_file.rename(backup_file)
                print(f"   💾 Backed up original to: {backup_file.name}")
            output_file = safetensors_file

        try:
            safetensors.torch.save_file(state_dict, output_file)
            print(f"   ✅ Saved cleaned file: {output_file.name}")
        except Exception as e:
            print(f"   ❌ Failed to save {output_file}: {e}")
            continue

    # Clean the config.json to remove quantization settings
    if config_path.exists():
        print("\n🔧 Cleaning config.json...")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Remove quantization-related settings
        keys_to_remove = [
            'quantization_config',
            'load_in_4bit', 'load_in_8bit',
            'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant',
            'torch_dtype'  # Will be set properly below
        ]

        removed_keys = []
        for key in keys_to_remove:
            if key in config:
                del config[key]
                removed_keys.append(key)

        if removed_keys:
            print(f"   Removed config keys: {', '.join(removed_keys)}")

        # Ensure proper torch_dtype
        config['torch_dtype'] = 'float16'

        # Save cleaned config
        if output_path != model_path:
            output_config_path = Path(output_path) / "config.json"
            with open(output_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"   ✅ Saved cleaned config to: {output_config_path}")
        else:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("   ✅ Updated config.json in place")

    print("\n📊 Summary:")
    print(f"   Total tensors processed: {total_tensors}")
    print(f"   Quantization artifacts removed: {total_cleaned}")
    print(f"   Clean tensors remaining: {total_tensors - total_cleaned}")

    if total_cleaned > 0:
        print("✅ Model cleaned successfully!")
        return True
    else:
        print("ℹ️  No quantization artifacts found - model was already clean")
        return True

def main():
    """Main function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python clean_quantization_artifacts.py <model_path> [output_path]")
        print("  model_path: Path to the model to clean")
        print("  output_path: Optional output path (defaults to modifying model_path in place)")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(model_path).exists():
        print(f"❌ Model path not found: {model_path}")
        sys.exit(1)

    try:
        success = clean_quantization_artifacts(model_path, output_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
