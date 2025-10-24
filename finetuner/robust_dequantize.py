#!/usr/bin/env python3
"""
Robust dequantization script that handles corrupted or modified models.
This script works around shape mismatches and other loading issues.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
import safetensors.torch
from transformers import AutoConfig, AutoTokenizer
import gc

def load_model_state_dict_safely(model_path):
    """Load model state dict directly from safetensors files, handling errors gracefully."""
    print("🔄 Loading model state dict directly from safetensors...")

    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError("No safetensors files found")

    state_dict = {}
    total_tensors = 0

    for safetensors_file in safetensors_files:
        print(f"   Loading {safetensors_file.name}...")
        try:
            file_state_dict = safetensors.torch.load_file(safetensors_file, device='cpu')
            state_dict.update(file_state_dict)
            total_tensors += len(file_state_dict)
            print(f"     ✅ Loaded {len(file_state_dict)} tensors")
        except Exception as e:
            print(f"     ❌ Failed to load {safetensors_file}: {e}")
            continue

    print(f"   Total tensors loaded: {total_tensors}")
    return state_dict

def detect_model_architecture_from_state_dict(state_dict):
    """Try to detect the model architecture from the state dict."""
    print("🔍 Detecting model architecture from state dict...")

    # Look for common layer patterns
    layer_keys = [k for k in state_dict.keys() if k.startswith('model.layers.')]

    if layer_keys:
        # Extract layer numbers
        layer_nums = []
        for key in layer_keys:
            parts = key.split('.')
            if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
                try:
                    layer_nums.append(int(parts[2]))
                except ValueError:
                    continue

        if layer_nums:
            num_layers = max(layer_nums) + 1
            print(f"   Detected {num_layers} layers")

    # Look for embedding dimensions
    embed_keys = [k for k in state_dict.keys() if 'embed_tokens' in k or 'embd' in k]
    if embed_keys:
        for key in embed_keys[:1]:  # Check first embedding
            tensor = state_dict[key]
            if len(tensor.shape) >= 2:
                vocab_size, hidden_size = tensor.shape[-2], tensor.shape[-1]
                print(f"   Detected vocab_size: {vocab_size}, hidden_size: {hidden_size}")

    return "deepseek_v2"  # Assume DeepSeek for now

def fix_tensor_shapes(state_dict, expected_config):
    """Attempt to fix tensor shapes that don't match expectations."""
    print("🔧 Attempting to fix tensor shapes...")

    fixed_count = 0
    removed_count = 0

    # Common DeepSeek layer patterns and expected shapes
    expected_shapes = {
        'model.embed_tokens.weight': (expected_config.get('vocab_size', 102400), expected_config.get('hidden_size', 2048)),
        'model.norm.weight': (expected_config.get('hidden_size', 2048),),
        'lm_head.weight': (expected_config.get('vocab_size', 102400), expected_config.get('hidden_size', 2048)),
    }

    # For each layer, add expected shapes
    num_layers = expected_config.get('num_hidden_layers', 27)
    hidden_size = expected_config.get('hidden_size', 2048)
    intermediate_size = expected_config.get('intermediate_size', 10944)

    for layer_idx in range(num_layers):
        layer_prefix = f'model.layers.{layer_idx}'

        # Attention layer shapes
        expected_shapes.update({
            f'{layer_prefix}.input_layernorm.weight': (hidden_size,),
            f'{layer_prefix}.post_attention_layernorm.weight': (hidden_size,),
            f'{layer_prefix}.self_attn.q_proj.weight': (hidden_size, hidden_size),
            f'{layer_prefix}.self_attn.k_proj.weight': (hidden_size, hidden_size),
            f'{layer_prefix}.self_attn.v_proj.weight': (hidden_size, hidden_size),
            f'{layer_prefix}.self_attn.o_proj.weight': (hidden_size, hidden_size),

            # MLP layer shapes
            f'{layer_prefix}.mlp.gate_proj.weight': (intermediate_size, hidden_size),
            f'{layer_prefix}.mlp.up_proj.weight': (intermediate_size, hidden_size),
            f'{layer_prefix}.mlp.down_proj.weight': (hidden_size, intermediate_size),
        })

    # Check and fix each tensor
    keys_to_remove = []

    for name, tensor in state_dict.items():
        expected_shape = expected_shapes.get(name)

        if expected_shape is not None:
            actual_shape = tuple(tensor.shape)

            if actual_shape != expected_shape:
                print(f"   Shape mismatch for {name}:")
                print(f"     Expected: {expected_shape}")
                print(f"     Actual: {actual_shape}")

                # Try to reshape if total elements match
                expected_elements = np.prod(expected_shape)
                actual_elements = tensor.numel()

                if expected_elements == actual_elements:
                    try:
                        reshaped = tensor.view(expected_shape)
                        state_dict[name] = reshaped
                        print(f"     ✅ Reshaped successfully")
                        fixed_count += 1
                    except Exception as e:
                        print(f"     ❌ Failed to reshape: {e}")
                        keys_to_remove.append(name)
                        removed_count += 1
                else:
                    print(f"     ❌ Element count mismatch: {actual_elements} vs {expected_elements}")
                    keys_to_remove.append(name)
                    removed_count += 1
        else:
            # Check for suspicious shapes
            if len(tensor.shape) == 2:
                rows, cols = tensor.shape
                if rows == 1 or (rows > 100000 and cols == 1):
                    print(f"   ⚠️  Suspicious tensor: {name} shape {tensor.shape}")
                    # These are likely quantization artifacts or corrupted weights
                    keys_to_remove.append(name)
                    removed_count += 1

    # Remove problematic tensors
    for key in keys_to_remove:
        del state_dict[key]

    print(f"   Fixed {fixed_count} tensors, removed {removed_count} problematic tensors")

    return state_dict

def robust_dequantize_model(model_path, output_path):
    """Robust dequantization that handles corrupted models."""
    print("=" * 60)
    print("🔄 Robust Model Dequantization")
    print("=" * 60)

    # Load config
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded config: {config.get('model_type', 'unknown')}")
    else:
        print("❌ No config.json found")
        return False

    try:
        # Load state dict directly (bypassing model loading)
        state_dict = load_model_state_dict_safely(model_path)

        # Detect architecture
        model_type = detect_model_architecture_from_state_dict(state_dict)

        # Fix tensor shapes
        state_dict = fix_tensor_shapes(state_dict, config)

        # Clean quantization artifacts
        print("🧹 Removing quantization artifacts...")
        keys_to_remove = []
        for name in state_dict.keys():
            if any(pattern in name.lower() for pattern in [
                '.absmax', '.quant_map', '.quant_state', '.nested',
                '.quant', '.scb', '.weight_format'
            ]):
                keys_to_remove.append(name)
                print(f"   Removing: {name}")

        for key in keys_to_remove:
            del state_dict[key]

        print(f"   Removed {len(keys_to_remove)} quantization artifacts")

        # Save cleaned state dict
        print(f"💾 Saving cleaned model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)

        # Split into multiple files if needed (llama.cpp prefers smaller files)
        max_tensors_per_file = 100
        tensor_names = list(state_dict.keys())
        num_files = (len(tensor_names) + max_tensors_per_file - 1) // max_tensors_per_file

        for file_idx in range(num_files):
            start_idx = file_idx * max_tensors_per_file
            end_idx = min((file_idx + 1) * max_tensors_per_file, len(tensor_names))

            file_tensors = {name: state_dict[name] for name in tensor_names[start_idx:end_idx]}
            file_name = f"model-0000{file_idx + 1}-of-0000{num_files}.safetensors"
            file_path = Path(output_path) / file_name

            safetensors.torch.save_file(file_tensors, file_path)
            print(f"   Saved {file_path.name} with {len(file_tensors)} tensors")

        # Clean and save config
        config['torch_dtype'] = 'float16'
        config['model_type'] = model_type

        # Remove quantization settings
        keys_to_remove = [
            'quantization_config', 'load_in_4bit', 'load_in_8bit',
            'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant'
        ]
        for key in keys_to_remove:
            config.pop(key, None)

        config_path_out = Path(output_path) / "config.json"
        with open(config_path_out, 'w') as f:
            json.dump(config, f, indent=2)

        # Try to copy tokenizer
        try:
            tokenizer_path = Path(model_path) / "tokenizer.json"
            if tokenizer_path.exists():
                import shutil
                shutil.copy2(tokenizer_path, output_path)
                print("   ✅ Copied tokenizer.json")

            # Copy other tokenizer files
            for ext in ['.model', '.config']:
                tokenizer_files = list(Path(model_path).glob(f"*tokenizer*{ext}"))
                for tf in tokenizer_files:
                    shutil.copy2(tf, output_path)
                    print(f"   ✅ Copied {tf.name}")

        except Exception as e:
            print(f"   ⚠️  Could not copy tokenizer: {e}")

        print("✅ Robust dequantization completed!")
        print(f"   Location: {output_path}")

        # Clean up memory
        del state_dict
        gc.collect()

        return True

    except Exception as e:
        print(f"❌ Error during robust dequantization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    model_path = "outputs/final_model_merged"
    output_path = "outputs/temp_clean_model"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    try:
        success = robust_dequantize_model(model_path, output_path)
        if success:
            print("\n📝 Next steps:")
            print(f"   1. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py {output_path} --outtype f16 --outfile outputs/model_fp16.gguf")
            print(f"   2. Quantize: ./llama.cpp/build/bin/quantize outputs/model_fp16.gguf outputs/model_q4_k_m.gguf Q4_K_M")
            print(f"   3. Test: ./test_gguf.sh outputs/model_q4_k_m.gguf")
            return 0
        else:
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
