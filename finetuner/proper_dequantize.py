#!/usr/bin/env python3
"""
Proper dequantization script for bitsandbytes quantized models.
This script actually converts quantized weights back to full precision.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import gc

def dequantize_bitsandbytes_model(model_path, output_path):
    """Properly dequantize a bitsandbytes quantized model to full precision."""
    print("=" * 60)
    print("🔄 Proper BitsAndBytes Dequantization")
    print("=" * 60)

    print(f"📁 Loading quantized model from: {model_path}")

    # First, check if the model has quantization config
    config_path = Path(model_path) / "config.json"
    has_quantization = False

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        has_quantization = 'quantization_config' in config
        print(f"   Quantization config found: {has_quantization}")

    if not has_quantization:
        print("   No quantization config found. Model may already be dequantized.")
        print("   Proceeding with standard loading...")

        # Load normally
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("   Loading with bitsandbytes quantization enabled...")

        # Load with quantization enabled first
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        print("   Dequantizing weights to float16...")

        # Dequantize each parameter
        for name, param in model.named_parameters():
            if hasattr(param, 'dtype') and param.dtype != torch.float16:
                # This parameter is quantized, dequantize it
                param.data = param.data.dequantize().to(torch.float16)
                print(f"   Dequantized: {name}")

        # Clear any quantization-related attributes
        for module in model.modules():
            if hasattr(module, '_hf_hook'):
                delattr(module, '_hf_hook')

        print("   Dequantization complete!")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"✅ Model loaded and dequantized: {type(model)}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Clean config
    print("🔧 Cleaning config...")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Remove all quantization-related settings
        keys_to_remove = [
            'quantization_config', 'load_in_4bit', 'load_in_8bit',
            'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant',
            'torch_dtype'  # We'll set this properly
        ]
        for key in keys_to_remove:
            if key in config:
                del config[key]
                print(f"   Removed: {key}")

        # Set proper dtype
        config['torch_dtype'] = 'float16'

        # Save cleaned config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Save dequantized model
    print(f"💾 Saving dequantized model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Also save the cleaned config to output path
    output_config_path = Path(output_path) / "config.json"
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Dequantized model saved!")
    print(f"   Location: {output_path}")

    # Clean up memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_path

def main():
    """Main function."""
    model_path = "outputs/final_model_merged"
    output_path = "outputs/temp_clean_model"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    try:
        dequantized_path = dequantize_bitsandbytes_model(model_path, output_path)

        print("")
        print("📝 Next steps:")
        print(f"   1. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py {dequantized_path} --outtype f16 --outfile outputs/model_fp16.gguf")
        print(f"   2. Quantize: ./llama.cpp/build/bin/quantize outputs/model_fp16.gguf outputs/model_q4_k_m.gguf Q4_K_M")
        print(f"   3. Test: ./test_gguf.sh outputs/model_q4_k_m.gguf")

        return 0

    except Exception as e:
        print(f"❌ Error during dequantization: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
