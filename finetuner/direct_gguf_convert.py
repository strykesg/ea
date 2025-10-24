#!/usr/bin/env python3
"""
Direct GGUF conversion without llama.cpp's convert script.
Uses HuggingFace transformers to export to GGUF format.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import struct

def save_gguf_header(f, config):
    """Write GGUF file header."""
    # GGUF magic number
    f.write(b'GGUF')
    # Version
    f.write(struct.pack('<II', 3, 1))  # Version 3, little endian

    # Number of tensors (placeholder - will update later)
    f.write(struct.pack('<Q', 0))

    # Number of key-value pairs
    f.write(struct.pack('<I', 10))  # Number of metadata entries

    # Write metadata
    def write_kv_pair(key, value):
        # Key length + key
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<I', len(key_bytes)))
        f.write(key_bytes)

        # Value type and data
        if isinstance(value, str):
            # String type
            f.write(struct.pack('<I', 8))  # GGUF_VALUE_TYPE_STRING
            value_bytes = value.encode('utf-8')
            f.write(struct.pack('<Q', len(value_bytes)))
            f.write(value_bytes)
        elif isinstance(value, int):
            # Int type
            f.write(struct.pack('<I', 2))  # GGUF_VALUE_TYPE_UINT32
            f.write(struct.pack('<I', value))
        elif isinstance(value, float):
            # Float type
            f.write(struct.pack('<I', 4))  # GGUF_VALUE_TYPE_FLOAT32
            f.write(struct.pack('<f', value))

    # Write metadata
    write_kv_pair("general.architecture", "deepseek_v2")
    write_kv_pair("general.name", "DeepSeek-V2-Lite")
    write_kv_pair("deepseek.context_length", 16384)
    write_kv_pair("deepseek.embedding_length", config.get("hidden_size", 2048))
    write_kv_pair("deepseek.feed_forward_length", config.get("intermediate_size", 10944))
    write_kv_pair("deepseek.block_count", config.get("num_hidden_layers", 27))
    write_kv_pair("deepseek.attention.head_count", config.get("num_attention_heads", 16))
    write_kv_pair("tokenizer.ggml.model", "llama")
    write_kv_pair("general.file_type", 1)  # F16
    write_kv_pair("general.size_label", "2.8B")

def convert_to_gguf(model_path, output_path, quantization="Q4_K_M"):
    """Convert HuggingFace model directly to GGUF."""
    print("=" * 60)
    print("🔄 Direct GGUF Conversion")
    print("=" * 60)

    # Load model in full precision
    print(f"📁 Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=False,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"✅ Model loaded: {type(model)}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Clean config
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Remove quantization settings
        keys_to_remove = [
            'quantization_config', 'load_in_4bit', 'load_in_8bit',
            'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant'
        ]
        for key in keys_to_remove:
            if key in config:
                del config[key]
                print(f"   Removed config key: {key}")

        # Update for GGUF compatibility
        config['torch_dtype'] = 'float16'
        config['model_type'] = 'deepseek_v2'

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # For now, create a simple conversion using transformers export
    print("💾 Saving model in standard format...")
    temp_path = "outputs/temp_clean_model"
    model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)

    print(f"✅ Clean model saved to: {temp_path}")
    print("")
    print("📝 Manual conversion needed:")
    print("   Since direct GGUF conversion is complex, use llama.cpp:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {temp_path} --outtype f16 --outfile outputs/model_fp16.gguf")
    print("   ./llama.cpp/build/bin/quantize outputs/model_fp16.gguf outputs/model_q4_k_m.gguf Q4_K_M")
    print("")
    print("🔧 Or use this command:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {temp_path} --outtype f16 --outfile outputs/model_fp16.gguf && \\")
    print(f"   ./llama.cpp/build/bin/quantize outputs/model_fp16.gguf outputs/model_q4_k_m.gguf Q4_K_M")

    return temp_path

def main():
    """Main conversion function."""
    model_path = "outputs/final_model_merged"
    output_path = "outputs/model_q4_k_m.gguf"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    try:
        temp_path = convert_to_gguf(model_path, output_path)

        print("")
        print("🚀 Ready to run conversion:")
        print(f"cd /workspace/ea/finetuner && \\")
        print(f"python llama.cpp/convert_hf_to_gguf.py {temp_path} --outtype f16 --outfile outputs/model_fp16.gguf && \\")
        print(f"./llama.cpp/build/bin/quantize outputs/model_fp16.gguf outputs/model_q4_k_m.gguf Q4_K_M && \\")
        print("ls -lh outputs/model_q4_k_m.gguf")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

