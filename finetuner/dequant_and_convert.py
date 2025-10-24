#!/usr/bin/env python3
"""
Simple dequantization and conversion script.
Loads the model in full precision and saves it for llama.cpp conversion.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def main():
    model_path = "outputs/final_model_merged"
    output_path = "outputs/temp_model"

    print("=" * 60)
    print("🔄 Dequantizing Model for GGUF Conversion")
    print("=" * 60)

    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    print(f"📁 Loading model from: {model_path}")

    try:
        # Load model in full precision (dequantize)
        print("🔄 Loading model (this may take a minute)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=False,  # Force full precision
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"✅ Model loaded successfully: {type(model)}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Saving to: {output_path}")

        # Save dequantized model
        print("💾 Saving dequantized model...")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        print("✅ Dequantized model saved!")
        print(f"   Location: {output_path}")
        print("")
        print("📝 Next steps:")
        print(f"   1. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py {output_path} --outtype f16 --outfile outputs/model_fp16.gguf")
        print(f"   2. Quantize: ./llama.cpp/build/bin/quantize outputs/model_fp16.gguf outputs/model_q4_k_m.gguf Q4_K_M")
        print(f"   3. Test: ./test_gguf.sh outputs/model_q4_k_m.gguf")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

