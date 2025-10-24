#!/bin/bash

# Simple manual conversion script
# Run this if the Python scripts are having issues

set -e

echo "🔄 Simple GGUF Conversion"
echo "========================"
echo ""

# Step 1: Install dependencies
echo "📦 Installing dependencies..."
pip install torch transformers accelerate numpy sentencepiece protobuf

# Step 2: Dequantize model
echo ""
echo "🔄 Dequantizing model..."
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

model_path = 'outputs/final_model_merged'
output_path = 'outputs/temp_clean_model'

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    load_in_4bit=False,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print('Saving dequantized model...')
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# Clean config
config_path = Path(output_path) / 'config.json'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Remove quantization settings
    keys_to_remove = ['quantization_config', 'load_in_4bit', 'load_in_8bit']
    for key in keys_to_remove:
        if key in config:
            del config[key]

    config['torch_dtype'] = 'float16'
    config['model_type'] = 'deepseek_v2'

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

print(f'✅ Model saved to: {output_path}')
"

# Step 3: Convert to GGUF
echo ""
echo "🔄 Converting to GGUF..."
python llama.cpp/convert_hf_to_gguf.py \
    outputs/temp_clean_model \
    --outtype f16 \
    --outfile outputs/model_fp16.gguf

# Step 4: Quantize
echo ""
echo "🔄 Quantizing to Q4_K_M..."
./llama.cpp/build/bin/quantize \
    outputs/model_fp16.gguf \
    outputs/model_q4_k_m.gguf \
    Q4_K_M

# Step 5: Cleanup
echo ""
echo "🧹 Cleaning up..."
rm -rf outputs/temp_clean_model outputs/model_fp16.gguf

echo ""
echo "🎉 SUCCESS!"
echo "Your model is ready: outputs/model_q4_k_m.gguf"
ls -lh outputs/model_q4_k_m.gguf

echo ""
echo "🧪 Test it:"
echo "   ./test_gguf.sh"


