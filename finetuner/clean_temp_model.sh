#!/bin/bash

# Manual cleanup script for quantization artifacts
# Run this if the automatic cleanup in proper_dequantize.py fails

set -e

echo "🧹 Manual Cleanup of Quantization Artifacts"
echo "==========================================="

MODEL_PATH="outputs/temp_clean_model"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model directory not found: $MODEL_PATH"
    echo "   Make sure proper_dequantize.py has been run first"
    exit 1
fi

echo "📁 Cleaning model at: $MODEL_PATH"

# Run the cleanup script
python clean_quantization_artifacts.py "$MODEL_PATH"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📝 Now you can proceed with GGUF conversion:"
echo "   python llama.cpp/convert_hf_to_gguf.py $MODEL_PATH --outtype f16 --outfile outputs/model_fp16.gguf"
