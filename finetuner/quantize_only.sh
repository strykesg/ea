#!/bin/bash

# Quantize-only script
# Usage: ./quantize_only.sh [model_path]

set -e

MODEL_PATH="${1:-outputs/final_model_merged}"

echo "🔄 Quantization Script"
echo "======================"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo ""
    echo "Usage: ./quantize_only.sh [model_path]"
    echo ""
    echo "Examples:"
    echo "  ./quantize_only.sh outputs/final_model_merged"
    echo "  ./quantize_only.sh /path/to/your/model"
    exit 1
fi

echo "📁 Model path: $MODEL_PATH"
echo ""

# Run quantization
echo "🚀 Starting quantization to Q4_K_M..."
python src/run_finetune.py --skip_finetune --model_path "$MODEL_PATH"

echo ""
echo "✅ Done! Your quantized model is at:"
echo "   outputs/model_q4_k_m.gguf"
echo ""

