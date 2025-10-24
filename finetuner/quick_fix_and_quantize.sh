#!/bin/bash

# One-click fix and quantize script
# Fixes the missing configuration_deepseek.py issue and quantizes the model

set -e

echo "🔧 DeepSeek Model Fix & Quantize"
echo "=================================="
echo ""

# Check if merged model exists
if [ ! -d "outputs/final_model_merged" ]; then
    echo "❌ Error: outputs/final_model_merged not found"
    echo "   Make sure training completed successfully first"
    exit 1
fi

# Run the fix
echo "Step 1: Fixing missing configuration files..."
python fix_missing_config.py

echo ""
echo "Step 2: Quantizing the model..."
python src/run_finetune.py --skip_finetune --model_path outputs/final_model_merged

echo ""
echo "=================================="
echo "✅ All done!"
echo ""
echo "Your quantized model is ready:"
echo "   outputs/model_q4_k_m.gguf"
echo ""
echo "Test it with:"
echo "   ./test_gguf.sh"
echo ""

