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
echo "Step 2: Robustly dequantizing model..."
python robust_dequantize.py

echo ""
echo "Step 2.5: Inspecting model structure..."
python inspect_model.py outputs/final_model_merged

echo ""
echo "Step 2.6: Remove known corrupted tensor..."
python remove_corrupted_tensor.py outputs/final_model_merged outputs/temp_clean_model

echo ""
echo "Step 2.7: Verify corrupted tensor removal..."
python find_problematic_tensor.py outputs/temp_clean_model

echo ""
echo "Step 2.8: Manual cleanup of artifacts (if needed)..."
./clean_temp_model.sh

echo ""
echo "Step 3: Converting to GGUF..."
python llama.cpp/convert_hf_to_gguf.py \
    outputs/temp_clean_model \
    --outtype f16 \
    --outfile outputs/model_fp16.gguf

echo ""
echo "Step 4: Quantizing to Q4_K_M..."
./llama.cpp/build/bin/quantize \
    outputs/model_fp16.gguf \
    outputs/model_q4_k_m.gguf \
    Q4_K_M

echo ""
echo "Step 5: Cleanup..."
rm -rf outputs/temp_clean_model outputs/model_fp16.gguf

echo ""
echo "=================================="
echo "✅ All done!"
echo ""
echo "Your quantized model is ready:"
echo "   outputs/model_q4_k_m.gguf"
echo ""
echo "Verify it exists:"
ls -lh outputs/model_q4_k_m.gguf
echo ""
echo "Test it with:"
echo "   ./test_gguf.sh"
echo ""

