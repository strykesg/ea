#!/bin/bash

# Test script for GGUF models using llama.cpp
# Usage: ./test_gguf.sh [model_path] [prompt]

set -e

MODEL_PATH="${1:-outputs/model_q4_k_m.gguf}"
PROMPT="${2:-Explain the concept of recursion in programming and provide a simple example.}"

echo "🧪 Testing GGUF Model with llama.cpp"
echo "====================================="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if llama.cpp is available
if [ ! -f "llama.cpp/main" ]; then
    echo "📦 llama.cpp not found. Installing..."
    echo ""
    
    # Clone llama.cpp
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    
    cd llama.cpp
    
    # Build with CUDA support if available
    if command -v nvcc &> /dev/null; then
        echo "🔧 Building llama.cpp with CUDA support..."
        make LLAMA_CUDA=1 -j$(nproc)
    else
        echo "🔧 Building llama.cpp (CPU only)..."
        make -j$(nproc)
    fi
    
    cd ..
    echo "✅ llama.cpp installed!"
    echo ""
fi

# Run inference
echo "🚀 Running inference..."
echo ""

./llama.cpp/main \
    -m "$MODEL_PATH" \
    -p "$PROMPT" \
    -n 512 \
    --temp 0.7 \
    --top-p 0.9 \
    --top-k 50 \
    --repeat-penalty 1.1 \
    -ngl 99

echo ""
echo "✅ Test complete!"

