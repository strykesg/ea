#!/bin/bash

# Fine-tune DeepSeek-V2-Lite with one command
# Usage: ./finetune.sh [options]

set -e  # Exit on any error

echo "🚀 DeepSeek-V2-Lite Fine-Tuner"
echo "=============================================="

# Check if data.jsonl exists
if [ ! -f "data.jsonl" ]; then
    echo "❌ Error: data.jsonl not found!"
    echo "Please create data.jsonl with your training data."
    echo "See example_data.jsonl for the required format."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "src/run_finetune.py" ]; then
    echo "❌ Error: Please run this script from the finetuner directory"
    exit 1
fi

# Set default options
BATCH_SIZE=2
MAX_STEPS=60
SEQ_LENGTH=16384

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --batch_size SIZE    Training batch size (default: 2)"
            echo "  --max_steps STEPS    Maximum training steps (default: 60)"
            echo "  --seq_length LENGTH  Maximum sequence length (default: 16384)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "📊 Configuration:"
echo "   Batch size: $BATCH_SIZE"
echo "   Max steps: $MAX_STEPS"
echo "   Sequence length: $SEQ_LENGTH"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

# Check for .env file
if [ -f ".env" ]; then
    echo "✅ Found .env file - environment variables will be loaded automatically"
    echo ""
else
    echo "ℹ️  No .env file found. You can create one with:"
    echo "   ./create_env.sh  (interactive)"
    echo "   or manually: cat > .env << 'EOF'"
    echo "   HF_TOKEN=your_token_here"
    echo "   WANDB_API_KEY=your_wandb_token_here"
    echo "   EOF"
    echo ""
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
if ! python -c "import torch, transformers, unsloth" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -e .
fi

echo "🎯 Starting fine-tuning..."
echo "   This may take a while depending on your hardware and dataset size."
echo "   Check finetune.log for detailed progress."
echo ""

# Run the fine-tuner
python src/run_finetune.py \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --seq_length "$SEQ_LENGTH" \
    "$@"

echo ""
echo "✅ Fine-tuning complete!"
echo "📁 Check the 'outputs' directory for your models:"
echo "   - outputs/final_model_merged/     # Full model ready for quantization"
echo "   - outputs/model_q4_k_m.gguf       # Quantized GGUF model"
echo ""
echo "🔧 Next steps:"
echo "   - Test your model with the provided inference scripts"
echo "   - Adjust training parameters if needed"
echo "   - Deploy the GGUF model with llama.cpp or similar tools"

