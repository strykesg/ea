#!/bin/bash
# Helper script to create .env file

echo "Creating .env file for fine-tuner..."
echo ""
echo "Please enter your HuggingFace token (get from https://huggingface.co/settings/tokens):"
read -s hf_token
echo ""

echo "Please enter your WandB token (optional, get from https://wandb.ai/authorize):"
read -s wandb_token
echo ""

# Create .env file
cat > .env << ENV_EOF
# Environment Variables for DeepSeek-V2-Lite Fine-Tuner
HF_TOKEN=$hf_token
WANDB_API_KEY=$wandb_token
ENV_EOF

echo "✅ .env file created successfully!"
echo "📁 File location: $(pwd)/.env"
echo ""
echo "Contents:"
cat .env

echo ""
echo "🔒 Note: .env file is automatically ignored by git (.gitignore)"
echo "🚀 Ready to run: python src/run_finetune.py"
