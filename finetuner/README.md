# DeepSeek-Coder-V2-Lite-Instruct Fine-Tuner

A one-click script to fine-tune DeepSeek-Coder-V2-Lite-Instruct on your custom dataset and quantize it to GGUF format for efficient inference.

## 🚀 Quick Start

1. **Prepare your data** in `data.jsonl` format (see [Data Format](#data-format))
2. **Run the fine-tuner**:
   ```bash
   cd finetuner
   python src/run_finetune.py
   ```
3. **Get your quantized model** in `outputs/model_q4_k_m.gguf`

## 📋 Requirements

### Hardware Requirements
- **GPU**: At least 16GB VRAM recommended (24GB+ for best results)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space for models and outputs

### Software Requirements
- Python 3.11+
- CUDA-compatible GPU (optional but recommended)
- HuggingFace account with API token

## 🛠️ Installation

1. **Clone and setup**:
   ```bash
   cd finetuner
   pip install -e .
   ```

2. **Set environment variables** (optional but recommended):
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   export WANDB_API_KEY=your_wandb_token_here  # For experiment tracking
   ```

## 📊 Data Format

Your training data should be in `data.jsonl` format with the following structure:

```json
{"instruction": "Explain quantum computing", "input": "What is a qubit?", "output": "A qubit is a quantum bit..."}
{"instruction": "Write code", "input": "Create a Python function", "output": "def example_function():..."}
```

### Required Fields
- `instruction`: The task description or prompt
- `input`: Additional context (can be empty string)
- `output`: The expected response

### Data Format Example
```jsonl
{"id": 1, "instruction": "Explain the concept of 'reenactment' in history and its significance, while also providing an example from popular culture.", "input": "Could you please explain what reenactment is in history and why it's important? Also, can you give me an example from a popular movie or TV show?", "output": "Reenactment in history refers to the practice of recreating historical events, battles, or everyday life scenes for educational, commemorative, or entertainment purposes. Its significance lies in three key aspects: firstly, it helps to engage learners in history by making it more tangible and interactive, fostering a deeper understanding of the past; secondly, it preserves and shares historical knowledge with future generations; and lastly, it challenges our perception of the past by revealing the human element and emotions behind historical events. For instance, in the TV series 'Rome' (2005-2007), the episode 'Egeria' features a reenactment of Julius Caesar's funeral. This scene not only recreates a vital moment in Roman history but also encapsulates the political turmoil and emotional spectacle surrounding Caesar's death, offering a engaging and insightful learning experience for viewers.", "generated_at": "2025-10-23T10:10:43.880357"}
```

## ⚙️ Configuration

The fine-tuner is pre-configured for optimal performance:

### Model Configuration
- **Base Model**: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`
- **Sequence Length**: 16,384 tokens (supports long-form reasoning)
- **Sample Packing**: Enabled (improves training efficiency)
- **Quantization**: Q4_K_M (balanced size/quality)

### Training Configuration
- **Batch Size**: 2 per device (adjustable)
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 8e-5
- **Training Steps**: 60 (adjustable for your dataset size)

## 🎛️ Usage

### Basic Usage
```bash
# Run complete pipeline (fine-tuning + quantization)
python src/run_finetune.py
```

### Advanced Usage
```bash
# Custom batch size and steps
python src/run_finetune.py --batch_size 1 --max_steps 100

# Custom sequence length
python src/run_finetune.py --seq_length 8192

# Only run quantization on existing model
python src/run_finetune.py --skip_finetune --model_path outputs/final_model_merged

# Custom data file
python src/run_finetune.py --data_file my_training_data.jsonl
```

### Programmatic Usage
```python
from src.finetune import main as finetune_main
from src.quantize import main as quantize_main

# Fine-tune with custom config
config = {
    'max_seq_length': 16384,
    'sample_packing': True,
    'max_steps': 100,
    'per_device_train_batch_size': 1
}

output_path, merged_path = finetune_main(config)

# Quantize the result
quantize_main(merged_path, "outputs/model_q4_k_m.gguf", "Q4_K_M")
```

## 📁 Output Structure

After successful completion, you'll find:

```
outputs/
├── final_model/              # Fine-tuned model (LoRA weights)
├── final_model_merged/       # Merged model (full weights)
├── model_q4_k_m.gguf         # Quantized GGUF model (ready for inference)
└── checkpoints/              # Training checkpoints
```

## 🔧 Customization

### Adjusting for Your Hardware

**For GPUs with less VRAM (< 16GB)**:
```bash
python src/run_finetune.py --batch_size 1 --seq_length 8192
```

**For more powerful GPUs (> 24GB VRAM)**:
```bash
python src/run_finetune.py --batch_size 4 --max_steps 200
```

### Training Configuration
Edit `src/finetune.py` to modify:
- `max_seq_length`: Sequence length (8192-16384)
- `sample_packing`: Enable/disable sample packing
- `max_steps`: Number of training steps
- `learning_rate`: Training learning rate

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size and sequence length
python src/run_finetune.py --batch_size 1 --seq_length 8192
```

**Model Download Issues**:
```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here
python src/run_finetune.py
```

**Data Format Errors**:
- Ensure `data.jsonl` contains valid JSON on each line
- Check that all required fields (`instruction`, `input`, `output`) are present
- Verify the file encoding is UTF-8

**Dependency Issues**:
```bash
# Reinstall dependencies
pip uninstall -y torch transformers unsloth
pip install -e .
```

### Getting Help

1. Check the log file: `finetune.log`
2. Verify your data format matches the example
3. Ensure you have sufficient GPU memory
4. Check that all dependencies are installed correctly

## 📈 Performance Tips

### For Better Results
1. **More Data**: Use 1000+ high-quality examples
2. **Quality Data**: Ensure diverse, well-formatted training examples
3. **Longer Training**: Increase `max_steps` for better convergence
4. **Experiment Tracking**: Set `WANDB_API_KEY` for detailed training logs

### Memory Optimization
- Use `load_in_4bit=True` (default) for memory efficiency
- Reduce `max_seq_length` if running out of memory
- Use gradient accumulation instead of larger batch sizes

## 🔄 Model Architecture

The fine-tuner uses:
- **Unsloth**: For memory-efficient fine-tuning
- **LoRA**: Low-rank adaptation for parameter-efficient training
- **Sample Packing**: Efficient batching of variable-length sequences
- **4-bit Quantization**: During training for memory efficiency
- **Q4_K_M**: Final quantization for deployment

## 📝 License

This fine-tuner is designed for research and educational purposes. Ensure you comply with the original model's license terms.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Fine-tuning! 🎯**
