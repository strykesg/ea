# 🚀 Quick Reference Card

## After Training Completes

### 1️⃣ What to Download
```bash
# Main file (REQUIRED)
scp outputs/model_q4_k_m.gguf your_laptop:/path/to/models/

# Backup (optional - for re-quantizing)
tar -czf backup.tar.gz outputs/final_model_merged/
scp backup.tar.gz your_laptop:/path/to/models/
```

### 2️⃣ Test the Model
```bash
# Quick test
./test_gguf.sh

# Custom prompt
./test_gguf.sh outputs/model_q4_k_m.gguf "Your prompt here"
```

### 3️⃣ Cleanup
```bash
cd outputs/
rm -rf checkpoints/ final_model/
# Keep: model_q4_k_m.gguf (and optionally final_model_merged/)
```

---

## Running Your Model

### Easiest: Ollama
```bash
# Setup (once)
curl -fsSL https://ollama.ai/install.sh | sh
cat > Modelfile << 'EOF'
FROM ./outputs/model_q4_k_m.gguf
EOF
ollama create my-model -f Modelfile

# Use
ollama run my-model "Your question"
```

### Best for Development: llama.cpp
```bash
# Setup (once)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make LLAMA_CUDA=1 -j$(nproc)

# Use
./main -m ../outputs/model_q4_k_m.gguf -p "Your question" -ngl 99
```

### Best for GUI: LM Studio
1. Download from https://lmstudio.ai/
2. Import `model_q4_k_m.gguf`
3. Chat!

---

## Re-Quantize (If Needed)

```bash
# Different quantization levels
python -c "
from src.quantize import main as quantize_main

# Q8_0 - Maximum quality (larger file)
quantize_main('outputs/final_model_merged', 'outputs/model_q8.gguf', 'Q8_0')

# Q4_0 - Ultra-fast (smaller file)
quantize_main('outputs/final_model_merged', 'outputs/model_q4.gguf', 'Q4_0')
"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use Q4_0 or reduce context length |
| Gibberish output | Check training logs, verify data format |
| File not found | Check `outputs/` directory exists |
| Slow inference | Enable GPU offloading (`-ngl 99`) |

---

## File Sizes Reference

| File | Size | Purpose |
|------|------|---------|
| `model_q4_k_m.gguf` | ~2-3GB | ✅ Production (recommended) |
| `final_model_merged/` | ~10-15GB | Full precision backup |
| Q8_0 quantization | ~5-6GB | Maximum quality |
| Q4_0 quantization | ~1.5GB | Ultra-fast |

---

## Training Parameters (3 Epochs)

```bash
# Default (H200)
python src/run_finetune.py --batch_size 24 --max_steps 621

# Consumer GPU (RTX 5090/4090)
python src/run_finetune.py --batch_size 4 --max_steps 621

# Low VRAM
python src/run_finetune.py --batch_size 1 --max_steps 621 --seq_length 8192
```

**Steps calculation:** `(num_samples / batch_size) * epochs = (4962 / 24) * 3 = 621`

---

## Full Guides

- **Installation & Training:** [README.md](README.md)
- **Post-Training & Deployment:** [POST_TRAINING_GUIDE.md](POST_TRAINING_GUIDE.md)

---

**Questions? Check the full guides above! 📚**

