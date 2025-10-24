# 📦 Post-Training Guide: What to Keep & How to Use Your Model

## 🎯 Quick Answer

**Main file to download:** `outputs/model_q4_k_m.gguf` (~2-3GB)

This is your production-ready, quantized model that can run on llama.cpp, Ollama, LM Studio, and other GGUF-compatible inference engines.

---

## 📁 Understanding Your Output Files

After training completes, here's what's in the `outputs/` directory:

| File/Folder | Size | Purpose | Action |
|-------------|------|---------|--------|
| **`model_q4_k_m.gguf`** | ~2-3GB | Quantized model (Q4_K_M) ready for production | ✅ **Download & Keep** |
| `final_model_merged/` | ~10-15GB | Full precision Hugging Face model | ⚠️ Keep if you want to re-quantize later |
| `final_model/` | ~200MB | LoRA adapters only (already merged into above) | ❌ Delete (not needed) |
| `checkpoints/` | Varies | Training checkpoints (intermediate states) | ❌ Delete (not needed after success) |

### 💾 Download Commands

```bash
# On your training server, compress and download

# Option 1: Just the GGUF (recommended - smallest)
cd /workspace/ea/finetuner/outputs
scp model_q4_k_m.gguf your_laptop:/path/to/models/

# Option 2: GGUF + full model (for re-quantizing later)
tar -czf finetuned_models.tar.gz model_q4_k_m.gguf final_model_merged/
scp finetuned_models.tar.gz your_laptop:/path/to/models/

# Option 3: Using rsync (resume support)
rsync -avz --progress outputs/model_q4_k_m.gguf your_laptop:/path/to/models/
```

---

## 🔄 Quantization Guide

### Automatic Quantization (Default)

The training script **automatically quantizes** to Q4_K_M after fine-tuning. No extra steps needed!

### Manual Re-Quantization

If you want a different quantization level:

```bash
# From the finetuner directory
python src/run_finetune.py --skip_finetune --model_path outputs/final_model_merged

# Or with Python directly
python -c "
from src.quantize import main as quantize_main
quantize_main('outputs/final_model_merged', 'outputs/model_q8.gguf', 'Q8_0')
"
```

### Quantization Levels Comparison

| Level | Size | Quality | Speed | VRAM | Use Case |
|-------|------|---------|-------|------|----------|
| **Q4_K_M** | ~2-3GB | Good | Fast | 4-6GB | ✅ **Recommended** - Balanced |
| **Q5_K_M** | ~3-4GB | Better | Medium | 6-8GB | Higher quality, still fast |
| **Q8_0** | ~5-6GB | Best | Slower | 8-10GB | Maximum quality |
| **Q4_0** | ~1.5GB | Lower | Fastest | 3-4GB | Ultra-fast/low-resource |
| **Q2_K** | ~1GB | Minimal | Very Fast | 2-3GB | Experimental/testing only |

**Recommendation:** Stick with **Q4_K_M** for the best balance of quality, speed, and size.

---

## 🧪 Testing Your Fine-Tuned Model

### Test 1: Hugging Face Model (Full Precision)

This uses the full `final_model_merged/` directory and runs on Hugging Face Transformers.

```bash
cd /workspace/ea/finetuner

# Run automated test suite
python test_model.py --model_path outputs/final_model_merged --test_suite

# Test with your own prompt
python test_model.py --model_path outputs/final_model_merged \
    --prompt "Explain the concept of recursion in programming"

# Adjust creativity (temperature)
python test_model.py --model_path outputs/final_model_merged \
    --prompt "Write a creative story about AI" \
    --temperature 0.9 \
    --max_length 1024
```

**When to use:**
- ✅ Testing immediately after training
- ✅ Comparing before quantization
- ✅ Maximum quality inference (but slower)

### Test 2: GGUF Model (Quantized)

This is your production model - faster and more portable.

#### Method 1: Automated Test Script (Easiest)

```bash
cd /workspace/ea/finetuner

# Simple test
./test_gguf.sh

# Custom prompt
./test_gguf.sh outputs/model_q4_k_m.gguf "Explain quantum entanglement"
```

This script automatically installs llama.cpp if needed.

#### Method 2: llama.cpp (Manual)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with GPU support
make LLAMA_CUDA=1 -j$(nproc)

# Run inference
./main -m ../outputs/model_q4_k_m.gguf \
    -p "Explain the concept of recursion" \
    -n 512 \
    --temp 0.7 \
    --top-p 0.9 \
    --top-k 50 \
    --repeat-penalty 1.1 \
    -ngl 99  # Offload all layers to GPU

# Interactive chat mode
./main -m ../outputs/model_q4_k_m.gguf \
    --interactive \
    --reverse-prompt "User:" \
    -ngl 99
```

#### Method 3: Python (llama-cpp-python)

```bash
# Install
pip install llama-cpp-python

# Simple usage
python -c "
from llama_cpp import Llama

model = Llama(
    model_path='outputs/model_q4_k_m.gguf',
    n_gpu_layers=99,  # Offload to GPU
    n_ctx=8192  # Match your training context
)

output = model(
    'Explain quantum computing in simple terms',
    max_tokens=512,
    temperature=0.7,
    top_p=0.9
)

print(output['choices'][0]['text'])
"
```

#### Method 4: Ollama (Easiest for Local Use)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Create a Modelfile
cat > Modelfile << EOF
FROM ./outputs/model_q4_k_m.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
EOF

# Import your model
ollama create my-finetuned-model -f Modelfile

# Run it!
ollama run my-finetuned-model "Explain recursion"

# Interactive chat
ollama run my-finetuned-model
```

#### Method 5: LM Studio (GUI - Best for Non-Technical Users)

1. Download **LM Studio** from https://lmstudio.ai/
2. Open LM Studio
3. Go to "Local Models" → "Import"
4. Select your `model_q4_k_m.gguf` file
5. Click "Load Model"
6. Start chatting!

---

## 📊 Comparing Models

Test both the full and quantized versions to ensure quality:

```bash
cd /workspace/ea/finetuner

# Test full model
python test_model.py --model_path outputs/final_model_merged \
    --prompt "Explain the traveling salesman problem" \
    > full_output.txt

# Test GGUF model
./test_gguf.sh outputs/model_q4_k_m.gguf \
    "Explain the traveling salesman problem" \
    > gguf_output.txt

# Compare side-by-side
diff -y full_output.txt gguf_output.txt

# Or just view them
echo "=== FULL MODEL ===" && cat full_output.txt
echo "=== GGUF MODEL ===" && cat gguf_output.txt
```

**Expected:** GGUF output should be very similar to full model (Q4_K_M maintains ~95% quality).

---

## 🚀 Deployment Options

### Option 1: llama.cpp Server

```bash
cd llama.cpp

# Start server
./server -m ../outputs/model_q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99

# Test with curl
curl http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Explain quantum computing",
        "max_tokens": 512,
        "temperature": 0.7
    }'
```

### Option 2: Ollama (Recommended for Local)

```bash
# Already shown above - easiest for local deployment
ollama create my-model -f Modelfile
ollama run my-model
```

### Option 3: Text Generation WebUI

```bash
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui
bash start_linux.sh  # or start_windows.bat

# Copy your model to models/ folder
cp /path/to/model_q4_k_m.gguf models/

# Access at http://localhost:7860
```

### Option 4: OpenAI-Compatible API (llama-cpp-python)

```python
# server.py
from llama_cpp import Llama
from flask import Flask, request, jsonify

app = Flask(__name__)
model = Llama(model_path='outputs/model_q4_k_m.gguf', n_gpu_layers=99)

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 512)
    
    output = model(prompt, max_tokens=max_tokens)
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

```bash
# Run server
pip install flask llama-cpp-python
python server.py

# Test
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 100}'
```

---

## 🧹 Cleanup After Training

Once you've verified your model works:

```bash
cd /workspace/ea/finetuner/outputs

# Remove unnecessary files
rm -rf final_model/  # LoRA adapters (already merged)
rm -rf checkpoints/  # Training checkpoints

# Optional: Remove full model if you only need GGUF
rm -rf final_model_merged/

# What's left:
ls -lh
# Should show: model_q4_k_m.gguf (~2-3GB)
```

---

## 📝 Summary Checklist

- [ ] Training completed successfully
- [ ] `outputs/model_q4_k_m.gguf` exists (~2-3GB)
- [ ] Tested the model locally
- [ ] Downloaded `model_q4_k_m.gguf` to your local machine
- [ ] (Optional) Downloaded `final_model_merged/` for re-quantizing
- [ ] Deleted unnecessary files (`checkpoints/`, `final_model/`)
- [ ] Chose a deployment method (Ollama/llama.cpp/LM Studio)
- [ ] Model is running and responding correctly

---

## 🆘 Troubleshooting

### Model outputs gibberish

- **Cause:** Training didn't converge or data format was wrong
- **Fix:** Check training logs for high loss values, verify data format

### GGUF model not loading

- **Cause:** Corrupted file or incompatible llama.cpp version
- **Fix:** Re-quantize the model, update llama.cpp to latest

### Out of memory during inference

- **Cause:** Context length too high for your GPU
- **Fix:** Reduce `n_ctx` parameter or use a more aggressive quantization (Q4_0)

### Model responses are too repetitive

- **Fix:** Increase `repeat_penalty` (try 1.1-1.3) and adjust `temperature`

---

**Congratulations! 🎉 Your model is trained, quantized, and ready for production!**

