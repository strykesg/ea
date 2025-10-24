# 🔧 Recovery Guide: Fix "configuration_deepseek.py not found" Error

## 🚨 Problem

You see this error during quantization:

```
ERROR:quantize: Quantization failed: outputs/final_model_merged does not appear to have a file named configuration_deepseek.py.
```

## ✅ Quick Fix (Current Training)

Your training **completed successfully**! Only the quantization step failed. Your fine-tuned model is safe in `outputs/final_model_merged/`.

### **Solution 1: Use the Fix Script (Easiest)**

```bash
cd /workspace/ea/finetuner

# Run the fix script
python fix_missing_config.py

# Then quantize
python src/run_finetune.py --skip_finetune --model_path outputs/final_model_merged
```

This will:
1. Download the missing configuration files from the base model
2. Copy them to your merged model
3. Allow quantization to proceed

### **Solution 2: Manual Fix**

```bash
cd /workspace/ea/finetuner

# Download base model config files
python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

# Download base model
base_path = snapshot_download(
    'deepseek-ai/DeepSeek-V2-Lite',
    allow_patterns=['*.py']
)

# Copy configuration files
for file in ['configuration_deepseek.py', 'modeling_deepseek.py', 
             'tokenization_deepseek.py', 'tokenization_deepseek_fast.py']:
    src = Path(base_path) / file
    dst = Path('outputs/final_model_merged') / file
    if src.exists():
        shutil.copy2(src, dst)
        print(f'Copied {file}')
"

# Now quantize
python src/run_finetune.py --skip_finetune --model_path outputs/final_model_merged
```

---

## 🔍 What Happened?

DeepSeek models use custom Python files for their model architecture:
- `configuration_deepseek.py` - Model configuration class
- `modeling_deepseek.py` - Model implementation
- `tokenization_deepseek.py` - Tokenizer implementation

When we saved the merged model, these files weren't automatically copied. The quantization script needs them to load the model properly.

---

## ✅ Verify the Fix

After running the fix script:

```bash
# Check that the files exist
ls -la outputs/final_model_merged/*.py

# Should see:
# configuration_deepseek.py
# modeling_deepseek.py
# tokenization_deepseek.py
# tokenization_deepseek_fast.py

# Now quantize
python src/run_finetune.py --skip_finetune --model_path outputs/final_model_merged
```

---

## 🚀 Complete Workflow (From Your Current State)

```bash
cd /workspace/ea/finetuner

# Step 1: Fix the missing config files
python fix_missing_config.py

# Step 2: Quantize the model
python src/run_finetune.py --skip_finetune --model_path outputs/final_model_merged

# Step 3: Verify the GGUF was created
ls -lh outputs/model_q4_k_m.gguf

# Step 4: Test it!
./test_gguf.sh
```

---

## 📊 What You Have Now

Your training **succeeded**! Here's what's in your `outputs/` directory:

```
outputs/
├── final_model/              # LoRA adapters (200MB)
├── final_model_merged/       # Full fine-tuned model (10-15GB) ✅
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── ... (missing *.py files - need to fix)
└── checkpoints/              # Training checkpoints
```

After the fix:
```
outputs/
├── final_model_merged/       # Full fine-tuned model (10-15GB) ✅
│   ├── config.json
│   ├── configuration_deepseek.py      # ← Added by fix
│   ├── modeling_deepseek.py           # ← Added by fix
│   ├── tokenization_deepseek.py       # ← Added by fix
│   ├── model.safetensors
│   └── tokenizer_config.json
└── model_q4_k_m.gguf         # Quantized model (2-3GB) ✅ (after quantization)
```

---

## 🔄 Alternative: Skip Quantization (Use Full Model)

If you don't need the quantized GGUF, you can use the full model directly:

```bash
# Test the full Hugging Face model (no quantization needed)
python test_model.py --model_path outputs/final_model_merged --test_suite

# Or with a custom prompt
python test_model.py --model_path outputs/final_model_merged \
    --prompt "Explain quantum computing"
```

The full model will work fine - you only need the fix if you want the quantized GGUF version.

---

## 🛡️ Prevention (For Future Training)

This is now **automatically fixed** for future training runs! I've updated `src/finetune.py` to automatically copy these configuration files when saving the merged model.

For your next training run, this error won't happen.

---

## 🆘 Troubleshooting

### "File not found" after fix

```bash
# Check if the fix script worked
ls -la outputs/final_model_merged/*.py

# If files are missing, manually download them
python -c "
from huggingface_hub import hf_hub_download
import shutil

files = ['configuration_deepseek.py', 'modeling_deepseek.py']
for f in files:
    src = hf_hub_download('deepseek-ai/DeepSeek-V2-Lite', f)
    shutil.copy2(src, f'outputs/final_model_merged/{f}')
    print(f'Copied {f}')
"
```

### "Connection error" during fix

```bash
# If you have network issues, try with different HF mirror
export HF_ENDPOINT=https://hf-mirror.com

python fix_missing_config.py
```

### "Permission denied"

```bash
# Make sure you have write permissions
chmod +w outputs/final_model_merged/
python fix_missing_config.py
```

---

## 📝 Summary

| Status | What | Action |
|--------|------|--------|
| ✅ **Success** | Training completed | No action needed |
| ✅ **Success** | Model saved | `outputs/final_model_merged/` |
| ❌ **Failed** | Quantization | Run fix script then quantize |

**Your model is safe and working!** Just need to copy 4 Python files and re-run quantization.

---

## 🎯 Quick Commands Reference

```bash
# Fix the issue
python fix_missing_config.py

# Quantize
python src/run_finetune.py --skip_finetune --model_path outputs/final_model_merged

# Test
./test_gguf.sh

# Done! 🎉
```

---

**Need help?** Check the full guides:
- [POST_TRAINING_GUIDE.md](POST_TRAINING_GUIDE.md) - Complete post-training workflow
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference card
- [README.md](README.md) - Full documentation

