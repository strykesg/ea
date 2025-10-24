# BitsAndBytes Dequantization Fix

## Problem
The original dequantization script (`dequant_and_convert.py`) was not properly dequantizing bitsandbytes quantized models. Even though it set `load_in_4bit=False`, the model weights remained in their quantized form because bitsandbytes saves weights in a compressed format.

When llama.cpp's `convert_hf_to_gguf.py` tried to process these "dequantized" weights, it encountered the error:
```
NotImplementedError: Quant method is not yet supported: 'bitsandbytes'
```

## Solution
The updated scripts handle various model corruption issues:

1. **`robust_dequantize.py`**: Robust dequantization that bypasses model loading issues
   - Loads state dict directly from safetensors files
   - Detects and fixes tensor shape mismatches
   - Removes quantization artifacts and corrupted tensors
   - Handles models with missing or corrupted weights

2. **`inspect_model.py`**: Diagnostic tool to inspect model structure
   - Shows tensor shapes and identifies suspicious ones
   - Tests model loading to diagnose issues
   - Provides detailed information about model architecture

3. **`clean_quantization_artifacts.py`**: Thoroughly removes quantization metadata
   - Scans all safetensors files for quantization artifacts
   - Removes `.absmax`, `.quant_map`, `.quant_state` tensors
   - Cleans model configuration

## Additional Tools
- **`clean_temp_model.sh`**: Manual cleanup script for the temp model directory
- **`check_quantization.py`**: Verify if a model still has quantization artifacts

## Usage
The fix is integrated into the `quick_fix_and_quantize.sh` script, so just run:

```bash
./quick_fix_and_quantize.sh
```

## Manual Usage
If you need to run dequantization separately:

```bash
python proper_dequantize.py
```

This will:
- Load the model from `outputs/final_model_merged`
- Dequantize it properly
- Save to `outputs/temp_clean_model`
- Then you can proceed with GGUF conversion

## Technical Details
The key difference is that bitsandbytes quantization is not just a loading option - it's a persistent compression of the weights. Simply loading with `load_in_4bit=False` doesn't decompress the saved weights. You need to explicitly call the dequantization methods on the quantized parameters.
