# BitsAndBytes Dequantization Fix

## Problem
The original dequantization script (`dequant_and_convert.py`) was not properly dequantizing bitsandbytes quantized models. Even though it set `load_in_4bit=False`, the model weights remained in their quantized form because bitsandbytes saves weights in a compressed format.

When llama.cpp's `convert_hf_to_gguf.py` tried to process these "dequantized" weights, it encountered the error:
```
NotImplementedError: Quant method is not yet supported: 'bitsandbytes'
```

## Known Issues
- **Shape Mismatch Error**: Models may contain a corrupted tensor with shape `[11206656, 1]` that should be `[2048, 10944]`. This causes:
  ```
  RuntimeError: Error(s) in loading state_dict for Linear:
          size mismatch for weight: copying a param with shape torch.Size([11206656, 1]) from checkpoint, the shape in current model is torch.Size([2048, 10944]).
  ```
  **Solution**: The robust scripts automatically detect and remove this corrupted tensor.

## Solution
The updated scripts handle various model corruption issues:

1. **`robust_dequantize.py`**: Robust dequantization that bypasses model loading issues
   - Loads state dict directly from safetensors files
   - Detects and fixes tensor shape mismatches
   - **Specifically handles the known [11206656, 1] corrupted tensor**
   - Removes quantization artifacts and corrupted tensors
   - Handles models with missing or corrupted weights

2. **`inspect_model.py`**: Diagnostic tool to inspect model structure
   - Shows tensor shapes and identifies suspicious ones (now checks ALL tensors)
   - Tests model loading to diagnose issues
   - Provides detailed information about model architecture

3. **`clean_quantization_artifacts.py`**: Thoroughly removes quantization metadata
   - Scans all safetensors files for quantization artifacts
   - Removes `.absmax`, `.quant_map`, `.quant_state` tensors
   - Cleans model configuration

4. **`remove_corrupted_tensor.py`**: Targeted fix for the specific corrupted tensor
   - Removes the [11206656, 1] tensor that's causing shape mismatch errors
   - Quick fix when you just need to remove this one problematic tensor

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
