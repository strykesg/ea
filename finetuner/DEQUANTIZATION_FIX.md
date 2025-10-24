# BitsAndBytes Dequantization Fix

## Problem
The original dequantization script (`dequant_and_convert.py`) was not properly dequantizing bitsandbytes quantized models. Even though it set `load_in_4bit=False`, the model weights remained in their quantized form because bitsandbytes saves weights in a compressed format.

When llama.cpp's `convert_hf_to_gguf.py` tried to process these "dequantized" weights, it encountered the error:
```
NotImplementedError: Quant method is not yet supported: 'bitsandbytes'
```

## Solution
The new `proper_dequantize.py` script properly handles bitsandbytes quantization by:

1. **Detecting quantization**: Checks if the model has a `quantization_config` in its config.json
2. **Loading with quantization**: If quantized, loads the model with bitsandbytes quantization enabled
3. **Actual dequantization**: Calls `.dequantize()` on each quantized parameter to convert back to full precision float16
4. **Cleaning up**: Removes quantization hooks and config entries
5. **Saving clean model**: Saves the fully dequantized model

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
