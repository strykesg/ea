#!/usr/bin/env python3
"""
Manual GGUF conversion script using llama.cpp's built-in tools.
This script doesn't use the Python wrapper - uses llama.cpp binaries directly.
"""

import os
import subprocess
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, check=True):
    """Run a shell command."""
    logger.info(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        check=False
    )

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr and result.returncode != 0:
        logger.error(result.stderr)

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

    return result


def main():
    """Convert model using llama.cpp binaries directly."""
    model_path = "outputs/final_model_merged"
    llama_cpp_dir = "llama.cpp"

    logger.info("=" * 60)
    logger.info("🔄 Manual GGUF Conversion")
    logger.info("=" * 60)

    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        return 1

    # Check if llama.cpp exists
    if not Path(llama_cpp_dir).exists():
        logger.error(f"❌ llama.cpp not found: {llama_cpp_dir}")
        logger.error("   Build llama.cpp first:")
        logger.error("   git clone https://github.com/ggerganov/llama.cpp.git")
        logger.error("   cd llama.cpp && mkdir build && cd build")
        logger.error("   cmake .. -DLLAMA_CURL=OFF -DGGML_CUDA=ON")
        logger.error("   cmake --build . --config Release")
        return 1

    # Check for convert script
    convert_script = Path(llama_cpp_dir) / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        convert_script = Path(llama_cpp_dir) / "convert.py"

    if not convert_script.exists():
        logger.error(f"❌ Convert script not found: {convert_script}")
        logger.error("   Available files in llama.cpp:")
        for f in Path(llama_cpp_dir).glob("*"):
            if f.is_file() and f.suffix == ".py":
                logger.error(f"     - {f}")
        return 1

    logger.info(f"✅ Found convert script: {convert_script}")

    # Install dependencies
    logger.info("📦 Installing dependencies...")
    try:
        run_command(["pip", "install", "-q", "numpy", "sentencepiece", "protobuf"])
    except:
        logger.warning("Could not install dependencies automatically")

    # Convert to FP16
    fp16_output = "outputs/model_fp16.gguf"
    logger.info(f"🔄 Converting to FP16 GGUF: {fp16_output}")

    cmd = [
        "python",
        str(convert_script),
        model_path,
        "--outtype", "f16",
        "--outfile", fp16_output
    ]

    try:
        run_command(cmd)
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        logger.error("   Try running manually:")
        logger.error(f"   python {convert_script} {model_path} --outtype f16 --outfile {fp16_output}")
        return 1

    # Check if FP16 was created
    if not Path(fp16_output).exists():
        logger.error(f"❌ FP16 GGUF not created: {fp16_output}")
        return 1

    logger.info(f"✅ FP16 GGUF created: {fp16_output}")
    logger.info(f"   Size: {Path(fp16_output).stat().st_size / 1e9:.2f} GB")

    # Quantize to Q4_K_M
    quantized_output = "outputs/model_q4_k_m.gguf"
    quantize_binary = Path(llama_cpp_dir) / "build" / "bin" / "quantize"

    if not quantize_binary.exists():
        quantize_binary = Path(llama_cpp_dir) / "build" / "quantize"

    if not quantize_binary.exists():
        logger.error(f"❌ Quantize binary not found: {quantize_binary}")
        logger.error("   Build llama.cpp first with:")
        logger.error("   cmake --build build --config Release")
        return 1

    logger.info(f"🔄 Quantizing to Q4_K_M: {quantized_output}")
    logger.info(f"   Using: {quantize_binary}")

    cmd = [
        str(quantize_binary),
        fp16_output,
        quantized_output,
        "Q4_K_M"
    ]

    try:
        run_command(cmd)
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        logger.error("   Try running manually:")
        logger.error(f"   {quantize_binary} {fp16_output} {quantized_output} Q4_K_M")
        return 1

    # Check if quantized file was created
    if not Path(quantized_output).exists():
        logger.error(f"❌ Quantized GGUF not created: {quantized_output}")
        return 1

    logger.info(f"✅ Quantized GGUF created: {quantized_output}")
    logger.info(f"   Size: {Path(quantized_output).stat().st_size / 1e9:.2f} GB")

    # Clean up FP16 if desired
    cleanup = input("🗑️  Remove intermediate FP16 file? (y/N): ").lower().strip()
    if cleanup == 'y':
        Path(fp16_output).unlink()
        logger.info(f"🗑️  Removed {fp16_output}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Conversion complete!")
    logger.info("=" * 60)
    logger.info(f"Final model: {quantized_output}")
    logger.info(f"Size: {Path(quantized_output).stat().st_size / 1e9:.2f} GB")
    logger.info("")
    logger.info("🧪 Test your model:")
    logger.info(f"   ./test_gguf.sh {quantized_output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

