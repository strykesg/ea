#!/usr/bin/env python3
"""
Direct GGUF quantization using llama.cpp.
This script clones llama.cpp, converts the model, and quantizes it.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    logger.info(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
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


def install_llama_cpp():
    """Clone and build llama.cpp if not already present."""
    llama_cpp_dir = Path("llama.cpp")
    
    if llama_cpp_dir.exists():
        logger.info("✅ llama.cpp already exists")
        # Check if already built
        if (llama_cpp_dir / "build" / "bin" / "quantize").exists() or \
           (llama_cpp_dir / "build" / "quantize").exists():
            logger.info("✅ llama.cpp already built")
            return llama_cpp_dir
        else:
            logger.info("ℹ️  llama.cpp exists but not built, building now...")
    else:
        logger.info("📥 Cloning llama.cpp...")
        run_command([
            "git", "clone", 
            "https://github.com/ggerganov/llama.cpp.git"
        ])
    
    # Check for CMake
    try:
        result = run_command(["cmake", "--version"], check=False)
        if result.returncode != 0:
            raise RuntimeError("CMake not found")
    except:
        logger.error("❌ CMake is required but not found!")
        logger.error("   Install CMake:")
        logger.error("   Ubuntu/Debian: sudo apt-get install cmake")
        logger.error("   Or: pip install cmake")
        raise RuntimeError("CMake is required to build llama.cpp")
    
    logger.info("🔧 Building llama.cpp with CMake...")
    
    # Check if CUDA is available
    try:
        run_command(["nvcc", "--version"], check=False)
        has_cuda = True
        logger.info("✅ CUDA detected, building with GPU support")
    except:
        has_cuda = False
        logger.info("ℹ️  No CUDA detected, building CPU-only version")
    
    # Create build directory
    build_dir = llama_cpp_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Configure with CMake
    cmake_args = [
        "cmake", "..",
        "-DLLAMA_CUDA=ON" if has_cuda else "-DLLAMA_CUDA=OFF",
    ]
    logger.info("📦 Configuring with CMake...")
    run_command(cmake_args, cwd=build_dir)
    
    # Build
    logger.info("🔨 Compiling (this may take a few minutes)...")
    run_command(
        ["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count() or 4)],
        cwd=build_dir
    )
    
    logger.info("✅ llama.cpp built successfully")
    return llama_cpp_dir


def convert_to_fp16(model_path, llama_cpp_dir):
    """Convert HuggingFace model to FP16 GGUF."""
    model_path = Path(model_path)
    fp16_output = Path("outputs") / "model_fp16.gguf"
    
    logger.info(f"🔄 Converting {model_path} to FP16 GGUF...")
    
    # Use llama.cpp's convert-hf-to-gguf.py script
    convert_script = llama_cpp_dir / "convert-hf-to-gguf.py"
    
    if not convert_script.exists():
        # Try older naming convention
        convert_script = llama_cpp_dir / "convert.py"
    
    if not convert_script.exists():
        raise FileNotFoundError(f"Convert script not found in {llama_cpp_dir}")
    
    # Install required Python packages for conversion
    logger.info("📦 Installing conversion dependencies...")
    run_command([
        sys.executable, "-m", "pip", "install", "-q",
        "numpy", "sentencepiece", "protobuf"
    ], check=False)
    
    # Run conversion
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outtype", "f16",
        "--outfile", str(fp16_output)
    ]
    
    run_command(cmd)
    
    if not fp16_output.exists():
        raise FileNotFoundError(f"FP16 GGUF not created: {fp16_output}")
    
    logger.info(f"✅ FP16 GGUF created: {fp16_output}")
    logger.info(f"   Size: {fp16_output.stat().st_size / 1e9:.2f} GB")
    
    return fp16_output


def quantize_gguf(fp16_path, quantization_type, llama_cpp_dir):
    """Quantize FP16 GGUF to specified quantization level."""
    output_path = Path("outputs") / f"model_{quantization_type.lower()}.gguf"
    
    logger.info(f"🔄 Quantizing to {quantization_type}...")
    
    # Look for quantize binary in build directory (CMake) or root (old Makefile)
    quantize_binary = llama_cpp_dir / "build" / "bin" / "quantize"
    if not quantize_binary.exists():
        quantize_binary = llama_cpp_dir / "build" / "quantize"
    if not quantize_binary.exists():
        quantize_binary = llama_cpp_dir / "quantize"
    
    if not quantize_binary.exists():
        raise FileNotFoundError(f"Quantize binary not found. Tried:\n"
                              f"  - {llama_cpp_dir}/build/bin/quantize\n"
                              f"  - {llama_cpp_dir}/build/quantize\n"
                              f"  - {llama_cpp_dir}/quantize")
    
    logger.info(f"Using quantize binary: {quantize_binary}")
    
    cmd = [
        str(quantize_binary),
        str(fp16_path),
        str(output_path),
        quantization_type
    ]
    
    run_command(cmd)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Quantized GGUF not created: {output_path}")
    
    logger.info(f"✅ Quantized GGUF created: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1e9:.2f} GB")
    
    return output_path


def main():
    """Main quantization workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Direct GGUF quantization using llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize with default settings
  python quantize_direct.py

  # Quantize specific model
  python quantize_direct.py --model_path outputs/final_model_merged

  # Different quantization level
  python quantize_direct.py --quantization Q8_0
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/final_model_merged",
        help="Path to the HuggingFace model directory"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="Q4_K_M",
        choices=["Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"],
        help="Quantization type (default: Q4_K_M)"
    )
    parser.add_argument(
        "--keep_fp16",
        action="store_true",
        help="Keep the intermediate FP16 GGUF file"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 Direct GGUF Quantization")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info("")
    
    try:
        # Step 1: Install/verify llama.cpp
        llama_cpp_dir = install_llama_cpp()
        
        # Step 2: Convert to FP16 GGUF
        fp16_path = convert_to_fp16(args.model_path, llama_cpp_dir)
        
        # Step 3: Quantize
        quantized_path = quantize_gguf(fp16_path, args.quantization, llama_cpp_dir)
        
        # Step 4: Rename to standard name
        final_path = Path("outputs") / "model_q4_k_m.gguf"
        if quantized_path != final_path:
            logger.info(f"📝 Renaming {quantized_path.name} to {final_path.name}")
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(quantized_path), str(final_path))
            quantized_path = final_path
        
        # Step 5: Cleanup
        if not args.keep_fp16 and fp16_path.exists() and fp16_path != quantized_path:
            logger.info(f"🗑️  Removing intermediate FP16 file")
            fp16_path.unlink()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Quantization complete!")
        logger.info("=" * 60)
        logger.info(f"Output: {quantized_path}")
        logger.info(f"Size: {quantized_path.stat().st_size / 1e9:.2f} GB")
        logger.info("")
        logger.info("🧪 Test your model:")
        logger.info(f"   ./test_gguf.sh {quantized_path}")
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error(f"❌ Quantization failed: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

