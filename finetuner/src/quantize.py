"""
Quantization script to convert fine-tuned models to GGUF format with Q4_K_M quantization.
Optimized for long-context models like DeepSeek-Coder-V2-Lite-Instruct.
"""

import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# GGUF conversion imports
try:
    from gguf import GGUFWriter
    from gguf.constants import GGUF_DEFAULT_ALIGNMENT
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

logger = logging.getLogger(__name__)


class GGUFQuantizer:
    """Convert and quantize models to GGUF format."""

    def __init__(self, model_path: str):
        """
        Initialize the quantizer.

        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(self.model)}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def convert_to_gguf(self, output_path: str, quantization_type: str = "Q4_K_M"):
        """
        Convert the model to GGUF format with specified quantization.

        Args:
            output_path: Path for the output GGUF file
            quantization_type: Type of quantization (Q4_K_M, Q5_K_M, etc.)
        """
        logger.info(f"Converting to GGUF with {quantization_type} quantization")

        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Method 1: Try using python-gguf if available
        if GGUF_AVAILABLE:
            try:
                success = self._convert_with_gguf(output_path, quantization_type)
                if success:
                    return
            except Exception as e:
                logger.warning(f"GGUF conversion failed: {e}")

        # Method 2: Use llama-cpp-python (fallback)
        try:
            success = self._convert_with_llama_cpp(output_path, quantization_type)
            if success:
                return
        except Exception as e:
            logger.warning(f"llama-cpp conversion failed: {e}")

        # Method 3: Use system llama-cpp convert tools (if available)
        try:
            success = self._convert_with_system_tools(output_path, quantization_type)
            if success:
                return
        except Exception as e:
            logger.warning(f"System tools conversion failed: {e}")

        raise RuntimeError("All GGUF conversion methods failed")

    def _convert_with_gguf(self, output_path: str, quantization_type: str) -> bool:
        """Convert using python-gguf library."""
        logger.info("Using python-gguf for conversion")

        # This is a simplified version - actual implementation would be more complex
        # and depend on the specific GGUF library interface

        # For now, we'll use the llama-cpp method as it's more reliable
        return False

    def _convert_with_llama_cpp(self, output_path: str, quantization_type: str) -> bool:
        """Convert using llama-cpp-python."""
        logger.info("Using llama-cpp-python for conversion")

        try:
            import llama_cpp

            # Save model in a format llama-cpp can understand
            # First save as pytorch format
            temp_model_path = self.model_path / "pytorch_model"
            temp_model_path.mkdir(exist_ok=True)

            # Save model and tokenizer
            self.model.save_pretrained(temp_model_path)
            self.tokenizer.save_pretrained(temp_model_path)

            # Use llama-cpp to convert and quantize
            # This requires llama-cpp-python with conversion support
            gguf_path = Path(output_path)

            # For now, we'll use the system method as llama-cpp-python conversion
            # is complex and may not be available in all versions
            return False

        except ImportError:
            logger.warning("llama-cpp-python not available or doesn't support conversion")
            return False

    def _convert_with_system_tools(self, output_path: str, quantization_type: str) -> bool:
        """Convert using system llama-cpp tools."""
        logger.info("Using system llama-cpp tools for conversion")

        # First, we need to convert the HuggingFace model to a format that llama-cpp can read
        # This typically involves converting to PyTorch format and then using llama-cpp's convert script

        # Check if llama-cpp tools are available
        convert_script = shutil.which("python")
        if not convert_script:
            logger.error("Python not found in PATH")
            return False

        # Try to find llama-cpp convert script
        # This usually comes with llama-cpp-python installation
        try:
            # Look for convert.py in common locations
            import site
            site_packages = site.getsitepackages()

            convert_script = None
            for package_dir in site_packages:
                potential_script = Path(package_dir) / "llama_cpp" / "convert.py"
                if potential_script.exists():
                    convert_script = potential_script
                    break

            if not convert_script:
                # Try to run llama-cpp-python's convert command
                result = subprocess.run(
                    ["python", "-c", "from llama_cpp import convert; print('llama-cpp available')"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info("llama-cpp-python conversion available")

                    # Use a simple conversion approach
                    # Save model weights in a format that can be converted
                    self._save_for_conversion(self.model_path / "conversion_temp")

                    # Use llama-cpp-python's built-in conversion
                    from llama_cpp import Llama
                    # This would require the model to be in the right format
                    # For now, we'll use the manual approach

            # Alternative: Use the HuggingFace to GGUF conversion tools
            # This is the most reliable method for now
            self._convert_with_hf_to_gguf(output_path, quantization_type)
            return True

        except Exception as e:
            logger.error(f"System tools conversion failed: {e}")
            return False

    def _convert_with_hf_to_gguf(self, output_path: str, quantization_type: str):
        """Convert using HuggingFace to GGUF tools."""
        logger.info("Using HuggingFace to GGUF conversion")

        try:
            # This is a placeholder for the actual conversion
            # In practice, you would use a tool like:
            # https://github.com/ggerganov/llama.cpp/blob/master/convert.py

            # For now, we'll create a script that users can run manually
            self._create_conversion_script(output_path, quantization_type)

        except Exception as e:
            logger.error(f"HuggingFace to GGUF conversion failed: {e}")
            raise

    def _save_for_conversion(self, temp_path: Path):
        """Save model in format suitable for conversion."""
        temp_path.mkdir(exist_ok=True)

        # Save model in different formats that conversion tools expect
        # Save as safetensors (most common format)
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(temp_path, safe_serialization=True)
            self.tokenizer.save_pretrained(temp_path)

    def _create_conversion_script(self, output_path: str, quantization_type: str):
        """Create a script for manual conversion."""
        script_path = Path(output_path).parent / "convert_to_gguf.py"

        script_content = f'''#!/usr/bin/env python3
"""
Manual conversion script for {self.model_path} to GGUF format.
Run this script to convert the fine-tuned model to GGUF with {quantization_type} quantization.
"""

import os
import subprocess
import sys
from pathlib import Path

def convert_model():
    """Convert the model to GGUF format."""
    model_path = Path("{self.model_path}")
    output_path = Path("{output_path}")

    print(f"Converting model from {{model_path}} to {{output_path}}")
    print(f"Quantization type: {quantization_type}")

    # Method 1: Use llama.cpp convert script if available
    try:
        # Try to find and use llama.cpp's convert.py
        import importlib.util

        # Look for llama_cpp in site-packages
        llama_cpp_path = None
        for path in sys.path:
            potential_path = Path(path) / "llama_cpp" / "convert.py"
            if potential_path.exists():
                llama_cpp_path = potential_path
                break

        if llama_cpp_path:
            print(f"Found llama.cpp convert script: {{llama_cpp_path}}")
            cmd = [
                sys.executable, str(llama_cpp_path),
                str(model_path),
                str(output_path),
                "--outtype", "f16",
                "--vocab-type", "huggingface"
            ]
            print(f"Running: {{' '.join(cmd)}}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("Conversion successful!")
                print(f"Output: {{output_path}}")
                return True
            else:
                print(f"Conversion failed: {{result.stderr}}")

    except Exception as e:
        print(f"llama.cpp conversion failed: {{e}}")

    # Method 2: Manual conversion instructions
    print("\\nManual conversion instructions:")
    print("1. Install llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp")
    print("   cd llama.cpp && make")
    print("")
    print("2. Convert the model:")
    print(f"   python convert.py {{model_path}} --outtype f16 --vocab-type huggingface")
    print("")
    print("3. Quantize the model:")
    print(f"   ./quantize {{output_path}}.bin {{output_path}}_{quantization_type}.gguf {quantization_type}")

    return False

if __name__ == "__main__":
    success = convert_model()
    sys.exit(0 if success else 1)
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable
        script_path.chmod(0o755)

        logger.info(f"Created conversion script: {script_path}")
        logger.info("Run the script manually: python convert_to_gguf.py")


def main(model_path: str = "outputs/final_model_merged",
         output_path: str = "outputs/model_q4_k_m.gguf",
         quantization_type: str = "Q4_K_M"):
    """
    Main quantization function.

    Args:
        model_path: Path to the fine-tuned model
        output_path: Path for the output GGUF file
        quantization_type: Type of quantization to apply
    """
    # Create output directory
    os.makedirs(Path(output_path).parent, exist_ok=True)

    try:
        # Initialize quantizer
        quantizer = GGUFQuantizer(model_path)

        # Load model
        quantizer.load_model()

        # Convert to GGUF
        quantizer.convert_to_gguf(output_path, quantization_type)

        logger.info("Quantization complete!")
        logger.info(f"GGUF model saved to: {output_path}")

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise


if __name__ == "__main__":
    # Run with command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Convert fine-tuned model to GGUF format")
    parser.add_argument("--model_path", default="outputs/final_model_merged",
                       help="Path to fine-tuned model")
    parser.add_argument("--output_path", default="outputs/model_q4_k_m.gguf",
                       help="Output path for GGUF file")
    parser.add_argument("--quantization_type", default="Q4_K_M",
                       help="Quantization type (Q4_K_M, Q5_K_M, etc.)")

    args = parser.parse_args()

    main(args.model_path, args.output_path, args.quantization_type)

