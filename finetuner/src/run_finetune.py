#!/usr/bin/env python3
"""
One-click script to fine-tune DeepSeek-Coder-V2-Lite-Instruct and quantize to GGUF.
This script handles the entire pipeline from data loading to final quantized model.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not available, use system env vars

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from finetune import main as finetune_main, FinetuneConfig
from quantize import main as quantize_main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('finetune.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class FinetunePipeline:
    """Complete fine-tuning pipeline from data to quantized model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.finetune_config = None
        self.output_paths = {}

    def check_requirements(self) -> bool:
        """Check if all requirements are met."""
        logger.info("Checking requirements...")

        # Check for data file
        if not Path("data.jsonl").exists():
            logger.error("❌ data.jsonl not found!")
            logger.error("Please place your training data in data.jsonl")
            logger.error("Format should match training_data.jsonl with 'instruction', 'input', 'output' fields")
            return False

        # Check data file size and format
        try:
            with open("data.jsonl", 'r') as f:
                lines = f.readlines()
                if not lines:
                    logger.error("❌ data.jsonl is empty!")
                    return False

                # Check format of first line
                first_line = lines[0].strip()
                if first_line:
                    sample_data = json.loads(first_line)
                    required_fields = {'instruction', 'input', 'output'}
                    if not all(field in sample_data for field in required_fields):
                        logger.error(f"❌ Invalid format in data.jsonl. Missing required fields: {required_fields - set(sample_data.keys())}")
                        return False

            logger.info(f"✅ Found data.jsonl with {len(lines)} examples")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"❌ Invalid JSON format in data.jsonl: {e}")
            return False

        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                logger.warning("⚠️  CUDA not available. Training will be slow on CPU.")
                logger.warning("   Consider using a GPU for better performance.")
        except ImportError:
            logger.warning("⚠️  PyTorch not available")

        # Check for HuggingFace token
        if not os.getenv("HF_TOKEN"):
            logger.warning("⚠️  HF_TOKEN not set. You may need it for some models.")
            logger.warning("   Set it with: export HF_TOKEN=your_token_here")

        # Check for wandb token (optional)
        if not os.getenv("WANDB_API_KEY"):
            logger.info("ℹ️  WANDB_API_KEY not set. Training logs will be saved locally only.")

        return True

    def setup_environment(self):
        """Set up the environment and dependencies."""
        logger.info("Setting up environment...")

        # Change to finetuner directory
        os.chdir(Path(__file__).parent.parent)

        # Install dependencies if needed
        if not self._check_dependencies():
            logger.info("Installing dependencies...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                             check=True, capture_output=True)
                logger.info("✅ Dependencies installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install dependencies: {e}")
                return False

        return True

    def _check_dependencies(self) -> bool:
        """Check if required packages are installed."""
        required_packages = [
            'torch', 'transformers', 'datasets', 'accelerate', 'peft',
            'unsloth', 'trl', 'huggingface_hub'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.warning(f"⚠️  Missing packages: {missing_packages}")
            return False

        logger.info("✅ All required packages are available")
        return True

    def run_finetuning(self) -> bool:
        """Run the fine-tuning process."""
        logger.info("🚀 Starting fine-tuning...")

        try:
            # Update configuration with user preferences
            config_dict = {
                'max_seq_length': 16384,  # High sequence length for long-form reasoning
                'sample_packing': True,   # Enable sample packing
                'max_steps': 60,          # Reasonable number of steps
                'per_device_train_batch_size': 2,
                'gradient_accumulation_steps': 4,
            }
            config_dict.update(self.config)

            # Run fine-tuning
            output_path, merged_path = finetune_main(config_dict)

            self.output_paths['finetuned'] = output_path
            self.output_paths['merged'] = merged_path

            logger.info("✅ Fine-tuning complete!")
            logger.info(f"   Fine-tuned model: {output_path}")
            logger.info(f"   Merged model: {merged_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            return False

    def run_quantization(self) -> bool:
        """Run the quantization process."""
        logger.info("🔄 Starting quantization...")

        try:
            # Use the merged model for quantization
            model_path = self.output_paths.get('merged', 'outputs/final_model_merged')
            output_path = "outputs/model_q4_k_m.gguf"

            # Run quantization
            quantize_main(model_path, output_path, "Q4_K_M")

            self.output_paths['gguf'] = output_path

            logger.info("✅ Quantization complete!")
            logger.info(f"   GGUF model: {output_path}")
            logger.info(f"   Quantization: Q4_K_M")
            logger.info(f"   File size: {Path(output_path).stat().st_size / 1e9:.2f} GB")

            return True

        except Exception as e:
            logger.error(f"❌ Quantization failed: {e}")
            return False

    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        logger.info("🎯 Starting complete fine-tuning pipeline...")
        logger.info("=" * 60)

        # Step 1: Check requirements
        if not self.check_requirements():
            return False

        # Step 2: Setup environment
        if not self.setup_environment():
            return False

        # Step 3: Fine-tuning
        if not self.run_finetuning():
            return False

        # Step 4: Quantization
        if not self.run_quantization():
            return False

        # Success!
        self.print_summary()
        return True

    def print_summary(self):
        """Print a summary of the completed pipeline."""
        logger.info("=" * 60)
        logger.info("🎉 PIPELINE COMPLETE!")
        logger.info("=" * 60)

        for step, path in self.output_paths.items():
            if Path(path).exists():
                size = Path(path).stat().st_size / 1e9 if Path(path).is_file() else "N/A"
                logger.info(f"✅ {step.upper()}: {path}")
                if size != "N/A":
                    logger.info(f"   Size: {size:.2f} GB")

        logger.info("")
        logger.info("📝 Next steps:")
        logger.info("   1. The quantized model is ready to use with llama.cpp")
        logger.info("   2. Test the model with your data")
        logger.info("   3. Adjust training parameters if needed")
        logger.info("")
        logger.info("🔧 Model configuration:")
        logger.info("   - Base model: DeepSeek-Coder-V2-Lite-Instruct")
        logger.info("   - Sequence length: 16384 tokens")
        logger.info("   - Sample packing: Enabled")
        logger.info("   - Quantization: Q4_K_M")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSeek-Coder-V2-Lite-Instruct and quantize to GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_finetune.py

  # Run with custom batch size
  python run_finetune.py --batch_size 1 --max_steps 100

  # Run only quantization on existing model
  python run_finetune.py --skip_finetune --model_path outputs/final_model_merged
        """
    )

    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size (default: 2)")
    parser.add_argument("--max_steps", type=int, default=60,
                       help="Maximum training steps (default: 60)")
    parser.add_argument("--seq_length", type=int, default=16384,
                       help="Maximum sequence length (default: 16384)")
    parser.add_argument("--skip_finetune", action="store_true",
                       help="Skip fine-tuning, only run quantization")
    parser.add_argument("--model_path", type=str,
                       help="Path to model for quantization (if skipping fine-tuning)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory (default: outputs)")
    parser.add_argument("--data_file", type=str, default="data.jsonl",
                       help="Path to training data file (default: data.jsonl)")

    args = parser.parse_args()

    # Create pipeline configuration
    config = {
        'per_device_train_batch_size': args.batch_size,
        'max_steps': args.max_steps,
        'max_seq_length': args.seq_length,
        'output_dir': args.output_dir,
    }

    # Initialize and run pipeline
    pipeline = FinetunePipeline(config)

    # Override data file if specified
    if args.data_file != "data.jsonl":
        os.rename(args.data_file, "data.jsonl")

    try:
        success = pipeline.run_pipeline()

        if success:
            logger.info("✅ Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Pipeline crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

