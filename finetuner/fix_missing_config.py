#!/usr/bin/env python3
"""
Fix script to copy missing configuration files from the base model to the merged model.
This resolves the "configuration_deepseek.py not found" error during quantization.
"""

import os
import shutil
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_merged_model(
    merged_path: str = "outputs/final_model_merged",
    base_model: str = "deepseek-ai/DeepSeek-V2-Lite"
):
    """
    Copy missing configuration files from the base model to the merged model.
    
    Args:
        merged_path: Path to the merged model directory
        base_model: Base model identifier on Hugging Face
    """
    merged_path = Path(merged_path)
    
    if not merged_path.exists():
        logger.error(f"❌ Merged model not found at {merged_path}")
        return False
    
    logger.info(f"🔧 Fixing merged model at {merged_path}")
    
    # Download base model to local cache
    logger.info(f"📥 Downloading base model configuration from {base_model}")
    try:
        base_model_path = snapshot_download(
            base_model,
            allow_patterns=[
                "*.py",
                "*.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json"
            ],
            ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.gguf"],
        )
    except Exception as e:
        logger.error(f"❌ Failed to download base model: {e}")
        return False
    
    base_model_path = Path(base_model_path)
    logger.info(f"✅ Base model downloaded to {base_model_path}")
    
    # Files to copy
    files_to_copy = [
        "configuration_deepseek.py",
        "modeling_deepseek.py",
        "tokenization_deepseek.py",
        "tokenization_deepseek_fast.py",
    ]
    
    copied_files = []
    for filename in files_to_copy:
        src_file = base_model_path / filename
        dst_file = merged_path / filename
        
        if src_file.exists():
            if dst_file.exists():
                logger.info(f"⏭️  {filename} already exists, skipping")
            else:
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)
                logger.info(f"✅ Copied {filename}")
        else:
            logger.warning(f"⚠️  {filename} not found in base model")
    
    if copied_files:
        logger.info(f"✅ Successfully copied {len(copied_files)} files:")
        for f in copied_files:
            logger.info(f"   - {f}")
    else:
        logger.info("ℹ️  No files needed to be copied")
    
    # Verify critical files exist
    critical_files = ["config.json", "tokenizer_config.json"]
    missing = []
    for filename in critical_files:
        if not (merged_path / filename).exists():
            missing.append(filename)
    
    if missing:
        logger.error(f"❌ Critical files missing: {missing}")
        return False
    
    logger.info("✅ Merged model fixed successfully!")
    logger.info(f"   You can now quantize with:")
    logger.info(f"   python src/run_finetune.py --skip_finetune --model_path {merged_path}")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix missing configuration files in the merged model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix the default merged model
  python fix_missing_config.py

  # Fix a specific merged model
  python fix_missing_config.py --merged_path outputs/my_model_merged

  # Use a different base model
  python fix_missing_config.py --base_model deepseek-ai/DeepSeek-V2-Lite
        """
    )
    
    parser.add_argument(
        "--merged_path",
        type=str,
        default="outputs/final_model_merged",
        help="Path to the merged model directory (default: outputs/final_model_merged)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Base model identifier (default: deepseek-ai/DeepSeek-V2-Lite)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("DeepSeek Model Configuration Fix Tool")
    logger.info("=" * 60)
    
    success = fix_merged_model(args.merged_path, args.base_model)
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ Fix complete! You can now quantize your model.")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("❌ Fix failed! Check the logs above.")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

