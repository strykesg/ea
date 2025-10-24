#!/usr/bin/env python3
"""
Test script for the fine-tuned DeepSeek-V2-Lite model.
Tests both the full Hugging Face model and the quantized GGUF model.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTester:
    """Test the fine-tuned model with various inputs."""

    def __init__(self, model_path: str, use_gguf: bool = False):
        """
        Initialize the model tester.

        Args:
            model_path: Path to the model (either HF directory or GGUF file)
            use_gguf: Whether to use llama.cpp for GGUF inference
        """
        self.model_path = Path(model_path)
        self.use_gguf = use_gguf
        self.model = None
        self.tokenizer = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

    def load_model(self):
        """Load the model and tokenizer."""
        if self.use_gguf:
            logger.info("GGUF mode requires llama.cpp. Use the CLI command below:")
            logger.info(f"./llama.cpp/main -m {self.model_path} -p 'Your prompt here' -n 512")
            return False

        logger.info(f"Loading model from {self.model_path}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info("✅ Model loaded successfully!")
        return True

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold

        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Generating response for prompt: {prompt[:100]}...")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def run_test_suite(self):
        """Run a series of test prompts to evaluate the model."""
        test_prompts = [
            {
                "name": "Reasoning Test",
                "prompt": "Explain the concept of recursion in programming and provide a simple example."
            },
            {
                "name": "Code Generation Test",
                "prompt": "Write a Python function to calculate the Fibonacci sequence recursively."
            },
            {
                "name": "Long-form Test",
                "prompt": "Describe the process of training a neural network from scratch, including data preprocessing, model architecture, and optimization."
            }
        ]

        logger.info("Running test suite...")
        logger.info("=" * 80)

        for i, test in enumerate(test_prompts, 1):
            logger.info(f"\n🧪 Test {i}: {test['name']}")
            logger.info(f"Prompt: {test['prompt']}")
            logger.info("-" * 80)

            try:
                response = self.generate(test['prompt'], max_length=512)
                logger.info(f"Response:\n{response}")
                logger.info("=" * 80)
            except Exception as e:
                logger.error(f"❌ Test failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test the fine-tuned DeepSeek-V2-Lite model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test the Hugging Face model
  python test_model.py --model_path outputs/final_model_merged

  # Test with custom prompt
  python test_model.py --model_path outputs/final_model_merged --prompt "Explain quantum computing"

  # Run full test suite
  python test_model.py --model_path outputs/final_model_merged --test_suite

  # Info for GGUF testing (requires llama.cpp)
  python test_model.py --model_path outputs/model_q4_k_m.gguf --gguf
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/final_model_merged",
        help="Path to the model directory or GGUF file"
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="Model is a GGUF file (will show llama.cpp command)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt to test"
    )
    parser.add_argument(
        "--test_suite",
        action="store_true",
        help="Run full test suite"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of generated text (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = ModelTester(args.model_path, use_gguf=args.gguf)

    # Load model (unless GGUF)
    if not args.gguf:
        if not tester.load_model():
            return

        # Run test suite or custom prompt
        if args.test_suite:
            tester.run_test_suite()
        elif args.prompt:
            logger.info(f"Testing with custom prompt...")
            response = tester.generate(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            logger.info(f"\n📝 Response:\n{response}")
        else:
            logger.error("Please specify --prompt or --test_suite")
    else:
        logger.info("\n🔧 To test GGUF model with llama.cpp:")
        logger.info(f"   ./llama.cpp/main -m {args.model_path} -p 'Your prompt' -n {args.max_length}")
        logger.info("\n   Or install llama-cpp-python:")
        logger.info("   pip install llama-cpp-python")
        logger.info("   python -c \"from llama_cpp import Llama; model = Llama(model_path='%s'); print(model('Your prompt'))\"" % args.model_path)


if __name__ == "__main__":
    main()

