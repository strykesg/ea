"""
Fine-tuning script for DeepSeek-V2-Lite model.
Optimized for long-form, complex reasoning with high sequence lengths and sample packing.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not available, use system env vars

# Unsloth imports for efficient fine-tuning
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer

# Local imports
from data_loader import load_and_prepare_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning."""

    # Model configuration
    model_name: str = "deepseek-ai/DeepSeek-V2-Lite"
    max_seq_length: int = 16384  # High sequence length for long-form reasoning
    dtype: str = "bfloat16"  # Use bfloat16 for efficiency
    load_in_4bit: bool = True  # 4-bit quantization for memory efficiency

    # RoPE scaling configuration for long contexts
    rope_scaling: dict = None  # Will be set for DeepSeek models

    # LoRA configuration
    lora_r: int = 64  # LoRA rank
    lora_alpha: int = 128  # LoRA alpha
    lora_dropout: float = 0.1  # LoRA dropout
    lora_target_modules: list = None  # Will be set based on model

    # Training configuration
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60  # Small number for quick fine-tuning
    learning_rate: float = 8e-5
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407

    # Sample packing for long sequences
    sample_packing: bool = True

    # Output configuration
    output_dir: str = "outputs"
    save_strategy: str = "steps"
    save_steps: int = 20
    save_total_limit: int = 1

    def __post_init__(self):
        """Set default configurations for DeepSeek models."""
        if self.lora_target_modules is None:
            # Common target modules for DeepSeek V2-Lite models
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        # Set RoPE scaling for DeepSeek-V2-Lite long context support
        if self.rope_scaling is None and "deepseek" in self.model_name.lower():
            self.rope_scaling = {
                "type": "dynamic",
                "factor": float(40.0),
                "beta_fast": float(32.0),
                "beta_slow": float(1.0)
            }


class DeepSeekFinetuner:
    """Fine-tuner for DeepSeek-V2-Lite model."""

    def __init__(self, config: FinetuneConfig):
        """
        Initialize the fine-tuner.

        Args:
            config: Configuration for fine-tuning
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        """Load the model and tokenizer with Unsloth optimizations."""
        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Max sequence length: {self.config.max_seq_length}")
        logger.info(f"Sample packing: {self.config.sample_packing}")

        if self.config.rope_scaling is not None:
            logger.info(f"RoPE scaling: {self.config.rope_scaling}")

        # Load model with Unsloth for memory efficiency
        load_kwargs = {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "dtype": getattr(torch, self.config.dtype),
            "load_in_4bit": self.config.load_in_4bit,
            "token": os.getenv("HF_TOKEN"),  # HuggingFace token if needed
        }

        # Add RoPE scaling for DeepSeek models
        if self.config.rope_scaling is not None:
            logger.info(f"Using RoPE scaling configuration: {self.config.rope_scaling}")
            load_kwargs["rope_scaling"] = self.config.rope_scaling

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

        # Fix RoPE scaling configuration for DeepSeek models
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'rope_scaling'):
            if isinstance(self.model.config.rope_scaling, dict):
                # Ensure all rope_scaling values are floats
                for key, value in self.model.config.rope_scaling.items():
                    if isinstance(value, int):
                        self.model.config.rope_scaling[key] = float(value)
                        logger.info(f"Fixed rope_scaling {key}: {value} -> {float(value)}")

        # Apply LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.lora_target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
            use_rslora=False,
            loftq_config=None,
        )

        logger.info("Model and tokenizer loaded successfully")
        logger.info(f"Trainable parameters: {self.model.num_parameters()}")
        logger.info(f"Memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")

    def prepare_data(self):
        """Prepare the dataset for training."""
        logger.info("Preparing dataset...")

        # Load and prepare the dataset
        dataset = load_and_prepare_dataset("data.jsonl")

        # Apply chat template if needed
        tokenizer = self.tokenizer

        def formatting_prompts_func(examples):
            """Format prompts for training."""
            texts = []
            for prompt, completion in zip(examples['prompt'], examples['completion']):
                # Format for instruction tuning
                text = f"{prompt}\n\nResponse: {completion}"
                texts.append(text)
            return {"text": texts}

        # Format the dataset
        formatted_dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
            remove_columns=['prompt', 'completion', 'id', 'generated_at']
        )

        logger.info(f"Dataset size: {len(formatted_dataset)}")
        logger.info(f"Sample text: {formatted_dataset[0]['text'][:200]}...")

        return formatted_dataset

    def setup_trainer(self, train_dataset):
        """Set up the training configuration."""
        logger.info("Setting up trainer...")

        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=not self.config.load_in_4bit,  # Use fp16 if not using 4-bit
            bf16=self.config.load_in_4bit,  # Use bf16 with 4-bit quantization
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=False,
            dataloader_drop_last=True,
            dataloader_num_workers=2,
            run_name="deepseek-coder-finetune",
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        )

        # Create trainer with Unsloth optimizations
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=2,
            packing=self.config.sample_packing,  # Enable sample packing for efficiency
            args=training_args,
        )

        logger.info("Trainer setup complete")

    def train(self):
        """Run the training process."""
        logger.info("Starting training...")

        # Train the model
        self.trainer.train()

        logger.info("Training complete!")

    def save_model(self, output_path: Optional[str] = None):
        """Save the fine-tuned model."""
        if output_path is None:
            output_path = f"{self.config.output_dir}/final_model"

        logger.info(f"Saving model to {output_path}")

        # Save the model
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save merged model (LoRA + base model)
        merged_path = f"{output_path}_merged"
        logger.info(f"Saving merged model to {merged_path}")

        # Merge LoRA weights with base model
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        self.tokenizer.save_pretrained(merged_path)

        logger.info("Model saved successfully!")

        return output_path, merged_path


def main(config_dict: Optional[Dict[str, Any]] = None):
    """
    Main function to run fine-tuning.

    Args:
        config_dict: Optional configuration overrides
    """
    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Initialize configuration
    config = FinetuneConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    logger.info("Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")

    # Check for data file
    if not Path("data.jsonl").exists():
        raise FileNotFoundError(
            "data.jsonl not found. Please place your training data in data.jsonl "
            "in the same format as training_data.jsonl"
        )

    # Initialize fine-tuner
    finetuner = DeepSeekFinetuner(config)

    try:
        # Load model and tokenizer
        finetuner.load_model_and_tokenizer()

        # Prepare data
        train_dataset = finetuner.prepare_data()

        # Setup trainer
        finetuner.setup_trainer(train_dataset)

        # Train
        finetuner.train()

        # Save models
        output_path, merged_path = finetuner.save_model()

        logger.info("Fine-tuning complete!")
        logger.info(f"Model saved to: {output_path}")
        logger.info(f"Merged model saved to: {merged_path}")

        return output_path, merged_path

    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise


if __name__ == "__main__":
    # Run with default configuration
    main()

