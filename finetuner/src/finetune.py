"""
Fine-tuning script for DeepSeek-V2-Lite model.
Optimized for long-form, complex reasoning with high sequence lengths and sample packing.
"""

import os
import logging
import torch
import time
import threading
import shutil
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from huggingface_hub import snapshot_download

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


def _enable_perf_tweaks():
    """Enable GPU performance features on supported hardware."""
    try:
        # Allow TF32 on matmul and cuDNN for faster GEMMs on Ampere/Blackwell
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        # Prefer higher precision matmul that maps to TF32 fast paths
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        # Prefer Flash and mem-efficient SDPA kernels when available
        from torch.backends.cuda import sdp_kernel

        sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass


def _sanitize_rope_scaling(rope: Optional[dict]) -> dict:
    """Ensure rope_scaling dict has correct types/values.

    DeepSeek expects dynamic RoPE with float values. If anything is
    missing or has integer types, coerce to sane float defaults.
    """
    default = {
        "type": "dynamic",
        "factor": 40.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    if not isinstance(rope, dict):
        return default

    sanitized = {
        "type": str(rope.get("type", default["type"])),
        "factor": float(rope.get("factor", default["factor"])),
        "beta_fast": float(rope.get("beta_fast", default["beta_fast"])),
        "beta_slow": float(rope.get("beta_slow", default["beta_slow"])),
    }
    if sanitized["factor"] < 1.0:
        sanitized["factor"] = 1.0
    return sanitized


def _preflight_and_set_hf_endpoint(repo_id: str) -> str:
    """Probe endpoints and set HF_ENDPOINT/HF_HUB_ENABLE_HF_TRANSFER accordingly.

    Returns the chosen endpoint base URL.
    """
    candidates = []
    # Respect user-provided HF_ENDPOINT first if set
    if os.getenv("HF_ENDPOINT"):
        candidates.append(os.getenv("HF_ENDPOINT"))
    # Default Hugging Face endpoint
    candidates.append("https://huggingface.co")
    # Public mirror
    candidates.append("https://hf-mirror.com")

    auth_header = None
    if os.getenv("HF_TOKEN"):
        auth_header = f"Bearer {os.getenv('HF_TOKEN')}"

    for endpoint in candidates:
        base = endpoint.rstrip("/")
        url = f"{base}/{repo_id}/resolve/main/config.json"
        try:
            req = urllib.request.Request(url, method="HEAD")
            if auth_header:
                req.add_header("Authorization", auth_header)
            with urllib.request.urlopen(req, timeout=6) as resp:
                code = getattr(resp, 'status', 200)
            logger.info(f"Preflight {url} → {code}")
            if 200 <= code < 400:
                os.environ["HF_ENDPOINT"] = base
                # Disable transfer accel on mirror to avoid provider issues
                if "hf-mirror" in base:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                logger.info(
                    f"Using HF endpoint: {base}; HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER','unset')}"
                )
                return base
        except Exception as e:
            logger.info(f"Preflight failed for {url}: {e}")

    # If all preflights failed, prefer mirror and disable transfer as last resort
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    logger.warning(
        f"All endpoint preflights failed; falling back to {os.environ['HF_ENDPOINT']} with HF_HUB_ENABLE_HF_TRANSFER=0"
    )
    return os.environ["HF_ENDPOINT"]


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning."""

    # Model configuration
    model_name: str = "deepseek-ai/DeepSeek-V2-Lite"
    max_seq_length: int = 16384  # High sequence length for long-form reasoning
    dtype: str = "bfloat16"  # Use bfloat16 for efficiency
    load_in_4bit: bool = True  # 4-bit quantization for memory efficiency

    # Note: RoPE scaling is handled automatically in the loading process

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

        # Note: RoPE scaling is now handled in the model loading process


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

        # Enable GPU performance features early
        _enable_perf_tweaks()

        # RoPE scaling is handled automatically for DeepSeek models

        # Proactively download the model snapshot with progress/heartbeat logs
        repo_id = self.config.model_name
        # Choose a working endpoint first
        _preflight_and_set_hf_endpoint(repo_id)
        local_dir = os.path.join("models", repo_id.replace("/", "__"))
        os.makedirs(local_dir, exist_ok=True)

        # Respect existing HF_HUB_ENABLE_HF_TRANSFER; do not force-enable here

        du = shutil.disk_usage(local_dir)
        logger.info(
            f"Preparing model download → target: {local_dir} | free: {du.free / 1e9:.2f} GB"
        )

        def _heartbeat(msg: str, stop_evt: threading.Event, interval: int = 30):
            while not stop_evt.wait(interval):
                logger.info(msg)

        # Heartbeat while snapshot_download runs
        hb_stop = threading.Event()
        hb_thread = threading.Thread(
            target=_heartbeat,
            args=("Downloading model snapshot… still working (network/IO)" , hb_stop, 30),
            daemon=True,
        )
        logger.info("Starting model snapshot download (this may take a while on first run)…")
        start_dl = time.monotonic()
        hb_thread.start()
        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=os.getenv("HF_TOKEN"),
                max_workers=8,
                allow_patterns=None,
                resume_download=True,
                local_files_only=False,
            )
        finally:
            hb_stop.set()
            hb_thread.join(timeout=5)

        # Summarize snapshot contents
        total_files = 0
        total_bytes = 0
        for root, _dirs, files in os.walk(snapshot_path):
            for fname in files:
                total_files += 1
                fpath = os.path.join(root, fname)
                try:
                    total_bytes += os.path.getsize(fpath)
                except OSError:
                    pass
        dl_secs = time.monotonic() - start_dl
        logger.info(
            f"Model snapshot ready: {snapshot_path} | files: {total_files} | size: {total_bytes / (1024**2):.1f} MB | took {dl_secs:.1f}s"
        )

        # Load model with Unsloth for memory efficiency from the local snapshot
        # Load without rope_scaling parameter to avoid validation issues
        logger.info("Loading model weights/tokenizer from local snapshot…")
        hb2_stop = threading.Event()
        hb2_thread = threading.Thread(
            target=_heartbeat,
            args=("Initializing model… still working (weight init/CPU→GPU)" , hb2_stop, 30),
            daemon=True,
        )
        start_load = time.monotonic()
        hb2_thread.start()
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=snapshot_path,
                max_seq_length=self.config.max_seq_length,
                dtype=getattr(torch, self.config.dtype),
                load_in_4bit=self.config.load_in_4bit,
                token=os.getenv("HF_TOKEN"),  # HuggingFace token if needed
            )
        finally:
            hb2_stop.set()
            hb2_thread.join(timeout=5)
        load_secs = time.monotonic() - start_load
        logger.info(f"Model load complete in {load_secs:.1f}s")

        # Fix RoPE scaling configuration for DeepSeek models
        if "deepseek" in self.config.model_name.lower():
            logger.info("DeepSeek model detected - ensuring rope_scaling configuration...")

            # Sanitize in-place to ensure float types without forcing a full config re-validate
            current = getattr(self.model.config, 'rope_scaling', None)
            self.model.config.rope_scaling = _sanitize_rope_scaling(current)
            logger.info(f"rope_scaling set to: {self.model.config.rope_scaling}")

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
        num_workers = min(max(4, (os.cpu_count() or 8) // 2), 16)
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=False,
            bf16=True,
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
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            torch_compile=True if os.getenv("ENABLE_TORCH_COMPILE") == "1" else False,
            torch_compile_backend=os.getenv("TORCH_COMPILE_BACKEND", "inductor"),
            run_name="deepseek-coder-finetune",
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        )

        # Create trainer with Unsloth optimizations
        dataset_proc = min(8, max(2, (os.cpu_count() or 8) // 2))
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=dataset_proc,
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

