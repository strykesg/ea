"""
DeepSeek-Coder-V2-Lite-Instruct Fine-Tuner Package

A comprehensive solution for fine-tuning and quantizing large language models
with a focus on long-form, complex reasoning capabilities.
"""

from .data_loader import load_and_prepare_dataset, JSONLDatasetLoader
from .finetune import main as finetune_main, FinetuneConfig, DeepSeekFinetuner
from .quantize import main as quantize_main, GGUFQuantizer
from .run_finetune import FinetunePipeline

__version__ = "0.1.0"
__all__ = [
    "load_and_prepare_dataset",
    "JSONLDatasetLoader",
    "finetune_main",
    "FinetuneConfig",
    "DeepSeekFinetuner",
    "quantize_main",
    "GGUFQuantizer",
    "FinetunePipeline"
]