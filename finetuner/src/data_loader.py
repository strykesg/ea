"""
Data loading utilities for fine-tuning with JSONL datasets.
Handles the format similar to training_data.jsonl with instruction, input, output fields.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datasets import Dataset, load_dataset
import pandas as pd

logger = logging.getLogger(__name__)


class JSONLDatasetLoader:
    """Load and process JSONL datasets for fine-tuning."""

    def __init__(self, data_file: str = "data.jsonl"):
        """
        Initialize the data loader.

        Args:
            data_file: Path to the JSONL data file
        """
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

    def load_jsonl_data(self) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file.

        Returns:
            List of dictionaries containing the training examples
        """
        data = []
        logger.info(f"Loading data from {self.data_file}")

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)
                    data.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(data)} examples from {self.data_file}")
        return data

    def format_for_training(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format the data for training by combining instruction and input as prompt,
        and using output as the completion.

        Args:
            data: Raw data from JSONL file

        Returns:
            Formatted data suitable for training
        """
        formatted_data = []

        for example in data:
            # Handle the format from training_data.jsonl
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output_text = example.get('output', '')

            # Combine instruction and input as the prompt
            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}"
            else:
                prompt = instruction

            formatted_example = {
                'prompt': prompt,
                'completion': output_text,
                'id': example.get('id', ''),
                'generated_at': example.get('generated_at', '')
            }

            formatted_data.append(formatted_example)

        logger.info(f"Formatted {len(formatted_data)} examples for training")
        return formatted_data

    def create_huggingface_dataset(self) -> Dataset:
        """
        Create a HuggingFace Dataset from the JSONL file.

        Returns:
            HuggingFace Dataset object
        """
        raw_data = self.load_jsonl_data()
        formatted_data = self.format_for_training(raw_data)

        # Create dataset
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))

        # Add some basic validation
        logger.info(f"Dataset info: {dataset}")
        logger.info(f"Sample example: {dataset[0] if len(dataset) > 0 else 'No examples'}")

        return dataset

    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        raw_data = self.load_jsonl_data()

        stats = {
            'total_examples': len(raw_data),
            'examples_with_instruction': sum(1 for x in raw_data if x.get('instruction')),
            'examples_with_input': sum(1 for x in raw_data if x.get('input')),
            'examples_with_output': sum(1 for x in raw_data if x.get('output')),
            'avg_instruction_length': 0,
            'avg_input_length': 0,
            'avg_output_length': 0
        }

        if raw_data:
            stats['avg_instruction_length'] = sum(len(str(x.get('instruction', ''))) for x in raw_data) / len(raw_data)
            stats['avg_input_length'] = sum(len(str(x.get('input', ''))) for x in raw_data) / len(raw_data)
            stats['avg_output_length'] = sum(len(str(x.get('output', ''))) for x in raw_data) / len(raw_data)

        return stats


def load_and_prepare_dataset(data_file: str = "data.jsonl") -> Dataset:
    """
    Convenience function to load and prepare dataset in one call.

    Args:
        data_file: Path to the JSONL data file

    Returns:
        HuggingFace Dataset ready for training
    """
    loader = JSONLDatasetLoader(data_file)
    return loader.create_huggingface_dataset()


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    try:
        loader = JSONLDatasetLoader("data.jsonl")
        dataset = loader.create_huggingface_dataset()
        stats = loader.get_data_stats()

        print("Dataset loaded successfully!")
        print(f"Statistics: {stats}")
        print(f"Dataset features: {dataset.column_names}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data.jsonl exists in the current directory")
    except Exception as e:
        print(f"Error loading dataset: {e}")

