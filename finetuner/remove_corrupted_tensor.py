#!/usr/bin/env python3
"""
Remove the specific corrupted tensor causing shape mismatch errors.
This is a targeted fix for the known [11206656, 1] tensor issue.
"""

import os
import torch
from pathlib import Path
import safetensors.torch

def remove_corrupted_tensor(model_path, output_path=None):
    """Remove the specific corrupted tensor that's causing loading failures."""
    if output_path is None:
        output_path = model_path

    print("🗑️  Removing Corrupted Tensor")
    print(f"   Input: {model_path}")
    print(f"   Output: {output_path}")

    # The corrupted tensor signature
    corrupted_shape = (11206656, 1)
    found_and_removed = False

    # Find safetensors files
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensors_files:
        print("❌ No safetensors files found!")
        return False

    for safetensors_file in safetensors_files:
        print(f"   Checking {safetensors_file.name}...")
        try:
            state_dict = safetensors.torch.load_file(safetensors_file, device='cpu')

            # Look for the corrupted tensor
            corrupted_key = None
            for name, tensor in state_dict.items():
                if tuple(tensor.shape) == corrupted_shape:
                    corrupted_key = name
                    print(f"   🎯 Found corrupted tensor: {name}")
                    break

            if corrupted_key:
                # Remove the corrupted tensor
                del state_dict[corrupted_key]
                found_and_removed = True
                print(f"   ✅ Removed corrupted tensor: {corrupted_key}")

                # Save the cleaned file
                if output_path != model_path:
                    os.makedirs(output_path, exist_ok=True)
                    output_file = Path(output_path) / safetensors_file.name
                else:
                    # Backup original
                    backup_file = safetensors_file.with_suffix('.safetensors.backup')
                    if not backup_file.exists():
                        safetensors_file.rename(backup_file)
                        print(f"   💾 Backed up original to: {backup_file.name}")
                    output_file = safetensors_file

                safetensors.torch.save_file(state_dict, output_file)
                print(f"   ✅ Saved cleaned file: {output_file.name}")
                break

        except Exception as e:
            print(f"   ❌ Failed to process {safetensors_file}: {e}")

    if found_and_removed:
        print("✅ Corrupted tensor successfully removed!")
        return True
    else:
        print("❌ Corrupted tensor not found")
        return False

def main():
    """Main function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python remove_corrupted_tensor.py <model_path> [output_path]")
        print("  model_path: Path to the model to clean")
        print("  output_path: Optional output path (defaults to modifying model_path in place)")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(model_path).exists():
        print(f"❌ Model path not found: {model_path}")
        sys.exit(1)

    try:
        success = remove_corrupted_tensor(model_path, output_path)
        if success:
            print("\n📝 The corrupted tensor has been removed.")
            print("   You can now try loading the model again.")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
