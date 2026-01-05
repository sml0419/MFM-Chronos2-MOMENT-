"""
Utility functions for memory management and model saving.
"""

import gc
import shutil
from pathlib import Path
import torch


def clear_memory():
    """Clear GPU and CPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_memory_stats(prefix=""):
    """Print current GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated

        print(f"{prefix}GPU Memory: Allocated={allocated:.2f}GB, "
              f"Reserved={reserved:.2f}GB, "
              f"Free={free:.2f}GB, "
              f"Total={total:.2f}GB")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint with error handling and atomic write"""
    filepath = Path(filepath)
    temp_filepath = None

    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check disk space
        stat = shutil.disk_usage(filepath.parent)
        free_gb = stat.free / (1024**3)
        if free_gb < 2:
            print(f"WARNING: Low disk space ({free_gb:.2f}GB free). Skipping checkpoint save.")
            return False

        # Save to temporary file first
        temp_filepath = filepath.with_suffix('.tmp')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        torch.save(checkpoint, temp_filepath)
        temp_filepath.replace(filepath)

        print(f"Checkpoint saved: {filepath}")
        return True

    except Exception as e:
        print(f"WARNING: Failed to save checkpoint: {e}")
        if temp_filepath and temp_filepath.exists():
            try:
                temp_filepath.unlink()
            except:
                pass
        return False


def safe_save_model(model, filepath, model_name="model"):
    """Safely save model state dict with error handling and atomic write"""
    filepath = Path(filepath)
    temp_filepath = None

    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check disk space
        stat = shutil.disk_usage(filepath.parent)
        free_gb = stat.free / (1024**3)
        if free_gb < 2:
            print(f"WARNING: Low disk space ({free_gb:.2f}GB free). Skipping {model_name} save.")
            return False

        # Save to temporary file first for atomic write
        temp_filepath = filepath.with_suffix('.tmp')
        torch.save(model.state_dict(), temp_filepath)
        temp_filepath.replace(filepath)

        print(f"{model_name} saved: {filepath}")
        return True

    except Exception as e:
        print(f"WARNING: Failed to save {model_name}: {e}")
        if temp_filepath and temp_filepath.exists():
            try:
                temp_filepath.unlink()
            except:
                pass
        return False


def expected_seq_len(model, default: int = 512) -> int:
    """Try to read MOMENT configured sequence length from model/pipeline."""
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        cfg = getattr(root, "config", None)
        if cfg is not None and hasattr(cfg, "seq_len"):
            try:
                v = int(getattr(cfg, "seq_len"))
                if v > 0:
                    return v
            except Exception:
                pass
        if hasattr(root, "seq_len"):
            try:
                v = int(getattr(root, "seq_len"))
                if v > 0:
                    return v
            except Exception:
                pass
    return int(default)


def pad_or_truncate_to_seq_len(x, input_mask, seq_len: int):
    """Ensure x has shape [B, C, seq_len] and input_mask [B, seq_len]. Left-pad zeros if needed."""
    import torch
    if input_mask.dtype != torch.long:
        input_mask = input_mask.long()
    b, c, l = x.shape
    if l == seq_len:
        return x, input_mask
    if l > seq_len:
        return x[..., -seq_len:], input_mask[..., -seq_len:]
    pad = seq_len - l
    x_pad = torch.zeros((b, c, pad), device=x.device, dtype=x.dtype)
    m_pad = torch.zeros((b, pad), device=input_mask.device, dtype=torch.long)
    return torch.cat([x_pad, x], dim=-1), torch.cat([m_pad, input_mask], dim=-1)


def set_all_seeds(seed: int):
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism flags (best-effort)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
