"""Device handling utilities for CPU/CUDA/MPS."""

import torch
from typing import Literal


def get_device() -> torch.device:
    """
    Get the best available device.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_device(device: Literal["cuda", "mps", "cpu", "auto"] = "auto") -> torch.device:
    """
    Set and return the specified device.
    
    Args:
        device: Device type. If "auto", uses get_device()
        
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        return get_device()
    
    device_obj = torch.device(device)
    
    # Verify device availability
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS device requested but not available")
    
    return device_obj


def device_info() -> dict:
    """Get information about available devices."""
    info = {
        "cpu_available": True,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        info["cuda_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
    
    return info


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to device with type preservation."""
    return tensor.to(device)
