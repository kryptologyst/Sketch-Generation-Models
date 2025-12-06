"""
Sketch Generation Models - Modern PyTorch Implementation

This project implements state-of-the-art sketch generation models using PyTorch,
including Pix2Pix, CycleGAN, and edge detection baselines for image-to-sketch conversion.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from pathlib import Path

# Set deterministic behavior
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Set seed and device
set_seed(42)
DEVICE = get_device()

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
