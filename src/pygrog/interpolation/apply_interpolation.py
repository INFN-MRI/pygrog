import torch
from torch import nn
from typing import Tuple

def apply_interpolation(model: nn.Module, non_cartesian_data: torch.Tensor, 
                        trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies trained GROG interpolator to non-Cartesian data."""
    pass
