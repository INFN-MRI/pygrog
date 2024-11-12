import torch
from typing import Tuple

def sparse_to_dense(data: torch.Tensor, coords: torch.Tensor, weights: torch.Tensor, 
                    output_shape: Tuple[int, int]) -> torch.Tensor:
    """Converts sparse (data, coords, weights) to dense Cartesian grid."""
    pass
