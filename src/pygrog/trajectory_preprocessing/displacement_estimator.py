import torch
from typing import List


class DisplacementEstimator:
    """Estimates displacement vectors between source and target points."""

    def __init__(self, trajectory: torch.Tensor):
        self.trajectory = trajectory

    def estimate_displacement_distribution(
        self, arc_radius: float
    ) -> List[torch.Tensor]:
        """Estimates displacement vectors for each target point."""
        pass

    def upsample_trajectory(self, factor: int = 4) -> torch.Tensor:
        """Upsamples trajectory using linear interpolation."""
        pass
