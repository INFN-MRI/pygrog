"""Parameter schedule generation subroutines."""

import numpy as np

from numpy.typing import NDArray


def vfa_schedule() -> NDArray:
    """Simple linear piecewise flip angle schedule (in ``[deg]``)."""
    return np.concatenate(
        (np.linspace(5, 60, 350), np.linspace(60, 2, 350), 2 * np.ones(180))
    )
