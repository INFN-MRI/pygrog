"""Autocalibration region extraction subroutines."""

__all__ = ["extract_acr"]

import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import sigpy

from numpy.typing import NDArray
from mrinufft._array_compat import with_numpy_cupy


@with_numpy_cupy
def extract_acr(
    data: NDArray,
    cal_width: int = 24,
    ndim: int = None,
    trajectory: NDArray = None,
    weights: NDArray = None,
    shape: int = None,
) -> NDArray | tuple[NDArray, NDArray, NDArray | None]:
    """
    Extract calibration region from input dataset.

    Parameters
    ----------
    data : NDArray
        Input k-space dataset of shape ``(..., *shape)`` (Cartesian) or
        ``(..., npts)`` (Non Cartesian).
    cal_width : int, optional
        Calibration region size. The default is ``24``.
    ndim : int, optional
        Number of spatial dimensions. The default is ``None``.
        Required for Cartesian datasets.
    trajectory : NDArray, optional
        K-space trajectory of shape ``(..., npts, ndim)``, normalized between ``(-0.5, 0.5)``.
        Required for Non Cartesian datasets. The default is ``None``.
    weights : NDArray, optional
        K-space density compensation of shape ``(..., npts)``. The default is ``None``.
    shape : int, optional
        Matrix size of shape ``(ndim,)``.
        Required for Non Cartesian datasets. The default is ``None``.

    Raises
    ------
    ValueError
        If ``ndim`` is not provided for Cartesian datasets (``trajectory = None``) or
        ``shape`` is not provided for Non Cartesian datasets (``trajectory != None``).

    Returns
    -------
    cal_data : NDArray
        Calibration dataset of shape ``(..., *[cal_width]*ndim)`` (Cartesian) or
        ``(..., cal_width)`` (Non Cartesian).
    cal_trajectory : NDArray, optional
        Trajectory for calibration dataset of shape ``(..., cal_width, ndim)``.
    cal_weights : NDArray, optional
        Density compensation for calibration dataset of shape ``(..., cal_width)``.

    """
    if trajectory is None:
        if ndim is None:
            raise ValueError(
                "Please provide number of spatial dimensions for Cartesian datasets"
            )
        shape = list(data.shape[-ndim:])
        return sigpy.resize(data, data.shape[:-ndim] + ndim * [cal_width])

    else:
        if shape is None:
            raise ValueError("Please provide matrix size for Non Cartesian datasets")

        # get indexes for calibration samples
        cal_width = int(
            np.ceil(cal_width * 2**0.5)
        )  # make sure we can extract a squared cal region later
        cal_idx = np.amax(np.abs(trajectory), axis=-1) < cal_width / shape / 2

        _data = data[..., cal_idx]
        _trajectory = trajectory[..., cal_idx, :]
        if weights is not None:
            _weights = weights[..., cal_idx]
        else:
            _weights = None

        return _data, _trajectory, _weights
