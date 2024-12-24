"""Baseline NUFFT operators."""

__all__ = ["NUFFT", "NUFFTAdjoint"]

import warnings

import numpy as np
from numpy.typing import NDArray

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy import get_device
    from sigpy.config import cupy_enabled
    from sigpy.linop import Linop

from mrinufft import get_operator
from mrinufft._array_compat import with_numpy_cupy


class NUFFT(Linop):
    """
    NUFFT linear operator.

    Parameters
    ----------
    ishape : tuple[int]
        Shape of the input image array ``(..., H, W)`` (2D) or ``(..., D, H, W)`` (3D),
        with ``...`` representing an arbitrary number of batch dimensions ``B``.
    coord : NDArray
        Coordinates with values in the range ``(-0.5, 0.5)``, shaped ``(nintl, npts, ndims)``.
    eps: float, optional
        Numerical precision for NUFFT. The default is ``1e-6``.

    """

    def __init__(self, ishape: tuple[int], coord: NDArray, eps: float = 1e-6):
        ndim = coord.shape[-1]
        self.eps = eps
        self.nbatches = np.prod(ishape[:-ndim])
        oshape = list(ishape[:-ndim]) + list(coord.shape[:-1])
        super().__init__(oshape, ishape)

        # get grid shape
        self.mtx = ishape[-ndim:]
        self.cpu_nufft = get_operator("finufft")(
            samples=coord.reshape(-1, ndim),
            shape=self.mtx,
            n_batchs=self.nbatches,
            n_trans=self.nbatches,
            eps=self.eps,
        )

        if cupy_enabled:
            self.gpu_nufft = get_operator("cufinufft")(
                samples=coord.reshape(-1, ndim),
                shape=self.mtx,
                n_batchs=self.nbatches,
                n_trans=self.nbatches,
                eps=self.eps,
            )

    @with_numpy_cupy
    def _apply(self, input):
        device_id = get_device(input).id
        _input = input.reshape(self.nbatches, *input.shape[-self.ndim :])
        if device_id < 0:
            output = self.cpu_nufft.op(_input)
        else:
            output = self.gpu_nufft.op(_input)
        return output.reshape(self.oshape)

    def _adjoint_linop(self):
        return NUFFTAdjoint(self.ishape, self.coord, self.eps)


class NUFFTAdjoint(Linop):
    """
    NUFFT adjoint linear operator.

    Parameters
    ----------
    oshape : tuple[int]
        Shape of the output image array ``(..., H, W)`` (2D) or ``(..., D, H, W)`` (3D),
        with ``...`` representing an arbitrary number of batch dimensions ``B``.
    coord : NDArray
        Coordinates with values in the range ``(-0.5, 0.5)``, shaped ``(nintl, npts, ndims)``..
    eps: float, optional
        Numerical precision for NUFFT. The default is ``1e-6``.

    """

    def __init__(self, oshape: tuple[int], coord: NDArray, eps: float = 1e-6):
        ndim = coord.shape[-1]
        self.eps = eps
        self.nbatches = np.prod(oshape[:-ndim])
        ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])
        super().__init__(oshape, ishape)

        # get grid shape
        self.mtx = oshape[-ndim:]
        self.cpu_nufft = get_operator("finufft")(
            samples=coord.reshape(-1, ndim),
            shape=self.mtx,
            n_batchs=self.nbatches,
            n_trans=self.nbatches,
            eps=self.eps,
        )

        if cupy_enabled:
            self.gpu_nufft = get_operator("cufinufft")(
                samples=coord.reshape(-1, ndim),
                shape=self.mtx,
                n_batchs=self.nbatches,
                n_trans=self.nbatches,
                eps=self.eps,
            )

    @with_numpy_cupy
    def _apply(self, input):
        device_id = get_device(input).id
        _input = input.reshape(self.nbatches, -1)
        if device_id < 0:
            output = self.cpu_nufft.adj_op(_input)
        else:
            output = self.gpu_nufft.adj_op(_input)
        return output.reshape(self.oshape)

    def _adjoint_linop(self):
        return NUFFT(self.oshape, self.coord, self.eps)
