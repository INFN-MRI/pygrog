"""Baseline stacked NUFFT operators."""

__all__ = ["StackedNUFFT", "StackedNUFFTAdjoint"]

import warnings

import numpy as np
from numpy.typing import NDArray

from sigpy import get_device
from sigpy.config import cupy_enabled
from sigpy.linop import Linop

from mrinufft import get_operator
from mrinufft._array_compat import with_numpy_cupy


class StackedNUFFT(Linop):
    """
    Stacked NUFFT linear operator.

    Parameters
    ----------
    ishape : tuple[int]
        Shape of the input image array ``(..., C, H, W)`` (2D) or ``(..., C, D, H, W)`` (3D),
        with ``C`` representing a stack of trajectories axis (z-encodings, frames, contrasts)
        and ``...`` representing an arbitrary number of batch dimensions ``B``.
    coord : NDArray
        Coordinates with values in the range ``(-0.5, 0.5)``, shaped ``(C, nintl, npts, ndims)``.
    eps: float, optional
        Numerical precision for NUFFT. The default is ``1e-6``.

    """

    def __init__(self, ishape: tuple[int], coord: NDArray, eps: float = 1e-6):
        ndim = coord.shape[-1]
        self.eps = eps
        self.nstacks = ishape[-ndim - 1]
        self.nbatches = np.prod(ishape[: -ndim - 1])
        oshape = list(ishape[:-ndim]) + list(coord.shape[:-1])
        super().__init__(oshape, ishape)

        # get grid shape
        self.mtx = ishape[-ndim:]
        self.cpu_nufft = [
            get_operator("finufft")(
                samples=coord[n].reshape(-1, ndim),
                shape=self.mtx,
                n_batchs=self.nbatches,
                n_trans=self.nbatches,
                eps=self.eps,
            )
            for n in range(self.nstacks)
        ]

        if cupy_enabled():
            self.gpu_nufft = [
                get_operator("cufinufft")(
                    samples=coord[n].reshape(-1, ndim),
                    shape=self.mtx,
                    n_batchs=self.nbatches,
                    n_trans=self.nbatches,
                    eps=self.eps,
                )
                for n in range(self.nstacks)
            ]

    @with_numpy_cupy
    def _apply(self, input):
        device_id = get_device(input).id
        _input = input.reshape(self.nbatches, *input.shape[-self.ndim - 1 :])
        _input = input.swapaxes(0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if device_id < 0:
                output = [self.cpu_nufft.op(_input[n]) for n in range(self.nstacks)]
            else:
                output = [self.gpu_nufft.op(_input[n]) for n in range(self.nstacks)]
        output = np.stack(output).swapaxes(0, 1)
        return output.reshape(self.oshape)

    def _adjoint_linop(self):
        return StackedNUFFTAdjoint(self.ishape, self.coord, self.eps)


class StackedNUFFTAdjoint(Linop):
    """
    Stacked NUFFT adjoint linear operator.

    Parameters
    ----------
    oshape : tuple[int]
        Shape of the output image array ``(..., C, H, W)`` (2D) or ``(..., C, D, H, W)`` (3D),
        with ``C`` representing a stack of trajectories axis (z-encodings, frames, contrasts)
        and ``...`` representing an arbitrary number of batch dimensions ``B``.
    coord : NDArray
        Coordinates with values in the range ``(-0.5, 0.5)``, shaped ``(C, nintl, npts, ndims)``.
    eps: float, optional
        Numerical precision for NUFFT. The default is ``1e-6``.

    """

    def __init__(self, oshape: tuple[int], coord: NDArray, eps: float = 1e-6):
        ndim = coord.shape[-1]
        self.eps = eps
        self.nstacks = oshape[-ndim - 1]
        self.nbatches = np.prod(oshape[: -ndim - 1])
        ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])
        super().__init__(oshape, ishape)

        # get grid shape
        self.mtx = oshape[-ndim:]
        self.cpu_nufft = [
            get_operator("finufft")(
                samples=coord[n].reshape(-1, ndim),
                shape=self.mtx,
                n_batchs=self.nbatches,
                n_trans=self.nbatches,
                eps=self.eps,
            )
            for n in range(self.nstacks)
        ]

        if cupy_enabled():
            self.gpu_nufft = [
                get_operator("cufinufft")(
                    samples=coord[n].reshape(-1, ndim),
                    shape=self.mtx,
                    n_batchs=self.nbatches,
                    n_trans=self.nbatches,
                    eps=self.eps,
                )
                for n in range(self.nstacks)
            ]

    @with_numpy_cupy
    def _apply(self, input):
        device_id = get_device(input).id
        _input = input.reshape(self.nbatches, self.nstacks, -1)
        _input = input.swapaxes(0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if device_id < 0:
                output = [self.cpu_nufft.adj_op(_input[n]) for n in range(self.nstacks)]
            else:
                output = [self.gpu_nufft.adj_op(_input[n]) for n in range(self.nstacks)]
        output = np.stack(output).swapaxes(0, 1)
        return output.reshape(self.oshape)

    def _adjoint_linop(self):
        return StackedNUFFT(self.oshape, self.coord, self.eps)
