"""Baseline subspace NUFFT operators."""

__all__ = ["SubspaceNUFFT", "SubspaceNUFFTAdjoint"]

import warnings

warnings.simplefilter("ignore")

import numpy as np
from numpy.typing import NDArray

from sigpy.config import cupy_enabled
from sigpy.linop import Linop

from mrinufft import get_operator
from mrinufft._array_compat import with_torch


class SubspaceNUFFT(Linop):
    """
    Subspace NUFFT linear operator.

    Parameters
    ----------
    ishape : tuple[int]
        Shape of the input image array ``(..., Cin, H, W)`` (2D) or ``(..., Cin, D, H, W)`` (3D),
        with ``Cin`` representing subspace coefficient axis
        and ``...`` representing an arbitrary number of batch dimensions ``B``.
    coord : NDArray
        Coordinates with values in the range ``(-0.5, 0.5)``, shaped ``(Cout, nintl, npts, ndims)``.
    basis : NDArray
        Subspace basis of shape ``(Cin, Cout)``.
    eps: float, optional
        Numerical precision for NUFFT. The default is ``1e-6``.
    serial: bool, optional
        If True, process each stack in parallel. The default is ``True``.

    """

    def __init__(
        self,
        ishape: tuple[int],
        coord: NDArray,
        basis: NDArray,
        eps: float = 1e-6,
        serial: bool = False,
    ):
        ndim = coord.shape[-1]
        self.eps = eps
        self.basis = basis
        self.ncoeffs, self.nstacks = basis.shape
        self.nbatches = np.prod(ishape[: -ndim - 1])
        self.serial = serial
        self.coord = coord
        self.ndim = coord.shape[-1]
        oshape = list(ishape[: -ndim - 1]) + list(coord.shape[:-1])
        super().__init__(oshape, ishape)

        # get grid shape
        self.mtx = ishape[-ndim:]
        if self.serial:
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

            if cupy_enabled:
                self.gpu_nufft = [
                    get_operator("cufinufft")(
                        samples=coord[n].reshape(-1, ndim),
                        shape=self.mtx,
                        n_batchs=self.nbatches,
                        # n_trans=self.nbatches,
                        eps=self.eps,
                    )
                    for n in range(self.nstacks)
                ]
        else:
            self.cpu_nufft = get_operator("finufft")(
                samples=coord.reshape(-1, ndim),
                shape=self.mtx,
                n_batchs=self.nbatches * self.ncoeffs,
                n_trans=self.nbatches * self.ncoeffs,
                eps=self.eps,
            )

            if cupy_enabled:
                self.gpu_nufft = get_operator("cufinufft")(
                    samples=coord.reshape(-1, ndim),
                    shape=self.mtx,
                    n_batchs=self.nbatches * self.ncoeffs,
                    # n_trans=self.nbatches * self.ncoeffs,
                    eps=self.eps,
                )

    @with_torch
    def _apply(self, input):
        device_id = input.device.index
        _input = input.reshape(self.nbatches * self.ncoeffs, *input.shape[-self.ndim :])
        if self.serial:
            output = []
            for n in range(self.nstacks):
                if device_id < 0:
                    _output = self.cpu_nufft[n].op(_input)
                else:
                    _output = self.gpu_nufft[n].op(_input)
                _output = _output.reshape(self.nbatches, self.ncoeffs, -1)
                _output = (
                    _output * self.basis[:, n][..., None]
                )  # (nbatches, ncoeffs, nintl*npts) * (ncoeffs, 1)
                output.append(_output.sum(axis=-2))  # (nbatches, nintl*npts)
            output = np.stack(output).swapaxes(0, 1)  # (nbatches, nstacks, nintl*npts)
        else:
            if device_id < 0:
                output = self.cpu_nufft.op(_input)
            else:
                output = self.gpu_nufft.op(_input)
            output = output.reshape(self.nbatches, self.ncoeffs, self.nstacks, -1)
            output = (
                output * self.basis[..., None]
            )  # (nbatches, ncoeffs, nstacks, nintl*npts) * (ncoeffs, nstacks, 1)
            output = output.sum(axis=-3)  # (nbatches, nstacks, nintl*npts)
        return output.reshape(self.oshape)

    def _adjoint_linop(self):
        return SubspaceNUFFTAdjoint(
            self.ishape, self.coord, self.basis.T, self.eps, self.serial
        )


class SubspaceNUFFTAdjoint(Linop):
    """
    Subspace NUFFT adjoint linear operator.

    Parameters
    ----------
    oshape : tuple[int]
        Shape of the output image array ``(..., Cout, H, W)`` (2D) or ``(..., Cout, D, H, W)`` (3D),
        with ``C`` representing subspace coefficient axis
        and ``...`` representing an arbitrary number of batch dimensions ``B``.
    coord : NDArray
        Coordinates with values in the range ``(-0.5, 0.5)``, shaped ``(Cin, nintl, npts, ndims)``.
    basis : NDArray
        Subspace basis of shape ``(Cin, Cout)``.
    eps: float, optional
        Numerical precision for NUFFT. The default is ``1e-6``.
    serial: bool, optional
        If True, process each stack in parallel. The default is ``True``.

    """

    def __init__(
        self,
        oshape: tuple[int],
        coord: NDArray,
        basis: NDArray,
        eps: float = 1e-6,
        serial: bool = True,
    ):
        ndim = coord.shape[-1]
        self.eps = eps
        self.basis = basis
        self.nstacks, self.ncoeffs = basis.shape
        self.nbatches = np.prod(oshape[: -ndim - 1])
        self.serial = serial
        self.coord = coord
        self.ndim = coord.shape[-1]
        ishape = list(oshape[: -ndim - 1]) + list(coord.shape[:-1])
        super().__init__(oshape, ishape)

        # get grid shape
        self.mtx = oshape[-ndim:]
        if self.serial:
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

            if cupy_enabled:
                self.gpu_nufft = [
                    get_operator("cufinufft")(
                        samples=coord[n].reshape(-1, ndim),
                        shape=self.mtx,
                        n_batchs=self.nbatches,
                        # n_trans=self.nbatches,
                        eps=self.eps,
                    )
                    for n in range(self.nstacks)
                ]
        else:
            self.cpu_nufft = get_operator("finufft")(
                samples=coord.reshape(-1, ndim),
                shape=self.mtx,
                n_batchs=self.nbatches * self.ncoeffs,
                n_trans=self.nbatches * self.ncoeffs,
                eps=self.eps,
            )

            if cupy_enabled:
                self.gpu_nufft = get_operator("cufinufft")(
                    samples=coord.reshape(-1, ndim),
                    shape=self.mtx,
                    n_batchs=self.nbatches * self.ncoeffs,
                    # n_trans=self.nbatches * self.ncoeffs,
                    eps=self.eps,
                )

    @with_torch
    def _apply(self, input):
        device_id = input.device.index
        if self.serial:
            _input = input.reshape(self.nbatches, self.nstacks, -1)
            _input = input.swapaxes(0, 1)
            output = 0.0
            for n in range(self.nstacks):
                if device_id < 0:
                    _output = self.cpu_nufft[n].adj_op(
                        _input[n]
                    )  # (nbatches, *self.mtx)
                else:
                    _output = self.gpu_nufft[n].adj_op(
                        _input[n]
                    )  # (nbatches, *self.mtx)
                output += (
                    self.basis[n] * _output[..., None]
                )  # (ncoeff,) * (nbatches, *self.mtx, 1)
            output = (
                output[None, ...].swapaxes(0, -1)[..., 0].swapaxes(0, 1)
            )  # (nbatches, ncoeff, *self.mtx)
        else:
            _input = input.reshape(self.nbatches, self.nstacks, -1)
            _input = (
                self.basis.T[:, None, :, None] * _input
            )  # (ncoeff, nbatches, nstacks, nintl*npts)
            _input = _input.reshape(self.ncoeffs * self.nbatches, -1)
            if device_id < 0:
                output = self.cpu_nufft.adj_op(_input)  # (ncoeff * nbatches, *self.mtx)
            else:
                output = self.gpu_nufft.adj_op(_input)  # (ncoeff * nbatches, *self.mtx)
            output = output.reshape(self.ncoeffs, self.nbatches, *self.mtx).swapaxes(
                0, 1
            )
        return output.reshape(self.oshape)

    def _adjoint_linop(self):
        return SubspaceNUFFT(
            self.oshape, self.coord, self.basis.T, self.eps, self.serial
        )
