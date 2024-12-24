"""Baseline subspace NUFFT operators."""

__all__ = ["SubspaceNUFFT", "SubspaceNUFFTAdjoint"]

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
        serial: bool = True,
    ):
        ndim = coord.shape[-1]
        self.eps = eps
        self.basis = basis
        self.ncoeffs, self.nstacks = basis.shape
        self.nbatches = np.prod(ishape[: -ndim - 1])
        self.serial = serial
        oshape = list(ishape[:-ndim]) + list(coord.shape[:-1])
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
        else:
            self.cpu_nufft = get_operator("finufft")(
                samples=coord.reshape(-1, ndim),
                shape=self.mtx,
                n_batchs=self.nbatches * self.ncoeff,
                n_trans=self.nbatches * self.ncoeff,
                eps=self.eps,
            )

            if cupy_enabled():
                self.gpu_nufft = get_operator("cufinufft")(
                    samples=coord.reshape(-1, ndim),
                    shape=self.mtx,
                    n_batchs=self.nbatches * self.ncoeff,
                    n_trans=self.nbatches * self.ncoeff,
                    eps=self.eps,
                )

    @with_numpy_cupy
    def _apply(self, input):
        device_id = get_device(input).id
        _input = input.reshape(self.nbatches * self.ncoeff, *input.shape[-self.ndim :])
        if self.serial:
            output = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for n in range(self.nstacks):
                    if device_id < 0:
                        _output = self.cpu_nufft.op(_input)
                    else:
                        _output = self.gpu_nufft.op(_input)
                    _output = _output.reshape(self.nbatches, self.ncoeffs, -1)
                    _output = (
                        _output * self.basis[:, n][..., None]
                    )  # (nbatches, ncoeffs, nintl*npts) * (ncoeffs, 1)
                    output.append(_output.sum(axis=-2))  # (nbatches, nintl*npts)
            output = np.stack(output).swapaxes(0, 1)  # (nbatches, nstacks, nintl*npts)
        else:
            with warnings.catch_warnings():
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
        self.nstacks = oshape[-ndim - 1]
        self.nbatches = np.prod(oshape[: -ndim - 1])
        ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])
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
        else:
            self.cpu_nufft = get_operator("finufft")(
                samples=coord.reshape(-1, ndim),
                shape=self.mtx,
                n_batchs=self.nbatches * self.ncoeff,
                n_trans=self.nbatches * self.ncoeff,
                eps=self.eps,
            )

            if cupy_enabled():
                self.gpu_nufft = get_operator("cufinufft")(
                    samples=coord.reshape(-1, ndim),
                    shape=self.mtx,
                    n_batchs=self.nbatches * self.ncoeff,
                    n_trans=self.nbatches * self.ncoeff,
                    eps=self.eps,
                )

    @with_numpy_cupy
    def _apply(self, input):
        device_id = get_device(input).id
        if self.serial:
            _input = input.reshape(self.nbatches, self.nstacks, -1)
            _input = input.swapaxes(0, 1)
            output = 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for n in range(self.nstacks):
                    if device_id < 0:
                        _output = self.cpu_nufft.adj_op(
                            _input[n]
                        )  # (nbatches, *self.mtx)
                    else:
                        _output = self.gpu_nufft.adj_op(
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
                self.basis[:, None, :] * _input
            )  # (ncoeff, nbatches, nstacks, nintl*npts)
            _input = _input.reshape(self.ncoeff * self.nbatches, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if device_id < 0:
                    output = self.cpu_nufft.adj_op(
                        _input
                    )  # (ncoeff * nbatches, *self.mtx)
                else:
                    output = self.gpu_nufft.adj_op(
                        _input
                    )  # (ncoeff * nbatches, *self.mtx)
            output = output.reshape(self.ncoeff, self.nbatches, *self.mtx).swapaxes(
                0, 1
            )
        return output.reshape(self.oshape)

    def _adjoint_linop(self):
        return SubspaceNUFFT(
            self.oshape, self.coord, self.basis.T, self.eps, self.serial
        )
