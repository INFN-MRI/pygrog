"""Test dataset generation."""

__all__ = ["generate_mrf_case", "generate_spgr_case"]

import warnings

import numpy as np

from numpy.typing import NDArray
from mrinufft import get_operator

import mrtwin
import torchsim

from .._svd import estimate_bloch_subspace

from ._schedule import vfa_schedule
from ._trajectory import generate_spiral_trajectory

import sigpy


def generate_mrf_case(
    ndim: int,
    npix: int = 200,
    res: float = 1.125,
    num_coeff: int = 4,
    num_coils: int = 8,
    brainweb: bool = False,
) -> tuple[NDArray]:
    """
    Generate MR Fingerprinting datasets for test and benchmark.

    For ``ndim == 2``, this is a 2D spiral MR fingerprinting with ``880`` contrasts.
    For ``ndim == 3``, this is a 3D spiral projection MR fingerprinting with ``880``
    contrasts and ``55`` interleaves per contrast.

    In both cases, matrix and resolution are isotropic.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    npix : int, optional
        Matrix size. The default is ``200``.
    res : float, optional
        Image resolution in ``[mm]``. The default is ``1.125``.
    num_coeff : int, optional
        Number of subspace coefficients. The default is ``4``.
    num_coils : int, optional
        Number of coils. The default is ``8``.
    brainweb : bool, optional
        If ``True``, uses Brainweb phantom (more realistic, but involves download).
        If ``False``, uses Shepp Logan (no download, better for docstrings).

    Returns
    -------
    data : NDArray
        K-space data of shape ``(num_coils, num_contrasts, num_shots, num_pts)``.
    k : NDArray
        K-space trajectory of shape ``(num_contrasts, num_shots, num_pts, ndim)``.
    dcf : NDArray
        K-space density compensation of shape ``(num_contrasts, num_shots, num_pts)``.
    grog_training : NDArray
        Grappa trainining dataset of shape ``(num_coils, 24, 24)`` (``ndim == 2``) or
        ``(num_coils, 24, 24, 24)`` (``ndim == 3``).
    obj_coeff : NDArray
        Ground truth subspace coefficients for fully-sampled trajectory of shape
        ``(num_coeff, npix, npix)`` (``ndim == 2``) or ``(num_coeff, npix, npix, npix)``
        (``ndim == 3``).
    basis : NDArray
        Subspace basis of shape ``(num_contrasts, num_coeff)``.

    """
    # generate schedule
    TR = 8.0
    flip = vfa_schedule()

    # generate basis
    t1grid, t2grid = np.mgrid[5:3500:10, 5:350:10]
    training_data = torchsim.mrf_sim(flip, TR, t1grid.ravel(), t2grid.ravel())
    basis = estimate_bloch_subspace(training_data, num_coeff=num_coeff)

    # generate trajectory
    k, dcf, _ = generate_spiral_trajectory(
        ndim=ndim,
        npix=npix,
        ncontrasts=len(flip),
        res=res,
    )

    # generate object
    if brainweb:
        obj = mrtwin.brainweb_phantom(ndim, subject=4, shape=npix, output_res=res)
    else:
        obj = mrtwin.shepplogan_phantom(ndim, shape=npix)

    # generate sensitivity maps
    smaps = mrtwin.sensmap([num_coils] + ndim * [npix])

    # generate signals
    signals = torchsim.mrf_sim(flip, TR, obj.T1[1:], obj.T2[1:], M0=obj.M0[1:])
    signals_coeff = signals @ basis

    # ground truth images
    obj_coeff = np.zeros([num_coeff] + ndim * [npix], dtype=np.float32)
    for n in range(signals_coeff.shape[0]):
        mask = obj.segmentation == n + 1
        for m in range(num_coeff):
            obj_coeff[m, mask] = signals_coeff[n, m]

    # grog training data
    grog_training = smaps * obj_coeff[0]
    grog_training = sigpy.fft(grog_training, axes=list(range(-ndim, 0)))
    grog_training = sigpy.resize(grog_training, oshape=[num_coils] + ndim * [24])

    # kspace
    segmentation = [(obj.segmentation == n + 1) for n in range(signals.shape[0])]
    segmentation = np.stack(segmentation, axis=0)
    segmentation = smaps * segmentation[:, None, ...]

    # Fourier transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nufft = get_operator("finufft")(
            samples=k.reshape(-1, ndim),
            n_batchs=signals.shape[0],
            n_coils=smaps.shape[0],
            n_trans=signals.shape[0] * smaps.shape[0],
            shape=[npix] * ndim,
        )
    data = nufft.op(segmentation)
    data = data.reshape(signals.shape[0], smaps.shape[0], *k.shape[:-1])
    data = signals[:, None, :, None, None] * data
    data = data.sum(axis=0)

    return data, k, dcf, grog_training, obj_coeff, basis


def generate_spgr_case(
    ndim: int,
    npix: int = 440,
    res: float = 0.5,
    num_echoes: int = 6,
    num_coils: int = 8,
    brainweb: bool = False,
) -> tuple[NDArray]:
    """
    Generate multiecho SPGR datasets for test and benchmark.

    For ``ndim == 2``, this is a 2D spiral SPGR.
    For ``ndim == 3``, this is a 3D stack of spirals SPGR with 144 mm coverage in z.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    npix : int, optional
        Matrix size. The default is ``400``.
    res : float, optional
        Image resolution in ``[mm]``. The default is ``0.5``.
    num_echoes : int, optional
        Number of echoes. The default is ``6``.
    num_coils : int, optional
        Number of coils. The default is ``8``.
    brainweb : bool, optional
        If ``True``, uses Brainweb phantom (more realistic, but involves download).
        If ``False``, uses Shepp Logan (no download, better for docstrings).

    Returns
    -------
    data : NDArray
        K-space data of shape ``(num_coils, num_contrasts, num_shots, num_pts)``.
    k : NDArray
        K-space trajectory of shape ``(num_contrasts, num_shots, num_pts, ndim)``.
    dcf : NDArray
        K-space density compensation of shape ``(num_contrasts, num_shots, num_pts)``.
    grog_training : NDArray
        Grappa trainining dataset of shape ``(num_coils, 24, 24)`` (``ndim == 2``) or
        ``(num_coils, 24, 24, 24)`` (``ndim == 3``).
    obj_coeff : NDArray
        Ground truth subspace coefficients for fully-sampled trajectory of shape
        ``(num_coeff, npix, npix)`` (``ndim == 2``) or ``(num_coeff, npix, npix, npix)``
        (``ndim == 3``).
    basis : NDArray
        Subspace basis of shape ``(num_contrasts, num_coeff)``.

    """
    # generate trajectory
    k, dcf, _ = generate_spiral_trajectory(
        ndim=2,
        npix=npix,
        ncontrasts=num_echoes * 55,
        res=res,
    )
    k = k.reshape(55, num_echoes, *k.shape[-2:]).swapaxes(0, 1)
    dcf = dcf.reshape(55, num_echoes, dcf.shape[-1]).swapaxes(0, 1)

    # generate parameters
    flip = 10.0
    TR = 50.0
    TE = 4e-3 * k.shape[-2] * np.arange(num_echoes)

    # generate object
    if ndim == 2:
        mtx = npix
    else:
        mtx = (int(np.ceil(144 / res)), npix, npix)
    if brainweb:
        obj = mrtwin.brainweb_phantom(ndim, subject=4, shape=mtx, output_res=res)
    else:
        obj = mrtwin.shepplogan_phantom(ndim, shape=mtx)

    if ndim == 2:
        mtx = ndim * [mtx]
    else:
        mtx = list(mtx)

    # generate sensitivity maps
    smaps = mrtwin.sensmap([num_coils] + mtx)
    if ndim == 3:
        smaps = smaps.swapaxes(0, 1)

    # generate signals
    signals = torchsim.spgr_sim(flip, TE, TR, obj.T1[1:], obj.T2[1:], M0=obj.M0[1:])

    # ground truth images
    obj_echoes = np.zeros([num_echoes] + mtx, dtype=np.float32)
    for n in range(signals.shape[0]):
        mask = obj.segmentation == n + 1
        for m in range(num_echoes):
            obj_echoes[m, mask] = signals[n, m]

    if ndim == 3:
        obj_echoes = obj_echoes.swapaxes(0, 1)

    # grog training data
    if ndim == 2:
        grog_training = smaps * obj_echoes[0]
    else:
        grog_training = smaps * obj_echoes[:, 0]
    grog_training = sigpy.fft(grog_training, axes=list(range(-2, 0)))
    grog_training = sigpy.resize(
        grog_training, oshape=grog_training.shape[:-2] + [24, 24]
    )

    # kspace
    segmentation = [(obj.segmentation == n + 1) for n in range(signals.shape[0])]
    segmentation = np.stack(segmentation, axis=0)
    if ndim == 3:
        segmentation = smaps * segmentation[:, :, None, ...]
    else:
        segmentation = smaps * segmentation[:, None, ...]

    # Fourier transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nufft = get_operator("finufft")(
            samples=k.reshape(-1, ndim),
            n_batchs=signals.shape[0],
            n_coils=smaps.shape[0],
            n_trans=signals.shape[0] * smaps.shape[0],
            shape=mtx[-2:],
        )
    data = nufft.op(segmentation)
    data = data.reshape(signals.shape[0], *smaps.shape[:-2], *k.shape[:-1])
    data = signals[:, None, None, :, None, None] * data
    data = data.sum(axis=0)

    return data, k, dcf, grog_training, obj_echoes
