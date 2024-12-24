"""Trajectory generation subroutines."""

__all__ = ["generate_spiral_trajectory"]

import warnings
import numpy as np

from numpy.typing import NDArray

from mrinufft import voronoi
from mrinufft.trajectories.maths import Rz, Rx
from mrinufft.trajectories.utils import initialize_tilt
from pulpy.grad import spiral_varden


def generate_spiral_trajectory(
    ndim: int, npix: tuple, ncontrasts: int, nintl: int = 55, res: float = 1.125
) -> NDArray:
    """
    Generate a 3D spiral trajectory for MR fingerprinting.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (2 or 3).
    npix : int
        Size of the imaging matrix (e.g., 128 for a ``ndim * [128]`` grid).
    ncontrasts : int
        Number of contrasts in the image series.
    nintl : int, optional
        Number of interleaves to fully sample a sprial plane.
        The default is ``55``.
    res : float, optional
        Spatial resolution in ``mm``.
        The default is ``1.125`` mm.

    Returns
    -------
    k : NDArray
        Array of k-space trajectory points ``(ncontrasts, nintl, npts, ndim)``.
    dcf : NDArray
        Array of density compensation factors ``(ncontrasts, nintl, npts)``.
    t : NDArray
        Array of sampling times along readout ``(npts,)``.

    """
    fov = npix * res / 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, k, t, _, _ = spiral_varden(
            fov=fov,
            res=res / 10.0,
            gts=4e-6,
            gslew=120 * 100,
            gamp=3.3,
            densamp=0,
            dentrans=0,
            nl=nintl,
        )

    # add z axis
    k = np.pad(k, ((0, 0), (0, 1))).astype(np.float32)

    # normalize trajectory between [-0.5, 0.5)
    kabs = (k**2).sum(axis=-1) ** 0.5
    k = k / kabs.max() / 2

    if ndim == 2:
        # get spiral for each contrast by rotating base interleave
        golden_angles = initialize_tilt("golden", ncontrasts) * np.arange(ncontrasts)
        rotmat = [Rz(theta) for theta in golden_angles]
        rotmat = np.stack(rotmat).astype(np.float32)
        k = np.einsum("aij,bj->abi", rotmat, k)

        # compute dcf
        dcf = voronoi(k[..., :-1]).reshape(*k.shape[:-1]).astype(np.float32)

        return k[..., None, :-1], dcf[:, None, :], t

    # compute full spiral
    equispaced_angles = initialize_tilt("uniform", nintl) * np.arange(nintl)
    rotmat = [Rz(theta) for theta in equispaced_angles]
    rotmat = np.stack(rotmat).astype(np.float32)
    k = np.einsum("aij,bj->abi", rotmat, k)

    # compute dcf
    dcf = voronoi(k[..., :-1]).reshape(*k.shape[:-1]).astype(np.float32)

    # rotate about x axis
    golden_angles = initialize_tilt("golden", ncontrasts) * np.arange(ncontrasts)
    rotmat = [Rx(phi) for phi in golden_angles]
    rotmat = np.stack(rotmat).astype(np.float32)
    k = np.einsum("cij,abj->cabi", rotmat, k)

    # get projection along x (i.e., y and z components)
    krad = k[..., 1:]

    # calculate radial dcf component
    dcf_rad = (krad**2).sum(axis=-1) ** 0.5

    # correct dcf
    dcf = dcf_rad * dcf

    return k, dcf, t
