"""
"""

import numpy as np

import mrtwin
import torchsim

from .._svd import estimate_bloch_subspace

from .schedule import vfa_schedule
from .trajectory import generate_spiral_trajectory

def generate_simple_case():
    
    # generate schedule
    TR = 8.0
    flip = vfa_schedule()
    
    # generate basis
    t1grid, t2grid = np.mgrid[5:3500:10, 5:350:10]
    training_data = torchsim.mrf_sim(flip, TR, t1grid.ravel(), t2grid.ravel())
    basis = estimate_bloch_subspace(training_data, num_coeff=4)
    
    # generate trajectory
    k, dcf, _ = generate_spiral_trajectory(
        ndim=2, 
        npix=200,
        ncontrasts=len(flip)
        )
    
    # generate object
    obj = mrtwin.brainweb_phantom(2, subject=4, shape=200, output_res=1.125)
    
    # generate sensitivity maps
    smaps = mrtwin.sensmap((8, 200, 200))
    
    # generate signals
    signals = torchsim.mrf_sim(flip, TR, obj.T1, obj.T2, M0=obj.M0)
    signals_coeff = signals @ basis.to(signals.dtype)
    
    