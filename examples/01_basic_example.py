"""Basic example."""

import matplotlib.pyplot as plt

from pygrog.benchmark import generate_mrf_case
from pygrog.linop import SubspaceNUFFT

# generate data
data, k, dcf, grog_training, obj_coeff, basis = generate_mrf_case(2, brainweb=True)

# generate operation
ishape = (8, 4, 200, 200)
A = SubspaceNUFFT(ishape, k, basis.T)

# recon
img = A.H(dcf * data)
