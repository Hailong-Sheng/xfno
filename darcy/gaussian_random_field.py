import numpy as np
from scipy.fftpack import idctn
from numpy.random import normal

def grf(alpha, tau, s):
    # Random variables in KL expansion
    xi = normal(0, 1, (s,s))
    # xi = np.ones([s,s])

    # Define the (square root of) eigenvalues of the covariance operator
    k1, k2 = np.meshgrid(np.arange(s), np.arange(s))
    coef = tau**(alpha-1) * (np.pi**2 * (k1**2 + k2**2) + tau**2)**(-alpha/2)

    # Construct the KL coefficients
    l = s * coef * xi
    l[0,0] = 0

    # Perform inverse discrete cosine transform
    u = idctn(l, type=2, norm='ortho', axes=(0,1))

    return u