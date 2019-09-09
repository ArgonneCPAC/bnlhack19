"""Starting from a halo mass at z=0, the two functions below give descriptions for
how halo mass and Vmax smoothly evolve across time.
"""
import numpy as np


def halo_mass_vs_redshift(halo_mass_at_z0, redshift):
    """Fitting function from Behroozi+13, https://arxiv.org/abs/1207.6105,
    Equations (H2)-(H6). Calibration assumes h=0.7.

    Parameters
    ----------
    halo_mass_at_z0 : float or ndarray
        Mass of the halo at z=0 assuming h=0.7.

    redshift : float or ndarray

    Returns
    -------
    halo_mass : ndarray
        Mass of the halo at the input redshift assuming h=0.7.
    """
    M0, z = _get_1d_arrays(halo_mass_at_z0, redshift)
    return _M13(z)*10**_f(M0, z)


def vmax_vs_mhalo_and_redshift(mhalo, redshift):
    """Scaling relation between Vmax and Mhalo for host halos across redshift.

    Relation taken from Equation (E2) from Behroozi+19,
    https://arxiv.org/abs/1806.07893.

    Parameters
    ----------
    mhalo : float or ndarray
        Mass of the halo at the input redshift assuming h=0.7.

    redshift : float or ndarray

    Returns
    -------
    vmax : ndarray
        Vmax [physical km/s] for a halo of the input mass at the input redshift
    """
    mhalo, z = _get_1d_arrays(mhalo, redshift)
    a = 1./(1. + z)

    denom_term1 = (a/0.378)**-0.142
    denom_term2 = (a/0.378)**-1.79
    numerator = 1.64e12
    mpivot = numerator/(denom_term1 + denom_term2)
    return 200.*(mhalo/mpivot)**(1/3.)


def _M13(z):
    factor1 = 10**13.276
    factor2 = (1. + z)**3.0
    factor3 = (1. + 0.5*z)**-6.11
    factor4 = np.exp(-0.503*z)
    return factor1*factor2*factor3*factor4


def _f(M0, z):
    factor1 = np.log10(M0/_M13(0.))
    factor2_num = _g(M0, 1.)
    factor2_denom = _g(M0, 1./(1.+z))
    factor2 = factor2_num/factor2_denom
    return factor1*factor2


def _g(M0, a):
    a0 = _get_a0(M0)
    return 1. + np.exp(-4.651*(a-a0))


def _get_a0(M0):
    logarg = ((10**9.649)/M0)**0.18
    return 0.205 - np.log10(logarg + 1.)


def _get_1d_arrays(*args):
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
