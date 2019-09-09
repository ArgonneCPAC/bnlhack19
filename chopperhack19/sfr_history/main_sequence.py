"""Analytical model in UniverseMachine for the relation between
SFR and Vmax for main-sequence galaxies
"""
import numpy as np


def mean_sfr(vmax, z, **kwargs):
    """
    Parameters
    ----------
    vmax : ndarray
        Vmax [physical km/s] for a halo of the input mass at the input redshift

    redshift : float or ndarray

    Returns
    -------
    mean_sfr : ndarray
    """
    vmax, z = _get_1d_arrays(vmax, z)

    V = 10.**logV(z, **kwargs)
    v = vmax/V
    term1 = 1./(v**alpha(z, **kwargs) + v**beta(z, **kwargs))

    exp_arg = (-np.log10(v)**2)/(2.*delta_sfr(z, **kwargs))
    term2 = (10.**logGamma(z, **kwargs))*np.exp(exp_arg)

    return (10.**logEpsilon(z, **kwargs))*(term1 + term2)


def logV(z, logV_0=2.151, logV_a=-1.658, logV_lnz=1.68, logV_z=-0.233, **kwargs):
    a = 1./(1. + z)
    log10_V = logV_0 + logV_a*(1. - a) + logV_lnz*np.log(1.+z) + logV_z*z
    return log10_V


def logEpsilon(z, epsilon_0=0.109, epsilon_a=-3.441, epsilon_lnz=5.079, epsilon_z=-0.781, **kwargs):
    a = 1./(1. + z)
    log10_epsilon = epsilon_0 + epsilon_a*(1.-a) + epsilon_lnz*np.log(1.+z) + epsilon_z*z
    return log10_epsilon


def alpha(z, alpha_0=-5.598, alpha_a=-20.731, alpha_lnz=13.455, alpha_z=-1.321, **kwargs):
    a = 1./(1. + z)
    return alpha_0 + alpha_a*(1.-a) + alpha_lnz*np.log(1.+z) + alpha_z*z


def beta(z, beta_0=-1.911, beta_a=0.395, beta_z=0.747, **kwargs):
    a = 1./(1. + z)
    return beta_0 + beta_a*(1.-a) + beta_z*z


def logGamma(z, gamma_0=-1.699, gamma_a=4.206, gamma_z=-0.809, **kwargs):
    a = 1./(1. + z)
    return gamma_0 + gamma_a*(1.-a) + gamma_z*z


def delta_sfr(z, delta_0=0.055, **kwargs):
    return delta_0 + np.zeros_like(z)


def _get_1d_arrays(*args):
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
