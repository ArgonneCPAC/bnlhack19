"""Analytical model in UniverseMachine for the relation between
SFR and Vmax for main-sequence galaxies
"""
from math import exp, log, log10
from numba import njit

__all__ = ('mean_sfr_vs_vmax_redshift', )


@njit
def mean_sfr_vs_vmax_redshift(vmax, redshift, result, n,
            logV_0=2.151, logV_a=-1.658, logV_lnz=1.68, logV_z=-0.233,
            alpha_0=-5.598, alpha_a=-20.731, alpha_lnz=13.455, alpha_z=-1.321,
            beta_0=-1.911, beta_a=0.395, beta_z=0.747,
            gamma_0=-1.699, gamma_a=4.206, gamma_z=-0.809, delta_0=0.055,
            epsilon_0=0.109, epsilon_a=-3.441, epsilon_lnz=5.079, epsilon_z=-0.781):
    """Average star formation rate for main-sequence galaxies

    Parameters
    ----------
    vmax : ndarray
        Vmax [physical km/s] for a halo of the input mass at the input redshift

    redshift : ndarray

    n : int
        shape of vmax and redshift are (n, )

    Returns
    -------
    mean_sfr : ndarray of shape (n, )
    """
    for i in range(n):
        z = redshift[i]
        V = 10**_logV(z, logV_0, logV_a, logV_lnz, logV_z)
        v = vmax[i]/V
        _a = _alpha(z, alpha_0, alpha_a, alpha_lnz, alpha_z)
        _b = _beta(z, beta_0, beta_a, beta_z)
        term1 = 1/(v**_a + v**_b)

        _log10v = log10(v)
        exp_arg = (-_log10v*_log10v)/(2*_delta_sfr(z, delta_0))

        term2 = (10**_logGamma(z, gamma_0, gamma_a, gamma_z))*exp(exp_arg)
        result[i] = (10**_logEpsilon(z, epsilon_0, epsilon_a, epsilon_lnz, epsilon_z))*(term1 + term2)


@njit
def _logV(z, logV_0, logV_a, logV_lnz, logV_z):
    a = 1/(1 + z)
    log10_V = logV_0 + logV_a*(1 - a) + logV_lnz*log(1 + z) + logV_z*z
    return log10_V


@njit
def _logEpsilon(z, epsilon_0, epsilon_a, epsilon_lnz, epsilon_z):
    a = 1/(1 + z)
    log10_epsilon = epsilon_0 + epsilon_a*(1.-a) + epsilon_lnz*log(1+z) + epsilon_z*z
    return log10_epsilon


@njit
def _alpha(z, alpha_0, alpha_a, alpha_lnz, alpha_z):
    a = 1/(1 + z)
    return alpha_0 + alpha_a*(1-a) + alpha_lnz*log(1+z) + alpha_z*z


@njit
def _beta(z, beta_0, beta_a, beta_z):
    a = 1/(1 + z)
    return beta_0 + beta_a*(1-a) + beta_z*z


@njit
def _logGamma(z, gamma_0, gamma_a, gamma_z):
    a = 1/(1 + z)
    return gamma_0 + gamma_a*(1-a) + gamma_z*z


@njit
def _delta_sfr(z, delta_0):
    return delta_0
