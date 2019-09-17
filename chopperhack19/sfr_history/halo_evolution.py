"""Starting from a halo mass at z=0, the two functions below give descriptions for
how halo mass and Vmax smoothly evolve across time.
"""
from numba import njit
from math import log10, exp


__all__ = ('halo_mass_vs_redshift', 'vmax_vs_mhalo_and_redshift')


@njit
def halo_mass_vs_redshift(halo_mass_at_z0, redshift, halo_mass_at_z):
    """Fitting function from Behroozi+13, https://arxiv.org/abs/1207.6105,
    Equations (H2)-(H6).

    Parameters
    ----------
    halo_mass_at_z0 : float or ndarray
        Mass of the halo at z=0 assuming h=0.7

    redshift : float or ndarray

    halo_mass_at_z : ndarray
        Empty array that will be filled with halo mass at the input redshift
    """
    n = halo_mass_at_z.size

    for i in range(n):
        m_z0 = halo_mass_at_z0[i]
        z = redshift[i]
        a = 1./(1. + z)

        _M13_z0 = 10**13.276

        _M13_zfactor1 = (1. + z)**3.0
        _M13_zfactor2 = (1. + 0.5*z)**-6.11
        _M13_zfactor3 = exp(-0.503*z)
        _M13 = _M13_z0*_M13_zfactor1*_M13_zfactor2*_M13_zfactor3

        _exparg_factor1 = log10(m_z0/_M13_z0)

        logarg = ((10**9.649)/m_z0)**0.18
        a0 = 0.205 - log10(logarg + 1.)

        _factor2_num = 1. + exp(-4.651*(1-a0))
        _factor2_denom = 1. + exp(-4.651*(a-a0))
        _exparg_factor2 = _factor2_num/_factor2_denom
        _exparg = _exparg_factor1*_exparg_factor2

        halo_mass_at_z[i] = _M13*(10**_exparg)


@njit
def vmax_vs_mhalo_and_redshift(mhalo, redshift, vmax):
    """Scaling relation between Vmax and Mhalo for host halos across redshift.

    Relation taken from Equation (E2) from Behroozi+19,
    https://arxiv.org/abs/1806.07893.

    Parameters
    ----------
    mhalo : float or ndarray
        Mass of the halo at the input redshift assuming h=0.7

    redshift : float or ndarray

    vmax : ndarray
        Empty array that will be filled with Vmax [physical km/s]
        for a halo of the input mass and redshift
    """
    n = vmax.size
    for i in range(n):
        m = mhalo[i]
        z = redshift[i]
        a = 1/(1 + z)

        denom_term1 = (a/0.378)**-0.142
        denom_term2 = (a/0.378)**-1.79
        mpivot = 1.64e12/(denom_term1 + denom_term2)
        vmax[i] = 200*(m/mpivot)**(1/3.)
