"""
"""
from math import exp, log, log10
from numba import njit

__all__ = ('mean_sfr_vs_mpeak_at_z0_and_redshift', )


@njit
def mean_sfr_vs_mpeak_at_z0_and_redshift(halo_mpeak_at_z0, redshift, result,
            logV_0=2.151, logV_a=-1.658, logV_lnz=1.68, logV_z=-0.233,
            alpha_0=-5.598, alpha_a=-20.731, alpha_lnz=13.455, alpha_z=-1.321,
            beta_0=-1.911, beta_a=0.395, beta_z=0.747,
            gamma_0=-1.699, gamma_a=4.206, gamma_z=-0.809, delta_0=0.055,
            epsilon_0=0.109, epsilon_a=-3.441, epsilon_lnz=5.079, epsilon_z=-0.781):
    """
    """
    n = result.size
    for i in range(n):
        mp_z0 = halo_mpeak_at_z0[i]
        z = redshift[i]
        a = 1/(1 + z)

        ####################################
        #  Calculate halo mass at redshift z
        _M13_z0 = 10**13.276

        _M13_zfactor1 = (1. + z)**3.0
        _M13_zfactor2 = (1. + 0.5*z)**-6.11
        _M13_zfactor3 = exp(-0.503*z)
        _M13 = _M13_z0*_M13_zfactor1*_M13_zfactor2*_M13_zfactor3

        _exparg_factor1 = log10(mp_z0/_M13_z0)

        logarg = ((10**9.649)/mp_z0)**0.18
        a0 = 0.205 - log10(logarg + 1.)

        _factor2_num = 1. + exp(-4.651*(1-a0))
        _factor2_denom = 1. + exp(-4.651*(a-a0))
        _exparg_factor2 = _factor2_num/_factor2_denom
        _exparg = _exparg_factor1*_exparg_factor2

        halo_mass = _M13*(10**_exparg)

        ####################################
        #  Calculate vmax at redshift z
        denom_term1 = (a/0.378)**-0.142
        denom_term2 = (a/0.378)**-1.79
        mpivot = 1.64e12/(denom_term1 + denom_term2)
        vmax = 200*(halo_mass/mpivot)**(1/3.)

        ####################################
        #  Calculate sfr at redshift z
        V = 10**(logV_0 + logV_a*(1 - a) + logV_lnz*log(1 + z) + logV_z*z)
        v = vmax/V

        _a = alpha_0 + alpha_a*(1-a) + alpha_lnz*log(1+z) + alpha_z*z
        _b = beta_0 + beta_a*(1-a) + beta_z*z
        term1 = 1/(v**_a + v**_b)

        _log10v = log10(v)
        exp_arg = (-_log10v*_log10v)/(2*delta_0)

        _logGamma = gamma_0 + gamma_a*(1-a) + gamma_z*z
        term2 = (10**_logGamma)*exp(exp_arg)

        log10_epsilon = epsilon_0 + epsilon_a*(1.-a) + epsilon_lnz*log(1+z) + epsilon_z*z

        sfr = (10**log10_epsilon)*(term1 + term2)
        result[i] = sfr



