"""
"""
from math import exp, log, log10
from numba import njit

__all__ = ('mean_sfr_vs_vmax_redshift', )


@njit
def mean_sfr_vs_vmax_redshift(vmax, redshift, result,
            logV_0=2.151, logV_a=-1.658, logV_lnz=1.68, logV_z=-0.233,
            alpha_0=-5.598, alpha_a=-20.731, alpha_lnz=13.455, alpha_z=-1.321,
            beta_0=-1.911, beta_a=0.395, beta_z=0.747,
            gamma_0=-1.699, gamma_a=4.206, gamma_z=-0.809, delta_0=0.055,
            epsilon_0=0.109, epsilon_a=-3.441, epsilon_lnz=5.079, epsilon_z=-0.781):
    """
    """
    n = result.size
    for i in range(n):
        z = redshift[i]
        a = 1/(1 + z)

        V = 10**(logV_0 + logV_a*(1 - a) + logV_lnz*log(1 + z) + logV_z*z)
        v = vmax[i]/V

        _a = alpha_0 + alpha_a*(1-a) + alpha_lnz*log(1+z) + alpha_z*z
        _b = beta_0 + beta_a*(1-a) + beta_z*z
        term1 = 1/(v**_a + v**_b)

        _log10v = log10(v)
        exp_arg = (-_log10v*_log10v)/(2*delta_0)

        _logGamma = gamma_0 + gamma_a*(1-a) + gamma_z*z
        term2 = (10**_logGamma)*exp(exp_arg)

        log10_epsilon = epsilon_0 + epsilon_a*(1.-a) + epsilon_lnz*log(1+z) + epsilon_z*z

        result[i] = (10**log10_epsilon)*(term1 + term2)
