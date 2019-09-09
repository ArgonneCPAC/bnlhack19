"""
"""
import numpy as np
from scipy.special import erf


def quenched_fraction(vmax, z, **kwargs):
    """
    Parameters
    ----------
    vmax : ndarray
        Vmax [physical km/s] for a halo of the input mass at the input redshift

    redshift : float or ndarray

    Returns
    -------
    fq : ndarray
    """
    vmax, z = _get_1d_arrays(vmax, z)

    qmin_arr = _get_qmin(z, **kwargs)

    vq_arr = 10.**_get_log10_vq(z, **kwargs)
    sigma_vq_arr = _get_sigma_vq(z, **kwargs)

    erf_arg = np.log10(vmax/vq_arr)/(np.sqrt(2.)*sigma_vq_arr)
    erf_term = 0.5 + 0.5*(erf(erf_arg))

    return qmin_arr + (1. - qmin_arr)*erf_term


def _get_qmin(z, qmin_0=-1.944, qmin_a=-2.419, **kwargs):
    a = 1./(1. + z)
    qmin = qmin_0 + qmin_a*(1.-a)
    qmin = np.where(qmin < 0, 0., qmin)
    qmin = np.where(qmin > 1, 1., qmin)
    return qmin


def _get_log10_vq(z, vq_0=2.248, vq_a=-0.018, vq_z=0.124, **kwargs):
    a = 1./(1. + z)
    return vq_0 + vq_a*(1.-a) + vq_z*z


def _get_sigma_vq(z, sigma_vq_0=0.227, sigma_vq_a=0.037, sigma_vq_lnz=-0.107, **kwargs):
    a = 1./(1. + z)
    return sigma_vq_0 + sigma_vq_a*(1.-a) + sigma_vq_lnz*np.log(1.+z)


def _get_1d_arrays(*args):
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
