"""Functions calculating a weighted histogram.

A Gaussian PDF is dropped onto each data point; the function adds up the
total probability that each Gaussian contributes to each bin.

numpy_gw_hist is based on vectorized Numpy. numba_gw_hist is based on Numba.

"""
import numpy as np
from scipy.stats import norm
from math import erf as math_erf
from math import sqrt as math_sqrt
from numba import njit
from numba import cuda


__all__ = ('numpy_gw_hist', 'numba_gw_hist', 'cuda_gw_hist')


def numpy_gw_hist(data, bins, scale):
    """Calculate a Gaussian-weighted histogram.

    Drop a gaussian kernel on top of each data point.
    For every point, calculate the probability that the point lies in each bin,
    by evaluating the CDF of the Gaussian associated with each point.
    Sum the results across all bins and return the result.

    Converges to an ordinary histogram in the limit of large data where
    scale << binsize.

    Parameters
    ----------
    data : ndarray of shape (ndata, )

    bins : ndarray of shape (nbins, )

    scale : float or ndarray of shape (ndata, )

    Returns
    -------
    khist : ndarray of shape (nbins-1, )
    """
    data = np.atleast_1d(data)
    bins = np.atleast_1d(bins)
    nbins, ndata = bins.size, data.size

    scale = np.zeros(ndata) + scale

    logsm_bin_matrix = np.repeat(
        bins, ndata).reshape((nbins, ndata)).astype('f4')
    data_matrix = np.tile(data, nbins).reshape((nbins, ndata)).astype('f4')
    smoothing_kernel_matrix = np.tile(
        scale, nbins).reshape((nbins, ndata)).astype('f4')

    cdf_matrix = norm.cdf(
        logsm_bin_matrix, loc=data_matrix, scale=smoothing_kernel_matrix)

    prob_bin_member = np.diff(cdf_matrix, axis=0)  # Shape (nbins-1, ndata)

    total_num_bin_members = np.sum(
        prob_bin_member, axis=1)  # Shape (nbins-1, )

    return total_num_bin_members


@njit
def numba_gw_hist(data, bins, scale, khist):
    """
    Parameters
    ----------
    data : ndarray of shape (ndata, )

    bins : ndarray of shape (nbins, )

    scale : float or ndarray of shape (ndata, )

    khist : ndarray of shape (nbins-1, )
        Empty array used to store the result
    """
    ndata = len(data)
    nbins = len(bins)
    bot = bins[0]
    sqrt2 = math_sqrt(2)

    for i in range(ndata):
        x = data[i]

        z = (x - bot)/scale/sqrt2
        last_cdf = 0.5*(1.+math_erf(z))
        for j in range(1, nbins):
            bin_edge = bins[j]
            z = (x - bin_edge)/scale/sqrt2
            new_cdf = 0.5*(1.+math_erf(z))
            weight = last_cdf - new_cdf
            khist[j-1] += weight
            last_cdf = new_cdf


@cuda.jit
def cuda_gw_hist(data, bins, scale, gw_hist_out):
    """Increment weighted bin counts in gw_hist_out, given an array of bins
    Parameters
    ----------
    data: ndarray of shape (ndata, )

    bins: ndarray of shape (nbins, )

    scale: ndarray of shape (ndata, )

    gw_hist_out: ndarray of shape (nbins -1, )
         empty array to store result
    """
    # find where this job goes over
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # define some useful things
    bot = bins[0]
    sqrt2 = math_sqrt(2.)

    # loop over the data set - each thread now looks at one data point.
    for i in range(start, data.shape[0], stride):
        z = (data[i] - bot)/scale[i]/sqrt2
        last_cdf = 0.5*(1.+math_erf(z))
        # for each bin, calculate weight and add it in
        for j in range(1, bins.shape[0]):
            bin_edge = bins[j]
            z = (data[i] - bin_edge)/scale[i]/sqrt2
            new_cdf = 0.5*(1.+math_erf(z))
            weight = last_cdf - new_cdf
            # atomic add to bin to avoid race conditions
            cuda.atomic.add(gw_hist_out, j-1, weight)
            last_cdf = new_cdf
