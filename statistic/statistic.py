#
# Copyright(c) 2022 Daniel Knüttel
#

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
statistic.py
************

Module containing several statistic functions.

``autocovariance``
    Computes an improved autocovariance function of data.
``tauint``
    Computes an estimate of the integrated auto correlation time. 
    Uses ``autocovariance``.
``jackknife_std``
    Jackknife standard error estimator using a performance enhanced
    re-sampling method. Takes a function of the mean.
``jackknife2_std``
    Jackknife standard error estimator using a performance-worse method 
    compared to ``jackknife_std``. Takes a function of the samples itself.
``jackknife_cov``, ``jackknife2_cov``
    Jackknife covariance estimators. See ``jackknife_std``, ``jackknife2_std``.
``make_bins``
    Make bins.
"""


import numpy as np

def autocovariance(data, t):
    mu_before = np.average(data[:-t])
    mu_after = np.average(data[t:])
    
    return np.average((data[:-t] - mu_before) * (data[t:] - mu_after))
def tauint(data):
    var = np.var(data)
    n = np.size(data)
    result = 0.5
    i = 1
    while(i < n):
        autof = autocovariance(data, i)
        if(autof < 0):
            break
        i += 1
        result += autof / var * (1 - i/n)
    return result

def jackknife_std(corrf, statistic, *params, data_axis=1, statistic_axis=0, eps=None):
    N = corrf.shape[data_axis]
    xbar = np.average(corrf, axis=data_axis)
    fmean = statistic(xbar, *params)
    
    if(data_axis == 0):
        def slc(i):
            return i
    else:
        def slc(i):
            lst = [slice(None, None, None)]*data_axis
            return tuple(lst + [i])

    if(eps is None):
        jackknife_samples = np.array([
            statistic((N*xbar - corrf[slc(i)]) / (N - 1), *params) - fmean for i in range(N)
        ])
        
        return np.sqrt(np.sum(jackknife_samples**2, axis=statistic_axis) * (N - 1) / N)
    
    jackknife_samples = np.array([
        statistic(xbar + eps*(xbar - corrf[slc(i)]) / (N - 1), *params) - fmean for i in range(N)
    ])
    
    return np.sqrt(np.sum(jackknife_samples**2, axis=statistic_axis) * (N - 1) / N) / eps


def jackknife2_std(data, statistic, *params, data_axis=1, statistic_axis=0, masked_ok=True):
    """
    This is the full-featured standard deviation error compared to
    ``jackknife_std``. Unlike ``jackknife_std`` this function passes full
    samples instead of means to the ``statistic``.

    This function is slower but can be used for more complex statistics.

    See eqs. (1.4), (1.5)::
        @article{10.1214/aos/1176345462,
        author = {B. Efron and C. Stein},
        title = {{The Jackknife Estimate of Variance}},
        volume = {9},
        journal = {The Annals of Statistics},
        number = {3},
        publisher = {Institute of Mathematical Statistics},
        pages = {586 -- 596},
        keywords = {$U$ statistics, ANOVA decomposition, bootstrap, jackknife, variance estimation},
        year = {1981},
        doi = {10.1214/aos/1176345462},
        URL = {https://doi.org/10.1214/aos/1176345462}
        }
    """

    N = data.shape[data_axis]
    if(data_axis == 0):
        def slc(i):
            return i
    else:
        def slc(i):
            lst = [slice(None, None, None)]*data_axis
            return tuple(lst + [i])

    masked = np.ma.array(data, mask=False)

    # FIXME: This is terribly slow. 
    # Can we improve the performance by parallelism?
    # Are there other fancy ways to improve performance?
    jackknife_samples = []
    for i in range(N):
        masked.mask[slc(i)] = True
        if(masked_ok):
            jackknife_samples.append(statistic(masked, *params))
        else:
            jackknife_samples.append(statistic(np.array(masked), *params))
        masked.mask[slc(i)] = False
    jackknife_samples = np.array(jackknife_samples)

    jackmean = np.mean(jackknife_samples, axis=statistic_axis)

    return np.sqrt(np.sum((jackknife_samples - jackmean)**2, axis=statistic_axis) * (N - 1) / N)

def jackknife_cov(corrf, statistic, *params, data_axis=1, statistic_axis=0, cov_axis=0, eps=None):
    N = corrf.shape[data_axis]
    xbar = np.average(corrf, axis=data_axis)
    fmean = statistic(xbar, *params)
    
    if(data_axis == 0):
        def slc(i):
            return i
    else:
        def slc(i):
            lst = [slice(None, None, None)]*data_axis
            return tuple(lst + [i])

    if (eps is None):
        jackknife_samples = np.array([
            statistic((N*xbar - corrf[slc(i)]) / (N - 1), *params) - fmean for i in range(N)
        ])

        
        def slc(i):
            lst = [slice(None, None, None)]*(1 + cov_axis)
            return tuple(lst + [i])
        jackknife_cov_samples = np.array([[
            jackknife_samples[slc(i)] * jackknife_samples[slc(k)] for i in range(jackknife_samples.shape[1 + cov_axis])]
            for k in range(jackknife_samples.shape[1 + cov_axis])
        ])

        return np.sum(jackknife_cov_samples, axis=statistic_axis + 2) * (N - 1) / N 

    jackknife_samples = np.array([
        statistic((xbar + eps*(xbar - corrf[slc(i)])) / (N - 1), *params) - fmean for i in range(N)
    ])

    
    def slc(i):
        lst = [slice(None, None, None)]*(1 + cov_axis)
        return tuple(lst + [i])
    jackknife_cov_samples = np.array([[
        jackknife_samples[slc(i)] * jackknife_samples[slc(k)] for i in range(jackknife_samples.shape[1 + cov_axis])]
        for k in range(jackknife_samples.shape[1 + cov_axis])
    ])

    return np.sum(jackknife_cov_samples, axis=statistic_axis + 2) * (N - 1) / N / eps**2


def jackknife2_cov(corrf, statistic, *params, data_axis=1, statistic_axis=0, cov_axis=0, masked_ok=True):
    N = corrf.shape[data_axis]
    xbar = np.average(corrf, axis=data_axis)
    fmean = statistic(xbar, *params)
    
    if(data_axis == 0):
        def slc(i):
            return i
    else:
        def slc(i):
            lst = [slice(None, None, None)]*data_axis
            return tuple(lst + [i])

    masked = np.ma.array(data, mask=False)

    # FIXME: This is terribly slow. 
    # Can we improve the performance by parallelism?
    # Are there other fancy ways to improve performance?
    jackknife_samples = []
    for i in range(N):
        masked.mask[slc(i)] = True
        if(masked_ok):
            jackknife_samples.append(statistic(masked, *params))
        else:
            jackknife_samples.append(statistic(np.array(masked), *params))
        masked.mask[slc(i)] = False
    jackknife_samples = np.array(jackknife_samples)

    
    def slc(i):
        lst = [slice(None, None, None)]*(1 + cov_axis)
        return tuple(lst + [i])
    jackknife_cov_samples = np.array([[
        jackknife_samples[slc(i)] * jackknife_samples[slc(k)] for i in range(jackknife_samples.shape[1 + cov_axis])]
        for k in range(jackknife_samples.shape[1 + cov_axis])
    ])

    return np.sum(jackknife_cov_samples, axis=statistic_axis + 2) * (N - 1) / N 

def make_bins(data, nbins):
    bin_axis = len(data.shape) - 1
    total_items = data.shape[bin_axis]
    binsize = total_items // nbins
    if(total_items % nbins):
        raise ValueError(f"number of items({total_items}) must be multiple of nbins({nbins})")
    other_dimensions = data.shape[:-1]
    
    
    return np.mean(data.reshape(*other_dimensions, -1, binsize), axis=bin_axis + 1)
