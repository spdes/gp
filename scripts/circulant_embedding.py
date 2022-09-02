"""
Stationary GP sampling by using circulant embedding.

This generates Figure 8.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

np.random.seed(666)

ell, sigma = 1., 1.


def cov_func(t, s):
    """Exponential covariance function. Under this covariance function, the minimal circulant extension is positive
    definite.
    """
    return sigma ** 2 * np.exp(-np.abs(t - s) / ell)


def circulant_gp_sampling(cs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """GP sampling by circulant embedding.

    Parameters
    ----------
    cs : np.ndarray (T, )
        The first column of the covariance matrix. T > 1.

    Returns
    -------
    np.ndarray (T, ), np.ndarray (T, )
        Two independent GP samples.

    Notes
    -----
    In theory :code:`np.fft.ifft(cs_ext)` should return real numbers.
    """
    T = cs.shape[0]
    cs_ext = np.concatenate([cs, cs[-2:0:-1]])
    n = 2 * T - 2

    ds = n * np.fft.ifft(cs_ext)
    alphas, betas = np.random.randn(2, n)
    xis = alphas + betas * 1.j
    samples = (np.fft.fft(np.sqrt(ds) * xis) / math.sqrt(n))[:T]
    return np.real(samples), np.imag(samples)


# Times
T = 1000
ts = np.linspace(0., 1., T)

# First column
cs = cov_func(ts, ts[0])

# Set plotting parameters
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.serif': ["Computer Modern Roman"],
    'font.size': 18})

for i in range(3):
    sample1, sample2 = circulant_gp_sampling(cs)
    plt.plot(ts, sample1)
    plt.plot(ts, sample2)

plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.grid(linestyle='--', alpha=0.3, which='both')

plt.tight_layout(pad=0.1)
plt.savefig('../latex/figs/circulant-gp-sample.pdf')
plt.show()
