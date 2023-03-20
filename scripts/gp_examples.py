"""
Generate samples from a variety of GPs

This gives Figures 2 to 5.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Sequence


def cov_affine(ts, ss, params: Sequence[float]):
    """Covariance function of the affine GP.
    """
    ma, mb, va, vb = params
    return va * ts[:, None] * ss[None, :] + vb


def cov_fBM(ts, ss, r):
    """Covariance function of factional Brownian motion.

    When r = 0.5, this reduces to that of a standard Brownian motion.
    """
    p = 2 * r
    return 0.5 * (np.abs(ts)[:, None] ** p + np.abs(ss)[None, :] ** p - np.abs(ts[:, None] - ss[None, :]) ** p)


def cov_m12(ts, ss, ell, sigma):
    """Matern 1/2 (exponential) covariance function.
    """
    return sigma ** 2 * np.exp(-np.abs(ts[:, None] - ss[None, :]) / ell)


def cov_m32(ts, ss, ell, sigma):
    """Matern 3/2 covariance function.
    """
    p = math.sqrt(3) * np.abs(ts[:, None] - ss[None, :]) / ell
    return sigma ** 2 * (1 + p) * np.exp(-p)


def cov_rbf(ts, ss, ell, sigma):
    """Radial basis covariance function.
    """
    p = (ts[:, None] - ss[None, :]) / ell
    return sigma ** 2 * np.exp(-0.5 * p ** 2)


def cov_cos(ts, ss):
    """Cosine covariance function.
    """
    return np.cos(ss[:, None] - ts[None, :])


# Times
T = 500
ts = np.linspace(0., 5., T)
grids = np.meshgrid(ts, ts, indexing='ij')

# Random seed and number of samples
np.random.seed(888)
num_samples = 10

# Pseudo diagonal jitter for improving the condition number of covariance matrix
eps = 1e-9

# Set plotting parameters
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.serif': ["Computer Modern Roman"],
    'font.size': 16})


# Plot (fractional) Brownian motion with r = 0.5 and 0.9
def plot_fBM():
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey='row')

    for ind, r in zip([0, 1],
                      [0.5, 0.9]):
        cov_matrix = cov_fBM(ts, ts, r) + np.eye(T) * eps

        # Plot samples
        for _ in range(num_samples):
            sample = np.linalg.cholesky(cov_matrix) @ np.random.randn(T)
            axes[0, ind].plot(ts, sample)

        # Plot .95 region
        axes[0, ind].fill_between(ts,
                                  -1.96 * np.sqrt(np.diag(cov_matrix)),
                                  1.96 * np.sqrt(np.diag(cov_matrix)),
                                  color='black',
                                  edgecolor='none',
                                  alpha=0.15,
                                  label='Marginal 0.95 interval')

        axes[0, ind].grid(linestyle='--', alpha=0.3, which='both')
        axes[0, ind].set_title(f'$r={r}$')

        # Plot cov matrix
        axes[1, ind].contourf(*grids, cov_matrix, levels=15, cmap=plt.cm.Blues_r)

        # You can enable the colourbar by uncommenting this block, but it looks ugly IMO.
        # divider = make_axes_locatable(axes[1, ind])
        # cax = divider.append_axes('right', '3%', pad='2%')
        # plt.colorbar(pcm, cax=cax)

    axes[1, 0].set_xlabel('$t$')
    axes[1, 1].set_xlabel('$t$')

    axes[0, 0].set_ylabel('$X(t)$')
    axes[1, 0].set_ylabel('$s$')

    plt.tight_layout(pad=0.1)
    plt.savefig('../latex/figs/sample-cov-fbm.pdf')
    plt.show()


# Plot Matern GPs
def plot_matern():
    fig, axes = plt.subplots(figsize=(10, 5), nrows=2, ncols=4, sharex=True, sharey='row')

    # Plot Matern 1/2 (exponential)
    ell, sigma = 1., 1.
    cov_matrix = cov_m12(ts, ts, ell, sigma)

    for _ in range(num_samples):
        sample = np.linalg.cholesky(cov_matrix) @ np.random.randn(T)
        axes[0, 0].plot(ts, sample)

    # Plot cov matrix
    axes[1, 0].contourf(*grids, cov_matrix, levels=10, cmap=plt.cm.Blues_r)

    # Plot .95 region
    axes[0, 0].fill_between(ts,
                            -1.96 * np.sqrt(np.diag(cov_matrix)),
                            1.96 * np.sqrt(np.diag(cov_matrix)),
                            color='black',
                            edgecolor='none',
                            alpha=0.15,
                            label='Marginal 0.95 interval')

    axes[0, 0].grid(linestyle='--', alpha=0.3, which='both')
    axes[0, 0].set_title(rf'$\ell={ell}, \sigma={sigma}$')
    axes[1, 0].set_xlabel('$t$')
    axes[1, 0].set_xticks([1., 2., 3., 4.])

    # Plot Matern 3/2 with different ells and sigmas
    for ind, (ell, sigma) in zip([1, 2, 3],
                                 [(1., 1.), (0.3, 1.), (1., 0.5)]):
        cov_matrix = cov_m32(ts, ts, ell, sigma)

        # Plot samples
        for _ in range(num_samples):
            sample = np.linalg.cholesky(cov_matrix) @ np.random.randn(T)
            axes[0, ind].plot(ts, sample)

        # Plot cov matrix
        axes[1, ind].contourf(*grids, cov_matrix, levels=10, cmap=plt.cm.Blues_r)

        # Plot .95 region
        axes[0, ind].fill_between(ts,
                                  -1.96 * np.sqrt(np.diag(cov_matrix)),
                                  1.96 * np.sqrt(np.diag(cov_matrix)),
                                  color='black',
                                  edgecolor='none',
                                  alpha=0.15,
                                  label='Marginal 0.95 interval')

        axes[0, ind].grid(linestyle='--', alpha=0.3, which='both')
        axes[0, ind].set_title(rf'$\ell={ell}, \sigma={sigma}$')
        axes[1, ind].set_xlabel('$t$')
        axes[1, ind].set_xticks([1., 2., 3., 4.])

    axes[0, 0].set_ylabel('$X(t)$')
    axes[1, 0].set_ylabel('$s$')

    plt.tight_layout(pad=0.1)
    plt.savefig('../latex/figs/sample-cov-matern.pdf')
    plt.show()


# Plot RBF
def plot_rbf():
    plt.figure(figsize=(8, 5))

    ell, sigma = 1., 1.
    cov_matrix = cov_rbf(ts, ts, ell, sigma) + np.eye(T) * eps

    # Plot samples
    for _ in range(num_samples):
        sample = np.linalg.cholesky(cov_matrix) @ np.random.randn(T)
        plt.plot(ts, sample)

    # Plot .95 region
    plt.fill_between(ts,
                     -1.96 * np.sqrt(np.diag(cov_matrix)),
                     1.96 * np.sqrt(np.diag(cov_matrix)),
                     color='black',
                     edgecolor='none',
                     alpha=0.15,
                     label='Marginal 0.95 interval')

    plt.grid(linestyle='--', alpha=0.3, which='both')
    plt.xlabel('$t$')
    plt.xlabel('$t$')

    plt.ylabel('$X(t)$')

    plt.tight_layout(pad=0.1)
    plt.savefig('../latex/figs/sample-rbf.pdf')
    plt.show()


# Plot sinusoidal GPs
def plot_cos():
    T = 500
    ts = np.linspace(0., 10., T)
    grids = np.meshgrid(ts, ts, indexing='ij')

    fig, axes = plt.subplots(figsize=(12, 5), nrows=1, ncols=2, sharex=True)

    cov_matrix = cov_cos(ts, ts) + np.eye(T) * eps

    # Plot samples
    for _ in range(num_samples):
        a, b = np.random.randn(2)
        sample = a * np.sin(ts) + b * np.cos(ts)
        axes[0].plot(ts, sample)

    # Plot cov matrix
    axes[1].contourf(*grids, cov_matrix, levels=15, cmap=plt.cm.Blues_r)

    # Plot .95 region
    axes[0].fill_between(ts,
                         -1.96 * np.sqrt(np.diag(cov_matrix)),
                         1.96 * np.sqrt(np.diag(cov_matrix)),
                         color='black',
                         edgecolor='none',
                         alpha=0.15,
                         label='Marginal 0.95 interval')

    axes[0].grid(linestyle='--', alpha=0.3, which='both')

    axes[0].set_xlabel('$t$')
    axes[1].set_xlabel('$t$')

    axes[0].set_ylabel('$X(t)$')
    axes[1].set_ylabel('$s$')

    plt.tight_layout(pad=0.1)
    plt.savefig('../latex/figs/sample-cov-cos.pdf')
    plt.show()


plot_fBM()
plot_matern()
plot_rbf()
plot_cos()
