"""
See Section 3.4.

This generates Figure 6.

The `lanczos` function in this file is part of an unpublished software in https://github.com/spdes,
the license of which are not issued yet. Do not make a copy or distribute the implementation before the license is
announced.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import math
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from jax.config import config
from functools import partial
from typing import Tuple

config.update("jax_enable_x64", True)


def lanczos(a: jnp.ndarray, v0: jnp.ndarray, m: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Lanczos algorithm.

    .. math::

        A = V \, T \, V^T

    Parameters
    ----------
    a : JArray (n, n)
        A symmetric matrix of size n. This function does not deal with complex matrices.
    v0 : JArray (n, )
        An arbitrary vector with Euclidean norm 1, for example, v0 = [1, 0, 0, ...].
    m : int
        Number of Lanczos iterations (must >= 1 and <= n).

    Returns
    -------
    JArray (n, m), JArray (m, ), JArray (m - 1, )
        V, the diagonal of T, and the 1-off diagonal of T.

    References
    ----------
    Gene H. Golub and Charles F. Van Load. Matrix computations. 2013. The Johns Hopkins University Press, 4th edition.

    https://en.wikipedia.org/wiki/Lanczos_algorithm.

    Notes
    -----
    Let (\lambda, u) be a pair of eigenvalue and eigenvector of T, then you can approximately use (\lambda, V u) as
    the eigenvalue and eigenvector of A. When m = n, this is precise.
    """

    def scan_body(carry, _):
        _v, w = carry

        beta = jnp.sqrt(jnp.sum(w ** 2))
        v = w / beta
        wp = a @ v
        alpha = jnp.dot(wp, v)
        w = wp - alpha * v - beta * _v

        return (v, w), (v, alpha, beta)

    wp0 = a @ v0
    alpha0 = jnp.dot(wp0, v0)
    w0 = wp0 - alpha0 * v0

    _, (vs, alphas, betas) = jax.lax.scan(scan_body, (v0, w0), jnp.arange(m - 1))
    return jnp.vstack([v0, vs]).T, jnp.hstack([alpha0, alphas]), betas


ell, sigma = 2., 1.


@partial(jax.vmap, in_axes=[0, None])
@partial(jax.vmap, in_axes=[None, 0])
def cov_func(t, s):
    """Matern 3/2 covariance function.
    """
    p = math.sqrt(3) * jnp.abs(t - s) / ell
    return sigma ** 2 * (1 + p) * jnp.exp(-p)


def cholesky_draw(ts, key):
    """Draw a sample using Cholesky decomposition.
    """
    cov = cov_func(ts, ts)
    return jnp.linalg.cholesky(cov) @ jax.random.normal(key, ts.shape)


def lanczos_draw(ts, e1, num_iteration, key):
    """Draw a sample using Lanczos approximation.

    Notes
    -----
    In principle, the eigenvalues and eigenvectors should be computed efficiently from the tridiagonal matrix. But
    JAX does not support this feature yet (written January 2022).
    """
    cov = cov_func(ts, ts)

    vs, alphas, betas = lanczos(cov, jnp.asarray(e1), num_iteration)
    z = jnp.diag(alphas) + jnp.diag(betas, k=-1) + jnp.diag(betas, k=1)

    eigenvectors, eigenvalues = jax.lax.linalg.eigh(z)
    eigenvectors = vs @ eigenvectors

    rnds = jax.random.normal(key, (num_iteration,))
    return jnp.einsum('i,...i,i->...', jnp.sqrt(eigenvalues), eigenvectors, rnds)


# Random seed
key = jax.random.PRNGKey(666)

# Times
T = 100
ts = jnp.linspace(0., 10., T)

# Number of samples, number of Lanczos iterations
num_samples = 10
num_lanczos_iter = 20

e1 = jax.random.normal(key, ts.shape)
e1 = e1 / jnp.linalg.norm(e1, ord=2)

key, _ = jax.random.split(key)

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.serif': ["Computer Modern Roman"],
    'font.size': 16})

fig, axes = plt.subplots(nrows=1, ncols=2, sharey='row', figsize=(10, 5))

for i in range(num_samples):
    key, _ = jax.random.split(key)
    sample = cholesky_draw(ts, key)
    axes[0].plot(ts, sample, label=f'Sample {i}')
    axes[0].set_title('True samples.')

for i in range(num_samples):
    key, _ = jax.random.split(key)
    sample = lanczos_draw(ts, e1, num_lanczos_iter, key)
    axes[1].plot(ts, sample, label=f'Sample {i}')
    axes[1].set_title('Approximate samples via Lánczos.')

for ax in axes:
    ax.grid(linestyle='--', alpha=0.3, which='both')
    ax.set_xlabel('$t$')

plt.tight_layout(pad=0.1)
plt.savefig('../latex/figs/lanczos-demo.pdf')
plt.show()
