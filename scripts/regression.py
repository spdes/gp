"""
GP regression. See Section 4.

This generates Figure 9.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import math
import jax
import jax.numpy as jnp
import jax.scipy
import matplotlib.pyplot as plt
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)

ell, sigma = 1., 1.
a, b = 0.2, 1.


def m(t):
    """Mean function.
    """
    return jnp.sin(2 * t)


@partial(jax.vmap, in_axes=[0, None])
@partial(jax.vmap, in_axes=[None, 0])
def cov_m32(t, s):
    """Matern 3/2 covariance function
    """
    p = math.sqrt(3) * jnp.abs(t - s) / ell
    return sigma ** 2 * (1 + p) * jnp.exp(-p)


# Times
T = 1000
ts = jnp.linspace(0., 5., T)

# GP mean vector and covariance matrix
ms = m(ts)
cov = cov_m32(ts, ts)

# Generate trajectory
key = jax.random.PRNGKey(999)
xs = ms + jnp.linalg.cholesky(cov) @ jax.random.normal(key, (T,))

# Generate measurements
xi = 1.
key, _ = jax.random.split(key)
ys = a * xs + b + math.sqrt(xi) * jax.random.normal(key, (T,))

# Regression, see Equation (12)
G = a ** 2 * cov + xi * jnp.eye(T)
chol = jax.scipy.linalg.cho_factor(G)

posterior_mean = ms + a * cov @ jax.scipy.linalg.cho_solve(chol, ys - (a * ms + b))
posterior_cov = cov - a ** 2 * cov @ jax.scipy.linalg.cho_solve(chol, cov)

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.serif': ["Computer Modern Roman"],
    'font.size': 16})

plt.plot(ts, xs, c='tab:blue', linestyle='--', linewidth=2, label='True $X_{1:T}$')
plt.scatter(ts, ys, s=1, c='tab:purple', edgecolors=None, alpha=0.3, label='Measurements $Y_{1:T}$')
plt.plot(ts, posterior_mean, c='black', linewidth=2, label='Posterior mean')
plt.fill_between(ts,
                 posterior_mean - 1.96 * jnp.sqrt(jnp.diag(posterior_cov)),
                 posterior_mean + 1.96 * jnp.sqrt(jnp.diag(posterior_cov)),
                 color='black',
                 edgecolor='none',
                 alpha=0.15,
                 label='Marginal posterior 0.95 interval')
plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlabel('$t$')
plt.legend(framealpha=0.1)

plt.tight_layout(pad=0.1)
plt.savefig('../latex/figs/regression.pdf')
plt.show()
