"""
GP parameter estimation by MLE. See Section 5.

This generates Figure 10.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import math
import jax
import jax.numpy as jnp
import jax.scipy
import jax.scipy.optimize
import jaxopt
import matplotlib.pyplot as plt
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)


def m(t, w):
    """Mean function.
    """
    return w * jnp.sin(math.pi * t)


@partial(jax.vmap, in_axes=[0, None, None, None])
@partial(jax.vmap, in_axes=[None, 0, None, None])
def cov_m32(t, s, ell, sigma):
    """Matern 3/2 covariance function
    """
    p = math.sqrt(3) * jnp.abs(t - s) / ell
    return sigma ** 2 * (1 + p) * jnp.exp(-p)


# Times
T = 500
ts = jnp.linspace(0., 5., T)

# True model parameters
true_w, true_ell, true_sigma = 2., 1., 1.

# GP covariance matrix
cov = cov_m32(ts, ts, true_ell, true_sigma)

# Generate trajectory
key = jax.random.PRNGKey(666)
xs = m(ts, true_w) + jnp.linalg.cholesky(cov) @ jax.random.normal(key, (T,))

# Generate measurements
xi = 1.
key, _ = jax.random.split(key)
ys = xs + math.sqrt(xi) * jax.random.normal(key, (T,))


def g(theta):
    """Positive bijection. Softplus.
    """
    return jnp.log(1. + jnp.exp(theta))


def log_det(chol_factor_of_K, c: float = math.sqrt(2 * math.pi)) -> float:
    """Log determinant of a positive definite matrix using Cholesky.
    log |c^2 K| = log |c L L' c| = log |c L| + log |c L'| = 2 * sum log diag(c L)
    """
    diag = jnp.diag(chol_factor_of_K[0])  # note that cho_factor can give random data
    return 2 * jnp.sum(jnp.log(jnp.abs(c * diag)))


@jax.jit
def log_marginal_likelihood(theta):
    w, ell, sigma = g(theta)

    ms = m(ts, w)
    G = cov_m32(ts, ts, ell, sigma) + xi * jnp.eye(T)

    chol = jax.scipy.linalg.cho_factor(G)
    residual = ys - ms
    return 0.5 * (jnp.dot(residual, jax.scipy.linalg.cho_solve(chol, residual)) + log_det(chol))


key, _ = jax.random.split(key)
init_theta = jax.random.normal(key, (3,))
solver = jaxopt.ScipyMinimize(method='L-BFGS-B', fun=log_marginal_likelihood)
opt_result = solver.run(init_theta)
estimated_w, estimated_ell, estimated_sigma = g(opt_result.params)

print(f'Estimated w: {estimated_w}, ell: {estimated_ell}, and sigma: {estimated_sigma}')
print(opt_result.state)

# Regression with the estimated parameters
ms, cov = m(ts, estimated_w), cov_m32(ts, ts, estimated_ell, estimated_sigma)
G = cov + xi * jnp.eye(T)
chol = jax.scipy.linalg.cho_factor(G)

posterior_mean = ms + cov @ jax.scipy.linalg.cho_solve(chol, ys - ms)
posterior_cov = cov - cov @ jax.scipy.linalg.cho_solve(chol, cov)

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
plt.legend(framealpha=0.1, fontsize=14)

plt.tight_layout(pad=0.1)
plt.savefig('../latex/figs/param-mle.pdf')
plt.show()
