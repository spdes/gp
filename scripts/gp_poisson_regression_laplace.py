"""
Approximate GP regression with a non-linear Poisson likelihood. See Example 24.

This generates Figure 11.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import jax
import jax.numpy as jnp
import jax.scipy.stats
import jaxopt
import matplotlib.pyplot as plt
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)

ell, sigma = 1., 1.


@partial(jax.vmap, in_axes=[0, None])
@partial(jax.vmap, in_axes=[None, 0])
def cov_func(t, s):
    """Exponential covariance function.
    """
    return sigma ** 2 * jnp.exp(-jnp.abs(t - s) / ell)


def lam(x):
    """The lambda parameter of the Poisson likelihood.
    """
    return 2 * (jnp.tanh(x) + 1)


# Times
T = 100
ts = jnp.linspace(0., 5., T)

# Mean vector and covariance matrix
ms = jnp.zeros((T,))
cov = cov_func(ts, ts)

chol = jax.scipy.linalg.cho_factor(cov)
logdet = jnp.log(jnp.linalg.det(cov))

# Generate a trajectory from the GP
key = jax.random.PRNGKey(666)
sample = jnp.linalg.cholesky(cov) @ jax.random.normal(key, (T,))

# Generate measurements/data
key, _ = jax.random.split(key)
ys = jax.random.poisson(key, lam(sample))


def log_pdf_y_cond_x(xs):
    """Log conditional PDF of the Poisson likelihood model.
    """
    return jnp.sum(jax.scipy.stats.poisson.logpmf(ys, lam(xs)))


@jax.jit
def map_log_likelihood(xs):
    """MAP objective function, viz., log (p(y_{1:T} | x_{1:T}) p(x_{1:T}))
    """
    return 0.5 * (jnp.dot(xs, jax.scipy.linalg.cho_solve(chol, xs)) + logdet) - log_pdf_y_cond_x(xs)


# Solve the MAP problem
init_xs = jnp.zeros((T,))
solver = jaxopt.ScipyMinimize(method='L-BFGS-B', fun=map_log_likelihood)
opt_result = solver.run(init_xs)
print(f'Convergence: {opt_result.state}')

# Obtain the Laplace mean and covariance
map_estimates = opt_result.params
laplace_cov_estimates = jnp.linalg.inv(jax.hessian(map_log_likelihood)(map_estimates))

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.serif': ["Computer Modern Roman"],
    'font.size': 16})

plt.plot(ts, sample, c='tab:blue', linestyle='--', linewidth=2, label='True $X_{1:T}$')
plt.scatter(ts, ys, s=1, c='tab:purple', edgecolors=None, alpha=0.6, label='Measurements $Y_{1:T}$')
plt.plot(ts, map_estimates, c='black', linewidth=2, label='Laplace mean')
plt.fill_between(ts,
                 map_estimates - 1.96 * jnp.sqrt(jnp.diag(laplace_cov_estimates)),
                 map_estimates + 1.96 * jnp.sqrt(jnp.diag(laplace_cov_estimates)),
                 color='black',
                 edgecolor='none',
                 alpha=0.15,
                 label='Laplace 0.95 interval')

plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlabel('$t$')
plt.legend(fontsize=15)

plt.tight_layout(pad=0.1)
plt.savefig('../latex/figs/poisson-gp.pdf')
plt.show()
