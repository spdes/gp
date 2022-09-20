"""
GP regression. See Section 4.

This generates Figure 9.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt

ell, sigma = 1., 1.
a, b = 0.2, 1.


def m(t):
    """Mean function.
    """
    return np.sin(2 * t)


def cov_m32(ts, ss):
    """Matern 3/2 covariance function
    """
    p = math.sqrt(3) * np.abs(ts[:, None] - ss[None, :]) / ell
    return sigma ** 2 * (1 + p) * np.exp(-p)


# Times
T = 1000
ts = np.linspace(0., 5., T)

# GP mean vector and covariance matrix
ms = m(ts)
cov = cov_m32(ts, ts)

# Generate trajectory
np.random.seed(666)
xs = ms + np.linalg.cholesky(cov) @ np.random.randn(T)

# Generate measurements
xi = 1.
ys = a * xs + b + math.sqrt(xi) * np.random.randn(T)

# Regression, see Equation (12)
G = a ** 2 * cov + xi * np.eye(T)
chol = scipy.linalg.cho_factor(G)

posterior_mean = ms + a * cov @ scipy.linalg.cho_solve(chol, ys - (a * ms + b))
posterior_cov = cov - a ** 2 * cov @ scipy.linalg.cho_solve(chol, cov)

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
                 posterior_mean - 1.96 * np.sqrt(np.diag(posterior_cov)),
                 posterior_mean + 1.96 * np.sqrt(np.diag(posterior_cov)),
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
