"""
See Introduction.

This generates Figure 1 in the lecture note.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)


# See Equation 1
def x(t, a, b):
    return a * t + b


def cov_func(ts, ss, params):
    """Covariance function of the affine GP, see Example 2.
    """
    ma, mb, va, vb = params
    return va * ts[:, None] * ss[None, :] + vb


# Number of samples
num_samples = 10

# Means and variances of a and b
mean_a, mean_b = 1., 2.
variance_a, variance_b = 1., 3.

# Times
ts = np.linspace(0, 5, 100)

# Set plotting parameters
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.serif': ["Computer Modern Roman"],
    'font.size': 20})

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

# Draw samples
for i in range(num_samples):
    a, b = np.array([mean_a, mean_b]) + np.sqrt([variance_a, variance_b]) * np.random.randn(2)
    xs = x(ts, a, b)
    axes[0].plot(ts, xs, linewidth=1.5)

# Plot 0.95 region
axes[0].fill_between(ts,
                     mean_a * ts + mean_b - 1.96 * np.sqrt(ts ** 2 * variance_a + variance_b),
                     mean_a * ts + mean_b + 1.96 * np.sqrt(ts ** 2 * variance_a + variance_b),
                     color='black',
                     edgecolor='none',
                     alpha=0.15,
                     label='Marginal 0.95 interval')

axes[0].legend(loc='upper left', fontsize=21)
axes[0].set_xlabel('$t$')
axes[0].set_ylabel('$X(t)$')
axes[0].grid(linestyle='--', alpha=0.3, which='both')

# Plot covariance matrix
cov_matrix = cov_func(ts, ts, [1., 1., 2., 3.])
grids = np.meshgrid(ts, ts, indexing='ij')

pcm = axes[1].contourf(*grids, cov_matrix, levels=12, cmap=plt.cm.Blues_r)

axes[1].set_xlabel('$t$')
axes[1].set_ylabel('$s$')

fig.colorbar(pcm, ax=axes[1], pad=0.02)

plt.tight_layout(pad=0.1)
plt.savefig('../latex/figs/example-gp-affine-sample-cov.pdf')
plt.show()
