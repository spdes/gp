"""
Kosambi--Karhunen--Loève expansion of Brownian motion.

This generates Figure 7.

Zheng Zhao 2022
zheng.zhao@it.uu.se
"""
import math
import numpy as np
import matplotlib.pyplot as plt


def psi(i: int, ts: np.ndarray) -> np.ndarray:
    """Basis function.

    Parameters
    ----------
    i : int
        Index of basis.
    ts : np.ndarray (...)
        Times.

    Returns
    -------
    np.ndarray (...)
        The basis function evaluation at the times.
    """
    return math.sqrt(2) * np.sin((i - 0.5) * math.pi * ts)


def bw(ts: np.ndarray, xis: np.ndarray) -> np.ndarray:
    """Brownian motion path.

    Parameters
    ----------
    ts: np.ndarray (T, )
        Times.
    xis : np.ndarray (N, )
        N independent standard Normal random variables.

    Returns
    -------
    np.ndarray (T, )
        A truncated Brownian motion path evaluated at the times.
    """
    N = xis.shape[0]

    vs = np.array([1 / ((i - 0.5) * math.pi) ** 2 for i in range(1, N + 1)])
    phis = np.vstack([psi(i, ts) for i in range(1, N + 1)])

    return np.einsum('i,i,i...->...', np.sqrt(vs), xis, phis)


# Times
T = 1000
ts = np.linspace(0., 1., T)

# Expansion orders
Ns = [20, 100, 500]

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.serif': ["Computer Modern Roman"],
    'font.size': 16})

fig, axes = plt.subplots(nrows=1, ncols=3, sharey='row', figsize=(14, 4))

for idx in range(3):
    np.random.seed(666)
    xis = np.random.randn(Ns[idx])

    axes[idx].plot(ts, bw(ts, xis))
    axes[idx].set_xlabel('$t$')
    axes[idx].set_title(f'$N = {Ns[idx]}$')
    axes[idx].grid(linestyle='--', alpha=0.3, which='both')

axes[0].set_ylabel('$B(t)$')

plt.tight_layout(pad=0.1)
plt.savefig('../latex/figs/kkl-bw.pdf')
plt.show()
