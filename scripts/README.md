# Companion codes for the lecture note on Gaussian process

To use the codes, please install the dependencies in `requirements.txt` in your favourite Python environment. For
example, `pip install -r requirements.txt`.

The scripts `gp_examples.py`, `lanczos.gp`, `parameter_estimation_mle.py`, and `gp_poisson_regression_laplace.py` 
are written in JAX. To use them, please install `jax` and `jaxopt` by `pip install -r requirements_additional.txt`.

# Files

The Python codes in `./scripts` are as follows.

1. `./bw_kkt_expansion.py`: Kosambi--Karhunen--Loève expansion of Brownian motion. Related to Section 3.5.
2. `./circulant_embedding.py`: Sampling from stationary GP using circulant embedding. Related to Section 3.6.
3. `./affine-gp.py`: Generate samples from an affine GP from definition. Related to Figure 1 and Equation 1.
4. `./gp_examples.py`: A gallery of a bunch of example GPs. Related to Section 2.1.
5. `./gp_poisson_regression_laplace.py`: Laplace approximation to a GP regression model with Poisson likelihood.
   Related to Example 24.
6. `./lanczos_gp.py`: Approximate GP sampling by Lánczos iteration. Related to Section 3.4.
7. `./parameter_estimation_mle.py`: Estimate parameters by maximum likelihood. Related to Section 5.
8. `./regression.py`: Standard GP regression. Related to Section 4.

# Note

Most plotting scripts here use LaTeX to print math symbols in figure, hence, you need a LaTeX distribution installed,
e.g., Texlive. If you don't want to install LaTeX, please disable this feature by toggling `'text.usetex': True`
to `False` in the used scripts.
