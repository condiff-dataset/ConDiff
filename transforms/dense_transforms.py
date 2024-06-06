import numpy as np
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)

def get_analysis_data(N, Gauss, Polynomial, grid=None):
    x, w = Gauss(N) # parameters of quadrature
    P = []
    Q = []
    D_inv = []

    for n in range(N):
        p = Polynomial(n, x) # evaluate polynomials
        d = np.sum(w * p**2) # compute scalar product
        P.append(p)
        D_inv.append(1/d)
        if not (grid is None):
            q = Polynomial(n, grid)
            Q.append(q)

    A = np.vstack(P).T
    D_inv = np.diag(D_inv)

    analysis_data = [A, D_inv, w, x]
    if not (grid is None):
        analysis_data.append(np.vstack(Q).T)
    return analysis_data

def build_matrices(k, analysis_data):
    An = (analysis_data[1] @ analysis_data[0].T * analysis_data[2])[:k, :]
    Sn = analysis_data[0][:, :k] if len(analysis_data) == 4 else analysis_data[-1][:, :k]
    return jnp.array(An), jnp.array(Sn)

def get_transform_data(N, k, Gauss, Polynomial, grid=None):
    analysis_data = get_analysis_data(N, Gauss, Polynomial, grid=grid)
    transforms = build_matrices(k, analysis_data)
    transform_data = {
        "weights": jnp.array(analysis_data[2]),
        "points": jnp.array(analysis_data[3]),
        "analysis": transforms[0],
        "synthesis": transforms[1]
    }
    return transform_data
