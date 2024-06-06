import scipy.special as scp
from transforms.dense_transforms import get_transform_data
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

def Gegenbauer(N, k, alpha, grid=None):
    Gauss = lambda n: scp.roots_gegenbauer(n, alpha)
    Polynomial = lambda x, n: scp.eval_gegenbauer(x, alpha, n)
    return get_transform_data(N, k, Gauss, Polynomial, grid=grid)

def Legendre(N, k, grid=None):
    Gauss = scp.roots_legendre
    Polynomial = scp.eval_legendre
    return get_transform_data(N, k, Gauss, Polynomial, grid=grid)

def Chebyshev_1(N, k, grid=None):
    Gauss = scp.roots_chebyt
    Polynomial = scp.eval_chebyt
    return get_transform_data(N, k, Gauss, Polynomial, grid=grid)

def Chebyshev_2(N, k, grid=None):
    Gauss = scp.roots_chebyu
    Polynomial = scp.eval_chebyu
    return get_transform_data(N, k, Gauss, Polynomial, grid=grid)

def Jacobi(N, k, alpha, beta, grid=None):
    Gauss = lambda n: scp.roots_jacobi(n, alpha, beta)
    Polynomial = lambda x, n: scp.eval_jacobi(x, alpha, beta, n)
    return get_transform_data(N, k, Gauss, Polynomial, grid=grid)

def Fourier(N, k, grid=None):
    transform_data = {
        "weights": jnp.ones(N)/N,
        "points": jnp.linspace(0, 1, N+1)[:-1],
        "analysis": jnp.fft.fft(jnp.diag(jnp.ones(N))[:k, :]),
        "synthesis": None if grid is None else jnp.fft.ifft(jnp.diag(jnp.ones(N))[:, :k])
    }
    if not (grid is None):
        freq = 2*jnp.pi*k*jnp.fft.fftfreq(k)
        synthesis = jnp.exp(jnp.outer(grid, freq*1j)) / k
        transform_data["synthesis"] = synthesis
    return transform_data

def Real_Fourier(N, k, grid=None):
    k2 = k // 2 # sin
    k1 = k - k2 # cos
    transform_data = {
        "weights": jnp.ones(N)/N,
        "points": jnp.linspace(0, 1, N+1)[:-1],
    }
    x = transform_data["points"] if grid is None else grid
    y = x.reshape(-1, 1)
    S = [jnp.cos(2*jnp.pi*k*y) for k in range(k2)] + [jnp.sin(2*jnp.pi*k*y) for k in range(1, k1+1)]
    abs_w = jnp.array([abs(k) for k in range(k2)] + [abs(k) for k in range(1, k1+1)])
    ord = jnp.argsort(abs_w)
    S = jnp.hstack(S)[:, ord]
    h = 1 / N
    D = jnp.array([h,] + [2*h]*(k-1)).reshape(-1, 1)
    A = D * S.T
    transform_data["analysis"] = A
    transform_data["synthesis"] = S
    return transform_data

poly_data = {
    "Gegenbauer": {
        "nodes": lambda n, alpha: scp.roots_gegenbauer(n, alpha[0])[0],
        "interval": [-1, 1],
        "transform": lambda n, alpha, grid=None, k=None: Gegenbauer(n, n if k is None else k, alpha[0], grid=grid)
    },
    "Legendre": {
        "nodes": lambda n, alpha: scp.roots_legendre(n)[0],
        "interval": [-1, 1],
        "transform": lambda n, alpha, grid=None, k=None: Legendre(n, n if k is None else k, grid=grid)
    },
    "Chebyshev_u": {
        "nodes": lambda n, alpha: scp.roots_chebyu(n)[0],
        "interval": [-1, 1],
        "transform": lambda n, alpha, grid=None, k=None: Chebyshev_2(n, n if k is None else k, grid=grid)
    },
    "Chebyshev_t": {
        "nodes": lambda n, alpha: scp.roots_chebyt(n)[0],
        "interval": [-1, 1],
        "transform": lambda n, alpha, grid=None, k=None: Chebyshev_1(n, n if k is None else k, grid=grid)
    },
    "Jacobi": {
        "nodes": lambda n, alpha:  scp.roots_jacobi(n, alpha[0], alpha[1])[0],
        "interval": [-1, 1],
        "transform": lambda n, alpha, grid=None, k=None: Jacobi(n, n if k is None else k, alpha[0], alpha[1], grid=grid)
    },
    "Fourier": {
        "nodes": lambda n, alpha: jnp.linspace(0, 1, n+1)[:-1],
        "interval": [0, 1],
        "transform": lambda n, alpha, grid=None, k=None: Fourier(n, n if k is None else k, grid=grid)
    },
    "Real_Fourier": {
        "nodes": lambda n, alpha: jnp.linspace(0, 1, n+1)[:-1],
        "interval": [0, 1],
        "transform": lambda n, alpha, grid=None, k=None: Real_Fourier(n, n if k is None else k, grid=grid)
    }
}
