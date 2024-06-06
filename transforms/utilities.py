import jax.numpy as jnp

from transforms import library_of_transforms as lft
from jax import config
from jax.lax import dot_general
config.update("jax_enable_x64", True)

def get_analysis(polynomials, parameters, N_points, M_keep):
    '''
    polynomials = [poly_name1, poly_name2, ...]
    parameters = [[a1, b1], [a2, b2], ...]
    N_points = [n1, n2, ...] number of points used to approximate coefficients
    M_keep = [m1, m2, ...] number of modes to keep
    return = [A1, A2, ...] matrices with shapes [(m1, n1), (m2, n2), ...]
    '''
    a_operators = []
    for poly_name, param, n, m in zip(polynomials, parameters, N_points, M_keep):
        A = lft.poly_data[poly_name]["transform"](n, param, k=m)['analysis']
        a_operators.append(A)
    return a_operators

def get_synthesis(polynomials, parameters, grids, M_keep):
    '''
    polynomials = [poly_name1, poly_name2, ...]
    parameters = [[a1, b1], [a2, b2], ...]
    grids = [grid1, grid2, ...] grids (jax arrays) used for interpolation, if some grid is K (int), the standard grid with K points is used
    M_keep = [m1, m2, ...] number of modes keept
    return = [S1, S2, ...] matrices with shapes [(len(grid1), m1), (len(grid2), m2), ...] if gridK is K (int), the output is (K, mk)
    '''
    s_operators = []
    for poly_name, param, grid, m in zip(polynomials, parameters, grids, M_keep):
        grid_ = None if type(grid) == int else grid
        N_ = grid if type(grid) == int else len(grid)
        if m > N_ and (type(grid) == int):
            grid_ = lft.poly_data[poly_name]["nodes"](N_, param)
        S = lft.poly_data[poly_name]["transform"](max(N_, m), param, grid=grid_, k=m)['synthesis']
        s_operators.append(S)
    return s_operators

def get_operators(transformation_type, **kwargs):
    '''
    transformation_type either 'analysis' or 'synthesis'
    wrapper for get_analysis and get_synthesis
    '''
    if transformation_type == 'analysis':
        output = get_analysis(kwargs['polynomials'], kwargs['parameters'], kwargs['N_points'], kwargs['M_keep'])
    else:
        output = get_synthesis(kwargs['polynomials'], kwargs['parameters'], kwargs['grids'], kwargs['M_keep'])
    return output

def apply_operators(data, operators, batch=False):
    '''
    data.shape = [n_features, x1, x2, ...] or [batch, n_features, x1, x2, ...] depending on batch
    operator = [M1, M2, ...]
    '''
    shift = 2 if batch else 1
    for i, O in enumerate(operators):
        n = i + shift
        transposition = [*range(1, n+1)] + [0] + [*range(n+1, len(data.shape))]
        if (data.dtype == jnp.complex128) or (O.dtype == jnp.complex128):
            data = data + 0j
            O = O + 0j
        data = jnp.transpose(dot_general(O, data, (((1,), (n)), ((), ()))), transposition)
    return data

def transform_data(data, polynomials, parameters, transformation_type, grids):
    '''
    data.shape = [batch, n_features, x1, x2, ...]
    polynomials = [poly_name1, poly_name2, ...]
    transformation_type either "analysis" or "synthesis"
    grids = [grid1, grid2, ...] if gridK is None, the outpu grid is a standard one with the number of points the same as for the input
    '''
    grids_ = [data.shape[2+i] if g is None else g for i, g in enumerate(grids)]
    N_points_ = data.shape[2:]
    O = get_operators(transformation_type, parameters=parameters, polynomials=polynomials, grids=grids_, M_keep=N_points_, N_points=N_points_)
    output = apply_operators(data, O, batch=True)
    if transformation_type == "synthesis":
        output = jnp.real(output)
    return output
