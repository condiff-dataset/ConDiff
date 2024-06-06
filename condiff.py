import jax
import numpy as np
from scipy.sparse import spdiags, linalg as splinalg

import os
import h5py
import parafields

def diffusion(n_samples, grid, bounds, cov_model, var, rng, key):
    """
    Diffisuon equation with specificed covariance function `cov_model`, variance value `var` 
    and grid size `grid`. Generate `n_samples` realizations.
    """
    k_ls, rhs_ls, x_ls = [], [], []
    while len(k_ls) != n_samples:                                 # Bound contrast interval to help differentiate complexity
        key, subkey = jax.random.split(key)
        field = parafields.generate_field(cells=[grid+1, grid+1],
                                          covariance=cov_model,
                                          variance=var,
                                          seed=subkey[0].item())
        k = field.evaluate()
        contrast = np.exp(k.max() - k.min())
        if not bounds[0] <= contrast <= bounds[1]:
            continue
        
        coef = np.exp(k)
        A = fd_mtx2(coef)
        b = rng.normal(0, 1, grid*grid)
        x = splinalg.spsolve(A, b)
        
        k_ls.append(coef)
        rhs_ls.append(b)
        x_ls.append(x)
    
    k_ls = np.stack(k_ls, axis=0)
    rhs_ls = np.stack(rhs_ls, axis=0)
    x_ls = np.stack(x_ls, axis=0)
    return k_ls, rhs_ls, x_ls, key

def multiple_diffusion(save_dir, N_train, N_test, cov_model_ls, grid_ls, boundaries_ls, var_ls, seed_global):
    """
    Generate train and test subsets for diffusion equation for each covariance
    model from `cov_model_ls`, grid size from `grid_ls` and variance values
    from `var_ls`.
    """
    rng = np.random.default_rng(seed=seed_global)
    key = jax.random.PRNGKey(seed_global)
    for cov_model in cov_model_ls:
        for g, bound_ls in zip(grid_ls, boundaries_ls):
            for var_i, b_ in zip(var_ls, bound_ls):
                run_name = cov_model+str(var_i)+'_grid'+str(g)
                run_dir = os.path.join(save_dir, run_name)
                os.mkdir(run_dir)
                os.chdir(run_dir)

                k_train, rhs_train, x_train, key = diffusion(N_train, g, bounds=b_, cov_model=cov_model,
                                                             var=var_i, rng=rng, key=key)
                k_test, rhs_test, x_test, key = diffusion(N_test, g, bounds=b_, cov_model=cov_model,
                                                          var=var_i, rng=rng, key=key)

                # Save train
                hf_train = h5py.File(run_name+'_train.h5', 'w')
                hf_train.create_dataset('k', data=k_train)
                hf_train.create_dataset('rhs', data=rhs_train)
                hf_train.create_dataset('x', data=x_train)
                hf_train.close()

                # Save test
                hf_test = h5py.File(run_name+'_test.h5', 'w')
                hf_test.create_dataset('k', data=k_test)
                hf_test.create_dataset('rhs', data=rhs_test)
                hf_test.create_dataset('x', data=x_test)
                hf_test.close()
    return 

def poisson(n_samples, grid, rng):
    """
    Poisson equation with specificed grid size `grid`.
    Generate `n_samples` realizations.
    """
    rhs_ls, x_ls = [], []
    for _ in range(n_samples):
        A = fd_mtx2(np.ones([grid+1, grid+1], dtype='float64'))
        b = rng.normal(0, 1, grid*grid)
        x = splinalg.spsolve(A, b)
        
        rhs_ls.append(b)
        x_ls.append(x)    
    rhs_ls = np.stack(rhs_ls, axis=0)
    x_ls = np.stack(x_ls, axis=0)
    return rhs_ls, x_ls

def multiple_poisson(save_dir, N_train, N_test, grid_ls, seed_rng):
    """
    Generate train and test subsets for Poisson equation for each grid size
    from `grid_ls`.
    """
    rng = np.random.default_rng(seed=seed_rng)
    for g in grid_ls:
        run_name = 'poisson_grid'+str(g)
        run_dir = os.path.join(save_dir, run_name)
        os.mkdir(run_dir)
        os.chdir(run_dir)
        
        rhs_train, x_train = poisson(N_train, g, rng)
        rhs_test, x_test = poisson(N_test, g, rng)

        # Save train
        hf_train = h5py.File(run_name+'_train.h5', 'w')
        hf_train.create_dataset('rhs', data=rhs_train)
        hf_train.create_dataset('x', data=x_train)
        hf_train.close()
        
        # Save test
        hf_test = h5py.File(run_name+'_test.h5', 'w')
        hf_test.create_dataset('rhs', data=rhs_test)
        hf_test.create_dataset('x', data=x_test)
        hf_test.close()
    return

def fd_mtx2(a):
    """
    Cell-centered second-order finite volume approximation of a 2D scalar diffusion equation in QTT.
    This function creates a discretized Laplacian matrix with Dirichlet boundary conditions.
    """
    n = a.shape[0] - 1  # The coefficient is (n+1)x(n+1)

    # Initialize arrays
    ad = np.zeros((n, n))
    for i in range(n-1):
        for j in range(n):
            ad[i, j] = 0.5 * (a[i+1, j] + a[i+1, j+1])

    au = np.zeros((n, n))
    au[1:n, :] = ad[0:n-1, :]

    al = np.zeros((n, n))
    for i in range(n):
        for j in range(n-1):
            al[i, j] = 0.5 * (a[i, j+1] + a[i+1, j+1])

    ar = np.zeros((n, n))
    ar[:, 1:n] = al[:, 0:n-1]

    ac = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ac[i, j] = a[i, j] + a[i, j+1] + a[i+1, j] + a[i+1, j+1]

    bar_a = np.column_stack((                    # Flatten arrays and combine into matrix
        -al.flatten("F"),
        -ad.flatten("F"),
        ac.flatten("F"),
        -au.flatten("F"),
        -ar.flatten("F"))
    )
    offsets = [-n, -1, 0, 1, n]                  # Create diagonal offsets for the sparse matrix
    mat = spdiags(bar_a.T, offsets, n*n, n*n)    # Create the sparse matrix using spdiags
    mat = mat * (n + 1) ** 2                     # Multiply by scaling factor (n+1)^2
    return mat.tocsc()
