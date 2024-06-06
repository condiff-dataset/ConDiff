from datasets import load_dataset
import h5py
import os

def load_ConDiff(save_dir, pde, grid, covariance='cubic', variance=0.1):
    """
    Helper function for dataset loading from the Hugging Face Hub.
    This function is a recommended way to load ConDiff.
    
    Parameters
    ----------
    save_dir : str
        Existing directory to write/read files.
    pde : {'poisson', 'diffusion'}
        PDE. If `pde` is `poisson`, parameters 
        `covariance` and `variance` are ignored.
    covariance : {'cubic', 'exponential', 'gaussian'}, default 'cubic'
        Covariance model for Gaussian random field (GRF).
        The Diffusion coefficient `k` is generated as: k = exp(GRF).
    variance : {0.1, 0.4, 1.0, 2.0}, defalut 0.1
        Variance of the Gaussian random field.
    grid : {64, 128}
        Computational grid size.

    Returns
    -------
    train_data : {(rhs_train, x_train), (k_train, rhs_train, x_train)}
        If `pde` is `poisson`, returns (rhs_train, x_train), otherwise
        (k_train, rhs_train, x_train). 
    test_data : {(rhs_test, x_test), (k_test, rhs_test, x_test)}
        If `pde` is `poisson`, returns (rhs_test, x_test), otherwise
        (k_test, rhs_test, x_test). 
        
    Notes
    -----
    rhs_train, rhs_test : np.ndarray
        Right hand side of the PDE with shape=(num_samples, grid**2)
        for the subset train\test.
    x_train, x_test : np.ndarray
        Solution of the PDE with shape=(num_samples, grid**2) for the
        subset train\tets.
    k_train, k_test : np.ndarray
        Diffusion coefficient with shape=(num_samples, grid+1, grid+1)
        for the subset train/test.
    """
    assert os.path.isdir(save_dir)
    assert pde in ['poisson', 'diffusion']
    assert covariance in ['cubic', 'exponential', 'gaussian']
    assert str(variance) in ['0.1', '0.4', '1.0', '2.0']
    assert str(grid) in ['64', '128']
    
    if pde == 'poisson':
        name = 'poisson_grid' + str(grid)
    else:
        name = covariance + str(variance) + '_grid' + str(grid)
        
    dataset = load_dataset("condiff/ConDiff", name=name, cache_dir=save_dir)
    hf_train = h5py.File(dataset['train'].data.to_pandas().iloc[0, 0]['path'], "r")
    hf_test = h5py.File(dataset['test'].data.to_pandas().iloc[0, 0]['path'], "r")
    
    if pde == 'poisson':
        rhs_train, x_train = hf_train['rhs'][:], hf_train['x'][:]
        rhs_test, x_test = hf_test['rhs'][:], hf_test['x'][:]
        train_data = (rhs_train, x_train)
        test_data = (rhs_test, x_test)
    else:
        k_train, rhs_train, x_train = hf_train['k'][:], hf_train['rhs'][:], hf_train['x'][:]
        k_test, rhs_test, x_test = hf_test['k'][:], hf_test['rhs'][:], hf_test['x'][:]
        train_data = (k_train, rhs_train, x_train)
        test_data = (k_test, rhs_test, x_test)
    return train_data, test_data
