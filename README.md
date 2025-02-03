# ConDiff

The code repository for the paper
**ConDiff: A Challenging Dataset for Neural Solvers of Partial Differntial Equations**

## Installation

Clone this repo:

```
git clone https://github.com/condiff-dataset/ConDiff.git
```

Install requirements:

```
pip install -r requirements.txt
```

## Dataset

The ConDiff dataset is hosted on the Hagging Face Hub:
https://huggingface.co/datasets/condiff/ConDiff

You can download the dataset in two ways:

 - Clone the Hugging Face repo directly.
 
 - Use the `load_ConDiff.py` function.
 
The `load_ConDiff.py` function uses `datasets.load_dataset()` internally, but does not provide the same functionality.
Namely, it does not provide dataloader-like behaviour, but returns a tuple of `np.ndarray` of discretised coefficient functions,
right-hand sides and solution functions.

An example of basic usage can be found in `dataset_example.ipynb`.

To generate Gaussian random field we use highly efficient [parafields](https://github.com/parafields/parafields) library.

### Loading function `load_ConDiff.py` overview

```
Parameters
----------
save_dir : str
    Path to write/read data from the Hub.
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
```

### List of PDEs in ConDiff

ConDiff consists of diffusion equations with different distributions of the coefficient function.
PDEs vary with: (i) 4 different Gaussian random fields from {'cubic', 'exponential', 'gaussian'};
(ii) 4 different variance values from {0.1, 0.4, 1.0, 2.0}; (iii) 2 different grid sizes from {64, 128}.

 - cubic0.1_grid128
 - cubic0.1_grid64
 - cubic0.4_grid128
 - cubic0.4_grid64
 - cubic1.0_grid128
 - cubic1.0_grid64
 - cubic2.0_grid128
 - cubic2.0_grid64
 - exponential0.1_grid128
 - exponential0.1_grid64
 - exponential0.4_grid128
 - exponential0.4_grid64
 - exponential1.0_grid128
 - exponential1.0_grid64
 - exponential2.0_grid128
 - exponential2.0_grid64
 - gaussian0.1_grid128
 - gaussian0.1_grid64
 - gaussian0.4_grid128
 - gaussian0.4_grid64
 - gaussian1.0_grid128
 - gaussian1.0_grid64
 - gaussian2.0_grid128
 - gaussian2.0_grid64
 - poisson_grid128
 - poisson_grid64

## Data Generation

ConDiff can be reproduced with code from `generate_condiff.ipynb`.
Examples of obtaining a ground truth solution for a single PDE can be found in `dataset_example.ipynb`.

## Baseline Models Validation

Validate models on the ConDiff PDEs.

Here are some example notebook demonstrating the use of neural operators (such as [SNO](https://link.springer.com/article/10.1134/S1064562423701107) and [F-FNO](https://arxiv.org/abs/2111.13802)) and classical neural networks (such as [DilResNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Yu_Dilated_Residual_Networks_CVPR_2017_paper.html) and [UNet](https://arxiv.org/pdf/1505.04597)) to solve the Poisson equation 2D with grid = 64:

   * [F-FNO for Poisson 2D](https://github.com/condiff-dataset/ConDiff/blob/main/notebooks/FFNO%20for%20Poisson%202D.ipynb)
   * [SNO for Poisson 2D](https://github.com/condiff-dataset/ConDiff/blob/main/notebooks/SNO%20for%20Poisson%202D.ipynb)
   * [UNet for Poisson 2D](https://github.com/condiff-dataset/ConDiff/blob/main/notebooks/UNet%20for%20Poisson%202D.ipynb)
   * [DilResNet for Poisson 2D](https://github.com/condiff-dataset/ConDiff/blob/main/notebooks/DilResNet%20for%20Poisson%202D.ipynb)

These notebooks demonstrate the use of the same models to solve the 2D diffusion equation with grid = 64, covariance = 'cubic', variance = 0.1:

  * [F-FNO for Diffusion 2D](https://github.com/condiff-dataset/ConDiff/blob/main/notebooks/FFNO%20for%20Diffusion%202D.ipynb)
  * [SNO for Diffusion 2D](https://github.com/condiff-dataset/ConDiff/blob/main/notebooks/SNO%20for%20Diffusion%202D.ipynb)
  * [UNet for Diffusion 2D](https://github.com/condiff-dataset/ConDiff/blob/main/notebooks/UNet%20for%20Diffusion%202D.ipynb)
  * [DilResNet for Diffusion 2D](https://github.com/condiff-dataset/ConDiff/blob/main/notebooks/DilResNet%20for%20Diffusion%202D.ipynb)

## Citing

If you use the ConDiff dataset and/or find or code useful in your research, please cite ([arXiv link](https://arxiv.org/abs/2406.04709)):

```bibtex
@article{trifonov2024condiff,
  title={ConDiff: A Challenging Dataset for Neural Solvers of Partial Differential Equations},
  author={Trifonov, Vladislav and Rudikov, Alexander and Iliev, Oleg and Oseledets, Ivan and Muravleva, Ekaterina},
  journal={arXiv preprint arXiv:2406.04709},
  year={2024}
}
```
