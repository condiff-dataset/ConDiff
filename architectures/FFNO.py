# +
import equinox as eqx
import jax.numpy as jnp

from jax.nn import gelu
from jax.tree_util import tree_map
from jax.lax import dot_general
from transforms import utilities
from jax import config, random, grad, jit, vmap
from jax.lax import dot_general, scan, dynamic_slice_in_dim

config.update("jax_enable_x64", True)

def normalize_conv(A):
    A = eqx.tree_at(lambda x: x.weight, A, A.weight * jnp.sqrt(2. / A.weight.shape[1]))
    A = eqx.tree_at(lambda x: x.bias, A, jnp.zeros_like(A.bias))
    return A

class FFNO(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    convs1: list
    convs2: list
    A: jnp.array

    def __init__(self, N_layers, N_features, N_modes, key, D=1):
        n_in, n_processor, n_out = N_features

        keys = random.split(key, 3 + 2 * N_layers)
        self.encoder = normalize_conv(eqx.nn.Conv(D, n_in, n_processor, 1, key=keys[-1]))
        self.decoder = normalize_conv(eqx.nn.Conv(D, n_processor, n_out, 1, key=keys[-2]))
        self.convs1 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key)) for key in keys[:N_layers]]
        self.convs2 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key)) for key in keys[N_layers:2*N_layers]]
        self.A = random.normal(keys[-3], [N_layers, n_processor, n_processor, N_modes, D], dtype=jnp.complex64)*jnp.sqrt(2/n_processor)

    def __call__(self, u, x):
        u = jnp.concatenate([x, u], 0)
        u = self.encoder(u)
        for conv1, conv2, A in zip(self.convs1, self.convs2, self.A):
            u += gelu(conv2(gelu(conv1(self.spectral_conv(u, A)))))
        u = self.decoder(u)
        return u

    def spectral_conv(self, v, A):
        u = 0
        N = v.shape
        for i in range(A.shape[-1]):
            u_ = dynamic_slice_in_dim(jnp.fft.rfft(v, axis=i+1), 0, A.shape[-2], axis=i+1)
            u_ = dot_general(A[:, :, :, i], u_, (((1,), (0,)), ((2, ), (i+1, ))))
            u_ = jnp.moveaxis(u_, 0, i+1)
            u += jnp.fft.irfft(u_, axis=i+1, n=N[i+1])
        return u
    
def compute_loss(model, features, coordinates, targets):
    prediction = vmap(model, in_axes=(0, None))(features, coordinates)
    mean_error = jnp.mean(jnp.linalg.norm((prediction - targets).reshape(prediction.shape[0], -1), axis=1))
    return mean_error

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(carry, indices, optim):
    model, features, coordinates, targets, opt_state = carry
    loss, grads = compute_loss_and_grads(model, features[indices], coordinates, targets[indices])
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, coordinates, targets, opt_state], loss
