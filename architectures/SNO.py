# +
import equinox as eqx
import jax.numpy as jnp

from jax.nn import gelu
from jax.lax import dot_general
from jax.tree_util import tree_map
from transforms import utilities
from jax import config, random, grad, jit, vmap
from architectures.DilResNet import DilatedConvBlock

config.update("jax_enable_x64", True)

class tensor_cell(eqx.Module):
    A: jnp.array
    b: jnp.array
    D: int

    def __init__(self, N_features, key, N_modes):
        self.D = len(N_modes)
        keys = random.split(key, 2)
        self.A = random.normal(keys[0], [N_features,]*2 + list(N_modes)) / N_features
        self.b = random.normal(keys[1], [N_features,] + [1 for i in range(self.D)])

    def __call__(self, x):
        return jnp.rollaxis(dot_general(self.A, x, (((1,), (0, )), ([2 + i for i in range(self.D)], [1 + i for i in range(self.D)]))), -1) + self.b
    
class fSNO(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    conv: list
    processor: list

    def __init__(self, input_shape, features_out, N_layers, N_features, cell, key, activation=gelu):
        features_in = input_shape[0]
        N_input = list(input_shape[1:])
        D = len(N_input)
        keys = random.split(key, 4)

        self.encoder = DilatedConvBlock([features_in, N_features], [[1]*D, ], [[1]*D, ], keys[0])
        self.decoder = DilatedConvBlock([N_features, features_out], [[1]*D, ], [[1]*D, ], keys[1])

        keys = random.split(keys[2], N_layers)
        self.processor = [cell(N_features, k) for k in keys]

        keys = random.split(keys[3], N_layers)
        self.conv = [eqx.nn.Conv(num_spatial_dims=D, in_channels=N_features, out_channels=N_features, kernel_size=[1,]*D, key=key) for key in keys]

    def __call__(self, x, analysis, synthesis):
        x = self.encoder.linear_call(x)
        for pr, conv in zip(self.processor[:-1], self.conv[:-1]):
            y = conv(x)
            x = utilities.apply_operators(x, analysis)
            x = pr(x)
            x = utilities.apply_operators(x, synthesis)
            x = gelu(x + y)
        y = self.conv[-1](x)
        x = utilities.apply_operators(x, analysis)
        x = self.processor[-1](x)
        x = utilities.apply_operators(x, synthesis)
        x = x + y
        x = self.decoder.linear_call(x)
        return x

class fSNO_truncated(fSNO):
    pooling: eqx.Module

    def __init__(self, input_shape, features_out, N_layers, N_features, cell, key, output_shape, activation=gelu):
        super().__init__(input_shape, features_out, N_layers, N_features, cell, key, activation=activation)
        self.pooling = eqx.nn.AdaptivePool(output_shape[1:], len(list(input_shape[1:])), jnp.mean)

    def __call__(self, x, analysis, synthesis):
        x = super().__call__(x, analysis, synthesis)
        x = self.pooling(x)
        return x

def compute_loss(model, features, targets, analysis, synthesis):
    prediction = vmap(lambda x: model(x, analysis, synthesis), in_axes=(0, ))(features)
    mean_error = jnp.mean(jnp.linalg.norm((prediction - targets).reshape(prediction.shape[0], -1), axis=1))
    return mean_error

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(carry, indices, analysis, synthesis, optim):
    model, features, targets, opt_state = carry
    loss, grads = compute_loss_and_grads(model, features[indices], targets[indices], analysis, synthesis)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, opt_state], loss
