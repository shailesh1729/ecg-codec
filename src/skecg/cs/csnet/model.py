import numpy as np
import jax
from jax import numpy as jnp
import ml_collections

from flax import linen as nn
from flax.training import train_state
from flax import serialization
import optax


class CSNet(nn.Module):
  """CS Net for ECG"""

  @nn.compact
  def __call__(self, x):

    x = nn.Conv(features=64, kernel_size=(11,), padding='CAUSAL')(x)
    x = nn.relu(x)
    x = nn.Conv(features=32, kernel_size=(11,), padding='CAUSAL')(x)
    x = nn.relu(x)
    x = nn.Conv(features=1, kernel_size=(11,), padding='CAUSAL')(x)
    return x



def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # config.learning_rate = 0.1
  # config.momentum = 0.9
  config.learning_rate = 0.0005
  config.batch_size = 256
  config.num_epochs = 300
  return config

@jax.jit
def apply_model(state, X_input, X_true):
  """Computes gradients, loss for a single batch."""
  def loss_fn(params):
    X_est = state.apply_fn({'params': params}, X_input)
    x_diff = X_est - X_true
    # scale down
    x_diff = x_diff / 1024.0
    loss = jnp.mean(x_diff * x_diff) / 2.0
    return loss, X_est

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, X_est), grads = grad_fn(state.params)
  return grads, loss

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def create_train_state(rng, X, config):
    """Creates initial `TrainState`."""
    model = CSNet()
    params = model.init(rng, X)['params']
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


def train_epoch(state, X_risen, X_true, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = X_true.shape[0]
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []

  for perm in perms:
    batch_input = X_risen[perm, ...]
    batch_expected = X_true[perm, ...]
    grads, loss = apply_model(state, batch_input, batch_expected)
    state = update_model(state, grads)
    epoch_loss.append(loss)
  train_loss = np.mean(epoch_loss)
  return state, train_loss



def train_and_evaluate(Phi, X, Y, codec_params):
    config = get_config()
    X_risen = Y @ Phi / codec_params.d
    n  = codec_params.n
    X_true = jnp.expand_dims(X, 2)
    X_risen = jnp.expand_dims(X_risen, 2)
    print(X_true.shape, X_risen.shape)

    rng = jax.random.PRNGKey(0)

    ## train validation split
    n_total = X_risen.shape[0]
    n_validation = n_total // 8
    n_training = n_total - n_validation

    rng, split_rng = jax.random.split(rng)
    perms = jax.random.permutation(split_rng, n_total)
    train_idx = perms[:n_training]
    valid_idx = perms[n_training:]
    X_true_train = X_true[train_idx, ...]
    X_risen_train = X_risen[train_idx, ...]
    X_true_validation = X_true[valid_idx, ...]
    X_risen_validation = X_risen[valid_idx, ...]


    # initialize the network
    rng, init_rng = jax.random.split(rng)
    shape = (1, n, 1)
    dummy_x = jnp.empty(shape)
    state = create_train_state(init_rng, dummy_x, config)
    # print(jax.tree_util.tree_map(lambda x: x.shape, state.params))

    # perform training
    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss = train_epoch(state, X_risen_train, X_true_train,
            config.batch_size,
            input_rng)

        _, validation_loss = apply_model(state, X_risen_validation,
                                              X_true_validation)

        print(f'epoch:{epoch}, train_loss: {train_loss:.2e}, validation_loss: {validation_loss:.2e}')

    # return the final trained model
    return state 


def save_to_disk(state, file_path):
    params = state.params
    bytes_output = serialization.to_bytes(params)
    with open(file_path, 'wb') as f:
        f.write(bytes_output)
        f.close()


def load_from_disk(file_path, n):
    shape = (1, n, 1)
    x = jnp.empty(shape)
    model = CSNet()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x)['params']
    with open(file_path, 'rb') as f:
        bytes_output = f.read()
        f.close()
        params = serialization.from_bytes(params, bytes_output)
        print(jax.tree_util.tree_map(lambda x: x.shape, params))
        return model, params


def predict(net, params, Phi, Y, d):
    X_risen = Y @ Phi / d
    X_risen = jnp.expand_dims(X_risen, 2)
    X_est = net.apply({'params': params}, X_risen)
    return X_est


def test_loss(net, params, Phi, X, Y, d):
    X_risen = Y @ Phi / d
    X_true = jnp.expand_dims(X, 2)
    X_risen = jnp.expand_dims(X_risen, 2)
    print(X_true.shape, X_risen.shape)
    X_est = net.apply({'params': params}, X_risen)
    x_diff = X_est - X_true
    # scale down
    x_diff = x_diff / 1024.0
    loss = jnp.mean(x_diff * x_diff) / 2.0
    print(f'Test loss: {loss:.3e}')
