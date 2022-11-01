import jax
import jax.numpy as jnp

from skecg.cs.csnet import model

def main():
    config = model.get_config()
    n = 256
    net = model.CSNet()
    shape = (config.batch_size, n, 1)
    x = jnp.empty(shape)
    rng = jax.random.PRNGKey(0)
    params = net.init(rng, x)
    print(jax.tree_util.tree_map(lambda x: x.shape, params))
    x2 = net.apply(params, x)
    print(x.shape, x2.shape)


if __name__ == '__main__':
    main()
