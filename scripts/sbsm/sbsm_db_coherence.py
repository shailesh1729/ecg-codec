# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

import click

import numpy as np

from jax import random
import cr.sparse.dict as crdict

import matplotlib.pyplot as plt

@click.command()
@click.option('-n', default=512, help='Signal dimension')
@click.option('-m', default=256, help='Measurement dimension')
@click.option('-t', default=10, help='Number of dictionary trials for each configuration')
@click.option('-d', default=75, help='Maximum value of d')
@click.option('-w', default='db10', help='Wavelet')
def main(n, m, t, d, w):
    key0 = random.PRNGKey(0)
    keys  = random.split(key0, t)
    Psi = crdict.wavelet_basis(n, w)
    ds = np.arange(1, d+1)
    mean_mus = np.zeros(ds.shape)
    for i in range(len(ds)):
        d = ds[i]
        mus = np.zeros(len(keys))
        for j, key in enumerate(keys):
            Phi  = crdict.sparse_binary_mtx(key, m, n, d=d, 
                normalize_atoms=True, dense=True)
            mu = crdict.mutual_coherence(Psi, Phi.T)
            print(f'd={d}, j={j}, mu={mu:.3f}')
            mus[j] = mu
        mean_mus[i] = mus.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ds, mean_mus)
    ax.grid()
    ax.set_xlabel('d')
    ax.set_ylabel('Mutual coherence (average)')
    ax.set_title(f'Mutual coherence of sparse binary sensing matrix with {w} basis')
    plt.savefig(f'coherence-w_{w}-m_{m}-n_{n}-t_{t}-dmax_{d}.png')


if __name__ == '__main__':
    main()
