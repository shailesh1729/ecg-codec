from typing import NamedTuple
import os
import click
import numpy as np
import pandas as pd


import wfdb
import wfdb.processing

import matplotlib.pyplot as plt

# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.lop
import cr.wavelets as wt
import jax
from jax import random
import jax.numpy as jnp

from skecg.physionet import MIT_BIH

mit_bih_dir = 'F:/datasets/medical/ecg/mit-bih-arrhythmia-database-1.0.0'


def get_signal(record_num):
    record = wfdb.rdrecord(f'{mit_bih_dir}/{record_num}', channels=[0])
    signal = np.squeeze(record.p_signal)
    return signal

class Row(NamedTuple):
    family: str
    wavelet: str
    level: int
    record: int
    blocks: int
    k_min: int
    k_max: int
    k_mean: float


@click.command()
@click.argument('block_len', type=int)
@click.argument('energy_level', type=float)
def main(block_len, energy_level):

    experiment_name = f'size_{block_len}_energy_{energy_level:.2f}'
    destination = f'{experiment_name}.csv'
    df = pd.DataFrame(columns=Row._fields)

    wavelist = wt.wavelist(kind='discrete')
    for name in wavelist:
        wavelet = wt.to_wavelet(name)
        family = wavelet.short_name
        max_level = wt.dwt_max_level(block_len, wavelet.dec_len)
        print(f'{name}: {max_level}')
        PsiT = crs.lop.dwt(block_len, name, max_level)
        PsiT = crs.lop.jit(PsiT)
        for rec_num in record_nums:
            signal = get_signal(rec_num)
            X = crn.vec_to_windows(signal, block_len)
            blocks = X.shape[1]
            A = PsiT.times(X)
            ks = jnp.apply_along_axis(
                lambda a : crn.num_largest_coeffs_for_energy_percent(a, energy_level), 0, A)
            k_max = int(jnp.max(ks))
            k_min = int(jnp.min(ks))
            k_mean = float(jnp.mean(ks))
            row = Row(family=family, wavelet=name, level=max_level,
                record=rec_num, blocks=blocks,
                k_min=k_min, k_max=k_max, k_mean=k_mean)
            print(row)
            df.loc[len(df)] = row
        # save results after each wavelet
        df.to_csv(destination, index=True, float_format='%.2f')
    # save results after all wavelets
    df.to_csv(destination, index=True, float_format='%.2f')


if __name__ == '__main__':
    main()

