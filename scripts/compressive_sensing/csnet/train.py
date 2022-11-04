# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

import time
import click
import os
from decimal import Decimal
import math
import wfdb
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from jax import random, numpy as jnp

import cr.nimble as crn
from skecg.cs.csnet import model
import skecg.cs.codec_b as codec

from skecg.physionet import *

TEST_SET = [100, 101, 102, 107, 109, 111, 115, 117, 118, 119]

@click.command()
@click.option('-n', default=256, help='Window length')
@click.option('-p', '--pms', default=50, help='Percentage measurement savings')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option('-q', default=1, help='Quantization nmse factor')
@click.option('-c', default=2, help='Clipping nmse factor')
@click.option('-r', default=2, help='nmse threshold exponent')
@click.option('-w', default=64, help='Windows per frame')
@click.option("--dry", is_flag=True, 
    show_default=True, default=False, help="Dry run with small samples")
def main(n, pms, d, q, c, r, w, dry):
    # measurement ratio as percentage
    mr = 100 - pms
    # number of measurements
    m = math.ceil(n * mr / 100)
    mit_bih_dir = os.getenv('MIT_BIH_DIR')

    q_nmse_limit = Decimal((0, (q,), -r))
    c_nmse_limit = Decimal((0, (c,), -r))
    params = codec.EncoderParams(key=crn.KEY0, n=n, m=m, d=d, w=w, 
        adaptive=True,
        q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
    Phi = codec.build_sensor(params)

    models_dir = 'models'
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    ckpt_dir_name = f'n-{n}_pms-{pms}_d-{d}_q-{q}_c-{c}_r-{r}_w-{w}'
    ckpt_dir_path = os.path.join(models_dir, ckpt_dir_name)

    record_nums = MIT_BIH['record_nums']
    signals = []
    sampfrom = 0
    if dry:
        sampto=n*128
        epochs = 10
    else:
        sampto=None
        epochs = 400

    n_rec = 0
    for record_num in record_nums:
        if record_num in TEST_SET:
            continue
        n_rec += 1
        click.echo(f'Reading [{n_rec}]: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        fs = float(header.fs)
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0]
            , sampfrom=sampfrom, sampto=sampto, physical=False)
        # data
        ecg = np.squeeze(record.d_signal) - int(record.baseline[0])
        signals.append(ecg)
    print('Processing all records together for model training.')
    # all signals together
    ecg = np.concatenate(signals)
    # encode all signals in one go
    print('Encoding data')
    coded_ecg = codec.encode(params, ecg)
    # decode measurements
    print('Decoding measurements.')
    y = codec.decode_measurements(coded_ecg.bits)
    print('Windowing for training.')
    Y = crn.vec_to_windows(jnp.asarray(y), m)
    X = crn.vec_to_windows(jnp.asarray(ecg), n)
    X = X.T
    Y = Y.T
    print('Starting training.')
    config = model.get_config(epochs=epochs,
        ckpt_dir=ckpt_dir_path)
    result = model.train_and_evaluate(Phi, X, Y, params, config)

if __name__ == '__main__':
    main()
