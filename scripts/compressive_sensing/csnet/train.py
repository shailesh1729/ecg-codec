import time
import click
import os
from decimal import Decimal

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
@click.option('-m', default=128, help='Measurement length')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option('-q', default=1, help='Quantization nmse factor')
@click.option('-c', default=2, help='Clipping nmse factor')
@click.option('-w', default=16, help='Windows per frame')
def main(n, m, d, q, c, w):

    mit_bih_dir = os.getenv('MIT_BIH_DIR')

    q_nmse_limit = Decimal((0, (q,), -2))
    c_nmse_limit = Decimal((0, (c,), -2))
    params = codec.EncoderParams(key=crn.KEY0, n=n, m=m, d=d, w=w, 
        adaptive=True,
        q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
    Phi = codec.build_sensor(params)

    record_nums = MIT_BIH['record_nums']
    signals = []
    n_rec = 0
    for record_num in record_nums:
        if record_num in TEST_SET:
            continue
        n_rec += 1
        click.echo(f'Reading [{n_rec}]: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        fs = float(header.fs)
        sampfrom=0
        # sampto=n*100
        sampto=None
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
    state = model.train_and_evaluate(Phi, X, Y, params)
    file_path = f'model_n-{n}_m-{m}_d-{d}_q-{q}_c-{c}.mdl'
    model.save_to_disk(state, file_path)

if __name__ == '__main__':
    main()
