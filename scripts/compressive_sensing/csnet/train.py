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


@click.command()
@click.argument('record_num', type=int)
@click.option('-n', default=256, help='Window length')
@click.option('-m', default=128, help='Measurement length')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option('-q', default=1, help='Quantization nmse factor')
@click.option('-c', default=2, help='Clipping nmse factor')
@click.option('-w', default=16, help='Windows per frame')
def main(record_num, n, m, d, q, c, w):

    mit_bih_dir = os.getenv('MIT_BIH_DIR')
    path = f'{mit_bih_dir}/{record_num}'
    header = wfdb.rdheader(path)
    click.echo(f'Processing record: {record_num}: Sampling rate: {header.fs}')
    fs = float(header.fs)
    sampfrom=0
    sampto=None
    header = wfdb.rdheader(path)
    record = wfdb.rdrecord(path, channels=[0]
        , sampfrom=sampfrom, sampto=sampto, physical=False)
    # data
    ecg = np.squeeze(record.d_signal) - int(record.baseline[0])

    q_nmse_limit = Decimal((0, (q,), -2))
    c_nmse_limit = Decimal((0, (c,), -2))
    params = codec.EncoderParams(key=crn.KEY0, n=n, m=m, d=d, w=w, 
        adaptive=True,
        q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
    Phi = codec.build_sensor(params)
    coded_ecg = codec.encode(params, ecg)
    info = coded_ecg.info
    y = codec.decode_measurements(coded_ecg.bits)
    Y = crn.vec_to_windows(jnp.asarray(y), m)
    X = crn.vec_to_windows(jnp.asarray(ecg), n)
    X = X.T
    Y = Y.T
    state = model.train_and_evaluate(Phi, X, Y, params)
    file_path = f'model_n-{n}_m-{m}_d-{d}_q-{q}_c-{c}.mdl'
    model.save_to_disk(state, file_path)

if __name__ == '__main__':
    main()
