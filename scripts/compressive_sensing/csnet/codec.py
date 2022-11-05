# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

import math
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
@click.option('-p', '--pms', default=50, help='Percentage measurement savings')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option('-q', default=1, help='Quantization nmse factor')
@click.option('-c', default=2, help='Clipping nmse factor')
@click.option('-r', default=2, help='nmse threshold exponent')
@click.option('-w', default=64, help='Windows per frame')
def main(record_num, n, pms, d, q, c, r, w):
    # measurement ratio as percentage
    mr = 100 - pms
    # number of measurements
    m = math.ceil(n * mr / 100)
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

    q_nmse_limit = Decimal((0, (q,), -r))
    c_nmse_limit = Decimal((0, (c,), -r))
    params = codec.EncoderParams(key=crn.KEY0, n=n, m=m, d=d, w=w, 
        adaptive=True,
        q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
    print(params)
    Phi = codec.build_sensor(params)
    coded_ecg = codec.encode(params, ecg)


    models_dir = 'models'
    models_dir = os.path.abspath(models_dir)
    ckpt_dir_name = f'n-{n}_pms-{pms}_d-{d}_q-{q}_c-{c}_r-{r}_w-{w}'
    ckpt_dir_path = os.path.join(models_dir, ckpt_dir_name)
    print(f'Check-point directory: {ckpt_dir_name}')
    config = model.get_config(ckpt_dir=ckpt_dir_path)

    reconstructor = model.Reconstructor(config, params)
    # decompression
    decoded_ecg = codec.decode_general(coded_ecg.bits, 
        reconstructor=reconstructor)
    # encoding info
    info = coded_ecg.info
    print(info)
    # stats
    stats = codec.compression_stats(ecg, coded_ecg, decoded_ecg)
    print(stats)
    #model.test_loss(net, net_params, Phi, X, Y, d)

if __name__ == '__main__':
    main()
