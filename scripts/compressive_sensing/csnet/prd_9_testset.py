# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

# std imports
import math
import click
import os
from decimal import Decimal
from typing import NamedTuple

import wfdb
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from jax import random, numpy as jnp

import cr.nimble as crn
from skecg.cs.csnet import model
import skecg.cs.codec_b as codec

import pandas as pd
from statsmodels.stats import descriptivestats

# Optimal PMS reported for CSNet in Zhang et al. CSNet paper
CSNET_TEST_SET = [
(100, 88),
(101, 88),
(102, 63),
(107, 81),
(109, 87),
(111, 80),
(115, 91),
(117, 93),
(118, 87),
(119, 92),
]


class Row(NamedTuple):
    record: int
    "record number"
    reference: int
    "reference PMS"
    m: int
    "measurement space dimension"
    n: int
    "signal space dimension"
    d: int
    "number of ones per column in the sensing matrix"
    u_bits: int
    "Uncompressed bits count"
    c_bits: int
    "compressed bits count"
    cr: float
    "compression ratio"
    pms: int
    "percentage measurement ratio"
    pss: float
    "percentage space savings"
    snr: float
    "signal to noise ratio (dB)"
    prd: float
    "percent root mean square difference"
    rtime: float
    "reconstruction time"

@click.command()
@click.option('-n', default=256, help='Window length')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option('-q', default=1, help='Quantization nmse factor')
@click.option('-c', default=2, help='Clipping nmse factor')
@click.option('-r', default=2, help='nmse threshold exponent')
@click.option('-w', default=64, help='Windows per frame')
@click.option("--dry", is_flag=True, 
    show_default=True, default=False, help="Dry run with small samples")
def main(n, d, q, c, r, w, dry):
    destination = f'codec_b_target_prd_csnet_n-{n}_d-{d}-q-{q}-c-{c}-r-{r}-stats.csv'

    mit_bih_dir = os.getenv('MIT_BIH_DIR')
    q_nmse_limit = Decimal((0, (q,), -r))
    c_nmse_limit = Decimal((0, (c,), -r))
    all_stats = []

    sampfrom=0
    if dry:
        sampto=10*360
    else:
        sampto=None

    TARGET_PRD = 9
    models_dir = 'models'
    models_dir = os.path.abspath(models_dir)
    for example in CSNET_TEST_SET:
        record_num, optimal_pms = example
        click.echo(f'Processing record: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0]
            , sampfrom=sampfrom, sampto=sampto, physical=False)
        # data
        ecg = np.squeeze(record.d_signal) - int(record.baseline[0])
        pms = optimal_pms + 2
        CURRENT_PRD = 100
        while CURRENT_PRD > TARGET_PRD:
            # decrease pms
            pms = pms - 1
            if pms < 50:
                # it's too bad for this record
                break
            ckpt_dir_name = f'n-{n}_pms-{pms}_d-{d}_q-{q}_c-{c}_r-{r}_w-{w}'
            ckpt_dir_path = os.path.join(models_dir, ckpt_dir_name)
            # check point for the saved model for this configuration
            if not os.path.exists(ckpt_dir_path):
                click.echo(f'Check-point directory does not exist: {ckpt_dir_name}')
                continue
            mr = 100 - pms
            m = math.ceil(n * mr / 100)
            print(f'Trying with PMS={pms} and m={m}')
            params = codec.EncoderParams(key=crn.KEY0, 
                n=n, m=m, d=d, w=w, adaptive=True,
                q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
            # compression
            coded_ecg = codec.encode(params, ecg)
            config = model.get_config(ckpt_dir=ckpt_dir_path)
            reconstructor = model.Reconstructor(config, params)
            # decompression
            decoded_ecg = codec.decode_general(coded_ecg.bits, 
                reconstructor=reconstructor)
            # encoding info
            info = coded_ecg.info
            # click.echo(info)
            # stats
            stats = codec.compression_stats(ecg, coded_ecg, decoded_ecg)
            row = Row(record=record_num,
                reference=optimal_pms, 
                m=m, n=n, d=d,
                u_bits=stats.u_bits,
                c_bits=stats.c_bits,
                cr=stats.cr, pms=pms, pss=stats.pss,
                snr=stats.snr, prd=stats.prd, 
                rtime=stats.rtime,
                )
            click.echo(row)
            CURRENT_PRD = stats.prd
            if len(all_stats) % 4 == 0:
                # save results after every 4 record
                df = pd.DataFrame(all_stats, columns=Row._fields)
                df.to_csv(destination)
            all_stats.append(row)
    # save results after all records
    df = pd.DataFrame(all_stats, columns=Row._fields)
    df.to_csv(destination)

if __name__ == '__main__':
    main()
