# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

# std imports
import time
import click
import os
import math
from decimal import Decimal
from typing import NamedTuple

from skecg.physionet import *

import numpy as np
import pandas as pd
from statsmodels.stats import descriptivestats

import cr.nimble as crn
import skecg.cs.codec_b as codec
from skecg.cs.csnet import model
from skecg.util import kld_normal

import wfdb
from dotenv import load_dotenv
load_dotenv()

def get_db_dir():
    db_dir = os.getenv('MIT_BIH_DIR')
    if not db_dir:
        click.echo('ERROR: Please configure the environment variable MIT_BIH_DIR.')
        sys.exit(1)
    return db_dir

class Row(NamedTuple):
    record: int
    "record number"
    m: int
    "measurement space dimension"
    n: int
    "signal space dimension"
    d: int
    "number of ones per column in the sensing matrix"
    f: int
    "number of frames"
    w: int
    "number of windows"
    s: int
    "number of samples"
    u_bits: int
    "Uncompressed bits count"
    c_bits: int
    "compressed bits count"
    overhead_bits: int
    "Total number of header bits"
    bpm: float
    "bits per measurement"
    bps: float
    "bits per sample"
    overhead: float
    "overhead of header bits"
    cr: float
    "compression ratio"
    pms: float
    "percentage measurement savings"
    pss: float
    "percentage space savings"
    snr: float
    "signal to noise ratio (dB)"
    prd: float
    "percent root mean square difference"
    nmse: float
    "normalized mean square difference"
    rtime: float
    "reconstruction time"
    qc_snr: float
    qc_prd: float
    qc_nmse: float
    min_qc_snr: float
    max_qc_snr: float
    min_q: int
    max_q: int
    min_rng: int
    max_rng: int


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

    pms_vals = np.arange(5, 95, 5)
    destination = f'codec_b_csnet_n-{n}_d-{d}_q-{q}_c-{c}-r-{r}-prd-cr-stats.csv'
    q_nmse_limit = Decimal((0, (q,), -r))
    c_nmse_limit = Decimal((0, (c,), -r))

    mit_bih_dir = get_db_dir()
    record_nums = MIT_BIH['record_nums']
    all_stats = []

    sampfrom=0
    if dry:
        sampto=10*360
    else:
        sampto=None

    models_dir = 'models'
    models_dir = os.path.abspath(models_dir)

    for rci, record_num in enumerate(record_nums):
        click.echo(f'Processing record: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0]
            , sampfrom=sampfrom, sampto=sampto, physical=False)
        # data
        ecg = np.squeeze(record.d_signal) - int(record.baseline[0])
        for ri, pms in enumerate(pms_vals):
            print(f'At PMS: {pms} %')
            # measurement ratio as percentage
            mr = 100 - pms
            # number of measurements
            m = math.ceil(n * mr / 100)
            params = codec.EncoderParams(key=crn.KEY0, 
                n=n, m=m, d=d, w=w, adaptive=True,
                q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
            # compression
            coded_ecg = codec.encode(params, ecg)
            # CS-NET reconstruction wrapper
            ckpt_dir_name = f'n-{n}_pms-{pms}_d-{d}_q-{q}_c-{c}_r-{r}_w-{w}'
            ckpt_dir_path = os.path.join(models_dir, ckpt_dir_name)
            if not os.path.exists(ckpt_dir_path):
                click.echo(f'There is no checkpoint for PMS={pms}. Skipping.')
                continue
            # print(f'Check-point directory: {ckpt_dir_name}')
            config = model.get_config(ckpt_dir=ckpt_dir_path)
            reconstructor = model.Reconstructor(config, params)
            # decompression
            decoded_ecg = codec.decode_general(coded_ecg.bits, 
                reconstructor=reconstructor)
            # release older model from GPU
            reconstructor = None
            # encoding info
            info = coded_ecg.info
            click.echo(info)
            q_vals = info.q_vals
            rng_mults = info.rng_mults
            qc_snrs = info.qc_snrs
            # stats
            stats = codec.compression_stats(ecg, coded_ecg, decoded_ecg)
            row = Row(record=record_num, 
                m=m, n=n, d=d,
                w=info.n_windows, 
                f=info.n_frames, s=info.n_samples,
                overhead_bits=info.overhead_bits,
                overhead=info.total_overhead,
                u_bits=stats.u_bits,
                c_bits=stats.c_bits,
                cr=stats.cr, pms=pms, pss=stats.pss,
                bpm=stats.bpm, bps=stats.bps,
                snr=stats.snr, prd=stats.prd, 
                nmse=stats.nmse, rtime=stats.rtime,
                qc_snr=stats.qc_snr, qc_prd=stats.qc_prd,
                qc_nmse=stats.qc_nmse,
                min_qc_snr=np.min(qc_snrs),
                max_qc_snr=np.max(qc_snrs),
                min_q=np.min(q_vals),
                max_q=np.max(q_vals),
                min_rng=np.min(rng_mults),
                max_rng=np.max(rng_mults)
                )
            click.echo(row)
            all_stats.append(row)
            if len(all_stats) % 4 == 0:
                # save results after every 4 record
                df = pd.DataFrame(all_stats, columns=Row._fields)
                df.to_csv(destination)
        print('Processing of record complete')
    # save results after all records
    df = pd.DataFrame(all_stats, columns=Row._fields)
    df.to_csv(destination)

if __name__ == '__main__':
    main()
