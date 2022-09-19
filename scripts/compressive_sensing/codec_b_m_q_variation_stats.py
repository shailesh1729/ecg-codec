# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

from decimal import Decimal
from typing import NamedTuple
from skecg.apps.apputils import *
from skecg.physionet import *

import pandas as pd
from statsmodels.stats import descriptivestats

import skecg.cs.codec_b as codec


class Row(NamedTuple):
    record: int
    "record number"
    mr : float
    "measurement ratio"
    m: int
    "measurement space dimension"
    n: int
    "signal space dimension"
    d: int
    "number of ones per column in the sensing matrix"
    q: int
    "quantization parameter"
    b: int
    "block size for BSBL"
    w: int
    "number of windows"
    s: int
    "number of samples"
    u_bits: int
    "Uncompressed bits count"
    c_bits: int
    "compressed bits count"
    bpm: float
    "bits per measurement"
    bps: float
    "bits per sample"
    cr: float
    "compression ratio"
    pss: float
    "percentage space savings"
    overhead: float
    "overhead"
    snr: float
    "signal to noise ratio (dB)"
    prd: float
    "percent root mean square difference"
    nmse: float
    "normalized mean square difference"
    qs: float
    "quality score"
    rtime: float

@click.command()
@click.argument('record_num', type=int)
@click.option('-n', default=512, help='Window length')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option('-c', default=2, help='Clipping nmse factor')
@click.option('-w', default=16, help='Windows per frame')
@click.option('-b', '--block-size', default=32, help='BSBL block size')
@click.option("--dry", is_flag=True, 
    show_default=True, default=False, help="Dry run with small samples")
def main(record_num, n, d, c, w, block_size, dry):
    destination = f'codec-b-rec={record_num}-n={n}-d={d}-m-q-var-stats.csv'
    mit_bih_dir = get_db_dir()
    path = f'{mit_bih_dir}/{record_num}'
    header = wfdb.rdheader(path)
    click.echo(f'Processing record: {record_num}: Sampling rate: {header.fs}')

    sampfrom=0
    if dry:
        sampto=10*360
    else:
        sampto=None
    # sampto=None
    record = wfdb.rdrecord(f'{mit_bih_dir}/{record_num}', channels=[0]
        , sampfrom=sampfrom, sampto=sampto, physical=False)
    ecg = np.squeeze(record.d_signal) - int(record.baseline[0])

    mrs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    qs = [0, 1, 2, 3, 4, 5, 6, 7]
    all_stats = []
    q_nmse_limit = Decimal((0, (0,), -2))
    c_nmse_limit = Decimal((0, (c,), -2))
    for mr in mrs:
        for q in qs:
            m = int(round(mr * n))
            print(f'mr={mr}, m={m}, q={q}')
            params = codec.EncoderParams(key=crn.KEYS[0], n=n, m=m, d=d, w=w, 
                adaptive=False,
                q=q, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
            coded_ecg = codec.encode(params, ecg)
            info = coded_ecg.info
            print(info)
            decoded_ecg = codec.decode(coded_ecg.bits, block_size)
            stats = codec.compression_stats(ecg, coded_ecg, decoded_ecg)

            n_samples = info.n_samples
            n_windows = info.n_windows
            n_measurements = info.n_measurements
            row = Row(record=record_num,
                mr=mr, m=m, n=n, d=d, q=q, b=block_size,
                w=n_windows, s=n_samples,
                u_bits=stats.u_bits, c_bits=stats.c_bits,
                cr=stats.cr, pss=stats.pss, 
                bpm=stats.bpm, bps=stats.bps,
                overhead=info.total_overhead,
                snr=stats.snr, prd=stats.prd, 
                nmse=stats.nmse, qs=stats.qs,
                rtime=stats.rtime)
            click.echo(row)
            all_stats.append(row)
        # save results after every mr
        df = pd.DataFrame(all_stats, columns=Row._fields)
        df.to_csv(destination)



if __name__ == '__main__':
    main()
