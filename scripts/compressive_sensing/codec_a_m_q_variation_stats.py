from typing import NamedTuple
from skecg.apps.apputils import *
from skecg.physionet import *

import pandas as pd
from statsmodels.stats import descriptivestats

from skecg.cs.codec_a import (
    build_codec,
    compression_stats
    )


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
    q_mean: int
    "mean of quantized values"
    q_std: int
    "standard deviation of quantized values"
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
    snr: float
    "signal to noise ratio (dB)"
    prd: float
    "percent root mean square difference"
    nmse: float
    "normalized mean square difference"
    rtime: float

@click.command()
@click.argument('record_num', type=int)
@click.option('-n', default=512, help='Window length')
@click.option('-d', default=6, help='Ones in sensing matrix')
@click.option('-b', '--block-size', default=32, help='BSBL block size')
def main(record_num, n, d, block_size):
    destination = f'codec-a-n={n}-d={d}-m-q-var-stats.csv'
    mit_bih_dir = get_db_dir()
    path = f'{mit_bih_dir}/{record_num}'
    header = wfdb.rdheader(path)
    click.echo(f'Processing record: {record_num}: Sampling rate: {header.fs}')

    sampfrom=0
    # sampto = 10*360
    sampto=None
    record = wfdb.rdrecord(f'{mit_bih_dir}/{record_num}', channels=[0]
        , sampfrom=sampfrom, sampto=sampto, physical=False)
    ecg = np.squeeze(record.d_signal)

    mrs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    qs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_stats = []
    for mr in mrs:
        for q in qs:
            m = int(round(mr * n))
            print(f'mr={mr}, m={m}, q={q}')
            encoder, decoder = build_codec(n, m, d, block_size, q)
            coded_ecg = encoder(ecg)
            decoded_ecg = decoder(coded_ecg)
            stats = compression_stats(ecg, coded_ecg, decoded_ecg)
            n_samples = coded_ecg.n_samples
            n_windows = coded_ecg.n_windows
            n_measurements = coded_ecg.n_measurements
            row = Row(record=record_num,
                mr=mr, m=m, n=n, d=d, q=q, b=block_size,
                w=n_windows, s=n_samples,
                q_mean=stats.q_mean, q_std=stats.q_std,
                u_bits=stats.u_bits, c_bits=stats.c_bits,
                cr=stats.cr, pss=stats.pss, bpm=stats.bpm, bps=stats.bps,
                snr=stats.snr, prd=stats.prd, nmse=stats.nmse, rtime=stats.rtime)
            click.echo(row)
            all_stats.append(row)
        # save results after every mr
        df = pd.DataFrame(all_stats, columns=Row._fields)
        df.to_csv(destination)



if __name__ == '__main__':
    main()
