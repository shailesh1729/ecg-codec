# std imports
import math
from decimal import Decimal
from typing import NamedTuple

from skecg.apps.apputils import *
from skecg.physionet import *

import pandas as pd
from statsmodels.stats import descriptivestats

import skecg.cs.codec_b as codec
from skecg.util import kld_normal

class Row(NamedTuple):
    record: int
    "record number"
    m: int
    "measurement space dimension"
    n: int
    "signal space dimension"
    d: int
    "number of ones per column in the sensing matrix"
    b: int
    "block size for BSBL"
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
@click.option('-n', default=512, help='Window length')
@click.option('-m', default=256, help='Measurement length')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option('-q', default=1, help='Quantization nmse factor')
@click.option('-c', default=2, help='Clipping nmse factor')
@click.option('-w', default=16, help='Windows per frame')
@click.option('-b', '--block-size', default=32, help='BSBL block size')
@click.option("--dry", is_flag=True, 
    show_default=True, default=False, help="Dry run with small samples")
def main(n, m, d, q, c, w, block_size, dry):
    destination = f'codec-b-m={m}-n={n}-d={d}-q={q}-c={c}-b={block_size}-stats.csv'

    q_nmse_limit = Decimal((0, (q,), -2))
    c_nmse_limit = Decimal((0, (c,), -2))
    params = codec.EncoderParams(key=crn.KEYS[0], 
        n=n, m=m, d=d, w=w, adaptive=True,
        q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)

    mit_bih_dir = get_db_dir()
    record_nums = MIT_BIH['record_nums']
    all_stats = []

    sampfrom=0
    if dry:
        sampto=10*360
    else:
        sampto=None

    for record_num in record_nums:
        click.echo(f'Processing: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0]
            , sampfrom=sampfrom, sampto=sampto, physical=False)
        # data
        ecg = np.squeeze(record.d_signal) - int(record.baseline[0])
        # compression
        coded_ecg = codec.encode(params, ecg)
        # decompression
        decoded_ecg = codec.decode(coded_ecg.bits, block_size)
        # encoding info
        info = coded_ecg.info
        print(info)
        q_vals = info.q_vals
        rng_mults = info.rng_mults
        qc_snrs = info.qc_snrs

        # stats
        stats = codec.compression_stats(ecg, coded_ecg, decoded_ecg)

        row = Row(record=record_num, 
            m=m, n=n, d=d, b=block_size,
            w=info.n_windows, 
            f=info.n_frames, s=info.n_samples,
            overhead_bits=info.overhead_bits,
            overhead=info.total_overhead,
            u_bits=stats.u_bits,
            c_bits=stats.c_bits,
            cr=stats.cr, pss=stats.pss,
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
    # save results after all records
    df = pd.DataFrame(all_stats, columns=Row._fields)
    df.to_csv(destination)


if __name__ == '__main__':
    main()
