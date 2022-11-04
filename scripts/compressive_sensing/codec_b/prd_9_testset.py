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


# Optimal PMS reported for BSBL in Zhang et al. CSNet
BSBL_TEST_SET = [
(100, 71),
(101, 71),
(102, 70),
(107, 74),
(109, 75),
(111, 69),
(115, 68),
(117, 74),
(118, 76),
(119, 71),
]


class Row(NamedTuple):
    record: int
    "record number"
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
@click.option('-w', default=64, help='Windows per frame')
@click.option('-b', '--block-size', default=32, help='BSBL block size')
@click.option("--dry", is_flag=True, 
    show_default=True, default=False, help="Dry run with small samples")
def main(n, d, q, c, w, block_size, dry):
    destination = f'codec_b_target_prd_bsbl_n-{n}_d-{d}-stats.csv'

    mit_bih_dir = get_db_dir()
    q_nmse_limit = Decimal((0, (q,), -2))
    c_nmse_limit = Decimal((0, (c,), -2))
    all_stats = []

    sampfrom=0
    if dry:
        sampto=10*360
    else:
        sampto=None

    TARGET_PRD = 9
    for example in BSBL_TEST_SET:
        record_num, optimal_pms = example
        click.echo(f'Processing record: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0]
            , sampfrom=sampfrom, sampto=sampto, physical=False)
        # data
        ecg = np.squeeze(record.d_signal) - int(record.baseline[0])
        mr = 100 - optimal_pms
        m = math.ceil(n * mr / 100)
        CURRENT_PRD = 100
        while CURRENT_PRD > TARGET_PRD:
            params = codec.EncoderParams(key=crn.KEY0, 
                n=n, m=m, d=d, w=w, adaptive=True,
                q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
            # compression
            coded_ecg = codec.encode(params, ecg)
            # decompression
            decoded_ecg = codec.decode(coded_ecg.bits, block_size)
            # encoding info
            info = coded_ecg.info
            click.echo(info)
            # stats
            stats = codec.compression_stats(ecg, coded_ecg, decoded_ecg)
            row = Row(record=record_num, 
                m=m, n=n, d=d,
                u_bits=stats.u_bits,
                c_bits=stats.c_bits,
                cr=stats.cr, pss=stats.pss,
                snr=stats.snr, prd=stats.prd, 
                rtime=stats.rtime,
                )
            click.echo(row)
            CURRENT_PRD = stats.prd
            m = m + 2
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
