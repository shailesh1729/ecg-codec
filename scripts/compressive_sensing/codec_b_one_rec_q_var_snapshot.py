# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

from decimal import Decimal
import jax
import cr.nimble as crn
import cr.sparse as crs

import matplotlib.pyplot as plt

import numpy as np
import wfdb
from statsmodels.stats import descriptivestats
import cr.sparse.plots as crplot

from skecg.physionet import *
from skecg.apps.apputils import *
import skecg.cs.codec_b as codec


@click.command()
@click.argument('record_num', type=int)
@click.option('-n', default=512, help='Window length')
@click.option('-m', default=256, help='Measurement length')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option('-c', default=2, help='Clipping nmse factor')
@click.option('-w', default=16, help='Windows per frame')
@click.option('-b', '--block-size', default=32, help='BSBL block size')
def main(record_num, n, m, d, c, w, block_size):
    mit_bih_dir = get_db_dir()
    path = f'{mit_bih_dir}/{record_num}'
    header = wfdb.rdheader(path)
    click.echo(f'Processing record: {record_num}: Sampling rate: {header.fs}')
    fs = float(header.fs)
    sampfrom=0
    sampto=512*6
    header = wfdb.rdheader(path)
    record = wfdb.rdrecord(path, channels=[0]
        , sampfrom=sampfrom, sampto=sampto, physical=False)
    # data
    ecg = np.squeeze(record.d_signal) - int(record.baseline[0])

    x_hats = []
    qs = []
    q_stats = []
    q_nmse_limit = Decimal((0, (0,), -2))
    c_nmse_limit = Decimal((0, (c,), -2))
    for q_bits in [0, 2, 4, 5, 6, 7]:
        params = codec.EncoderParams(key=crn.KEY0, n=n, m=m, d=d, w=w, 
            adaptive=False,
            q=q_bits, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)
        coded_ecg = codec.encode(params, ecg)
        info = coded_ecg.info
        decoded_ecg = codec.decode(coded_ecg.bits, block_size)
        stats = codec.compression_stats(ecg, coded_ecg, decoded_ecg)
        x_hat = decoded_ecg.x
        x_hats.append(x_hat)
        qs.append(q_bits)
        q_stats.append(stats)
    nq = len(qs)
    ax = crplot.h_plots(nq+1, height=1.5)
    ax[0].plot(ecg)
    ax[0].set_title(f'sample from record {record_num}')
    for i in range(nq):
        x_hat = x_hats[i]
        q = qs[i]
        stat = q_stats[i]
        title = f'q={q}, snr={stat.snr:.1f} dB, prd={stat.prd:.1f} %, bps={stat.bps: .2f}, cr={stat.cr:.1f}, qs={stat.qs:.1f}'
        # ax[i+1].plot(ecg)
        ax[i+1].plot(x_hat)
        ax[i+1].set_title(title)
    plt.savefig(f"rec_{record_num}_q_cr_prd_qs.png", dpi=150)

if __name__ == '__main__':
    main()
