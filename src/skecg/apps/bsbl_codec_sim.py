from .apputils import *

import wfdb

import jax
import numpy as np
import jax.numpy as jnp


# CR-Suite libraries
import cr.nimble as crn
import cr.nimble.dsp as crdsp
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.block.bsbl as bsbl

from skecg.cs.codec_a import build_codec


@click.command()
@click.argument('record_num', type=int)
@click.option('-n', default=1024, help='Window length')
@click.option('-m', default=512, help='Measurement length')
@click.option('-d', default=12, help='Ones in sensing matrix')
@click.option('-b', '--block-size', default=32, help='BSBL block size')
@click.option('-q', '--q-bits', default=0, help='Quantization by bits')
def main(record_num, n, m, d, block_size, q_bits):
    encoder, decoder = build_codec(n, m, d, block_size, q_bits)

    # read the signal from the database
    mit_bih_dir = get_db_dir()
    path = f'{mit_bih_dir}/{record_num}'
    header = wfdb.rdheader(path)
    click.echo(f'Processing record: {record_num}: Sampling rate: {header.fs}')

    sampfrom=0
    sampto=None
    record = wfdb.rdrecord(f'{mit_bih_dir}/{record_num}', channels=[0]
        , sampfrom=sampfrom, sampto=sampto, physical=False)
    ecg = np.squeeze(record.d_signal)

    coded_ecg = encoder(ecg)
    n_samples = coded_ecg.n_samples
    n_windows = coded_ecg.n_windows
    n_measurements = coded_ecg.n_measurements
    y = coded_ecg.measurements
    g_max = np.max(np.abs(y))

    uncompressed_bits = n_samples * 11
    compressed_bits = len(coded_ecg.compressed)*32
    ratio = uncompressed_bits / compressed_bits
    print(f'Uncompressed bits: {uncompressed_bits} Compressed bits: {compressed_bits}, ratio: {ratio:.2f}x')
    print(f'bits per measurement in compressed data: {compressed_bits  / coded_ecg.n_measurements}')
    print(f'bits per measurement in cs measurements: {np.round(np.log2(2* g_max + 1))}')
    print(f'Compressed bits per sample: {compressed_bits/coded_ecg.n_samples:.2f}')


    decoded_ecg = decoder(coded_ecg)
    rtime = decoded_ecg.total_time

    x = ecg[:coded_ecg.n_samples]
    x_hat = decoded_ecg.x
    snr = crn.signal_noise_ratio(x, x_hat)
    prd = crn.percent_rms_diff(x, x_hat)
    nmse = crn.normalized_mse(x, x_hat)
    print(f'SNR: {snr:.2f} dB, PRD: {prd:.1f}%, NMSE: {nmse:.5f}, Time: {rtime:.2f} sec')
