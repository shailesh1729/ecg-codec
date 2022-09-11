from typing import NamedTuple
from skecg.apps.apputils import *
from skecg.physionet import *

import pandas as pd
from statsmodels.stats import descriptivestats

from skecg.cs.codec_a import build_codec

class Row(NamedTuple):
    record: int
    "record number"
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
    "reconstruction time"
    y_min: int
    y_max: int
    y_range: int
    y_iqr: int
    y_mean: float
    y_std: float
    y_median: int
    y_mad: float
    y_skew: float
    y_kurtosis: float
    y_jbera: float
    y_jb_pval: float


@click.command()
@click.option('-n', default=512, help='Window length')
@click.option('-m', default=256, help='Measurement length')
@click.option('-d', default=6, help='Ones in sensing matrix')
@click.option('-b', '--block-size', default=32, help='BSBL block size')
@click.option('-q', '--q-bits', default=0, help='Quantization by bits')
def main(n, m, d, block_size, q_bits):
    destination = f'codec-a-m={m}-n={n}-d={d}-q={q_bits}-stats.csv'
    mit_bih_dir = get_db_dir()
    record_nums = MIT_BIH['record_nums']
    all_stats = []

    encoder, decoder = build_codec(n, m, d, block_size, q_bits)
    sampfrom=0
    # sampto=10*360
    sampto=None

    for record_num in record_nums:
        click.echo(f'Processing: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0]
            , sampfrom=sampfrom, sampto=sampto, physical=False)
        # data
        ecg = np.squeeze(record.d_signal)
        # compression
        coded_ecg = encoder(ecg)
        # decompression
        decoded_ecg = decoder(coded_ecg)

        n_samples = coded_ecg.n_samples
        n_windows = coded_ecg.n_windows
        n_measurements = coded_ecg.n_measurements
        y = coded_ecg.measurements
        # compute stats
        stats = descriptivestats.describe(y)
        y_min = stats.loc['min'][0]
        y_max = stats.loc['max'][0]
        y_range = stats.loc['range'][0]
        y_mean = stats.loc['mean'][0]
        y_std = stats.loc['std'][0]
        y_median = stats.loc['median'][0]
        y_iqr = stats.loc['iqr'][0]
        y_mad = stats.loc['mad'][0]
        y_skew = stats.loc['skew'][0]
        y_kurtosis = stats.loc['kurtosis'][0]
        y_jbera = stats.loc['jarque_bera'][0]
        y_jb_pval = stats.loc['jarque_bera_pval'][0]

        uncompressed_bits = n_samples * 11
        compressed_bits = len(coded_ecg.compressed)*32
        ratio = crn.compression_ratio(uncompressed_bits, compressed_bits)
        pss = crn.percent_space_saving(uncompressed_bits, compressed_bits)
        bpm = compressed_bits  / coded_ecg.n_measurements
        bps = compressed_bits/coded_ecg.n_samples
        print(f'Uncompressed bits: {uncompressed_bits} Compressed bits: {compressed_bits}, ratio: {ratio:.2f}x')
        print(f'bits per measurement in compressed data: {bpm:.2f}')
        print(f'bits per measurement in cs measurements: {np.round(np.log2(2* y_max + 1))}')
        print(f'Compressed bits per sample: {bps:.2f}')


        rtime = decoded_ecg.total_time

        x = ecg[:coded_ecg.n_samples]
        x_hat = decoded_ecg.x
        snr = crn.signal_noise_ratio(x, x_hat)
        prd = crn.percent_rms_diff(x, x_hat)
        nmse = crn.normalized_mse(x, x_hat)
        print(f'SNR: {snr:.2f} dB, PRD: {prd:.1f}%, NMSE: {nmse:.5f}, Time: {rtime:.2f} sec')

        row = Row(record=record_num, 
            m=m, n=n, d=d, q=q_bits, b=block_size,
            w=n_windows, s=n_samples,
            q_mean=coded_ecg.mean_val,
            q_std=coded_ecg.std_val,
            u_bits=uncompressed_bits,
            c_bits=compressed_bits,
            cr=ratio, pss=pss,
            bpm=bpm, bps=bps,
            snr=float(snr), prd=float(prd), nmse=float(nmse), rtime=rtime,
            y_min=y_min, y_max=y_max, y_range=y_range, y_iqr=y_iqr, 
            y_mean=y_mean, y_std=y_std, y_median=y_median, y_mad=y_mad, 
            y_skew=y_skew, y_kurtosis=y_kurtosis, y_jbera=y_jbera, y_jb_pval=y_jb_pval
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