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
    w: int
    "number of windows"
    s: int
    "number of samples"
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
    y_kld: float

@click.command()
@click.option('-n', default=512, help='Window length')
@click.option('-m', default=256, help='Measurement length')
@click.option('-d', default=4, help='Ones in sensing matrix')
@click.option("--dry", is_flag=True, 
    show_default=True, default=False, help="Dry run with small samples")
def main(n, m, d, dry):
    destination = f'codec-b-m={m}-n={n}-d={d}-measurements.csv'

    q_nmse_limit = Decimal((0, (1,), -2))
    c_nmse_limit = Decimal((0, (1,), -2))
    params = codec.EncoderParams(key=crn.KEY0,
        n=n, m=m, d=d, w=32, adaptive=True,
        q=0, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit)

    # sensing matrix
    Phi = codec.build_sensor(params)

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
        # measurements
        y = codec.sense(params, Phi, ecg)
        n_measurements = y.size
        n_windows = n_measurements // params.m
        n_samples = params.n * n_windows

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
        y_kld = kld_normal(y)

        row = Row(record=record_num, 
            m=m, n=n, d=d,
            w=n_windows, 
            s=n_samples,
            y_min=y_min, y_max=y_max, y_range=y_range, y_iqr=y_iqr, 
            y_mean=y_mean, y_std=y_std, y_median=y_median, y_mad=y_mad, 
            y_skew=y_skew, y_kurtosis=y_kurtosis, y_jbera=y_jbera, y_jb_pval=y_jb_pval,
            y_kld=y_kld,
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
