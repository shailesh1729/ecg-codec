from skecg.apps.apputils import *
from skecg.physionet import *

import pandas as pd
from statsmodels.stats import descriptivestats

from skecg.cs.sensor import build_sensor


@click.command()
@click.argument('m', type=int)
@click.argument('n', type=int)
def main(m, n):
    destination = f'cs-measurements-{m}x{n}-stats.csv'
    mit_bih_dir = get_db_dir()
    record_nums = MIT_BIH['record_nums']
    all_stats = []
    sensor = build_sensor(key, m, n)
    for record_num in record_nums:
        click.echo(f'Processing: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0], physical=False)
        signal = np.squeeze(record.d_signal)
        # adjust the zero level
        signal = signal - int(record.baseline[0])
        # convert to blocks
        X = crn.vec_to_windows(signal, n).astype(jnp.int32)
        # sense measurement vectors
        Y = sensor.times(X)
        # flatten measurement values
        Y_flat = np.array(Y.flatten(order='F'))
        # compute stats
        stats = descriptivestats.describe(Y_flat)
        stats = stats.T
        # add record number
        stats.insert(0, 'record', [record_num])
        click.echo(stats)
        all_stats.append(stats)
    # save results after all records
    df = pd.concat(all_stats)
    df = df.drop(columns=['missing'])
    df['nobs'] = df['nobs'].astype(int)
    df['min'] = df['min'].astype(int)
    df['max'] = df['max'].astype(int)
    df['iqr'] = df['iqr'].astype(int)
    df['range'] = df['range'].astype(int)
    df.to_csv(destination)


if __name__ == '__main__':
    main()