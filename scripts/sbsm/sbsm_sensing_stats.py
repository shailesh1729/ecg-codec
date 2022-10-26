from skecg.apps.apputils import *
from skecg.physionet import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from statsmodels.stats import descriptivestats

import cr.sparse.dict as crdict
import cr.sparse.lop as crlop

def build_sensor(key, m, n, d):
    Phi = crdict.sparse_binary_mtx(key, m, n, d,
        normalize_atoms=False)
    Phi = crlop.sparse_real_matrix(Phi)
    return Phi



@click.command()
@click.argument('m', type=int)
@click.argument('n', type=int)
@click.argument('d', type=int)
def main(m, n, d):
    destination = f'sbsm-{d}-measurements-{m}x{n}-stats.csv'
    mit_bih_dir = get_db_dir()
    record_nums = MIT_BIH['record_nums']
    all_stats = []
    sensor = build_sensor(key, m, n, d)
    Y_vecs = []
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
        Y_vecs.append(Y)
        # compute stats
        stats = descriptivestats.describe(Y_flat)
        stats = stats.T
        # add record number
        stats.insert(0, 'record', [record_num])
        click.echo(stats)
        all_stats.append(stats)
    Y_vecs = np.concatenate(Y_vecs, axis=1)
    # save results after all records
    df = pd.concat(all_stats)
    df = df.drop(columns=['missing'])
    df['nobs'] = df['nobs'].astype(int)
    df['min'] = df['min'].astype(int)
    df['max'] = df['max'].astype(int)
    df['iqr'] = df['iqr'].astype(int)
    df['range'] = df['range'].astype(int)
    df.to_csv(destination)

    # take indices to rows
    Y_vecs = Y_vecs.T
    df = pd.DataFrame(Y_vecs)

    means = df.mean()
    stds = df.std()
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.errorbar(np.arange(m), means, stds, ecolor='r')
    fig.savefig(f'sbsm-{d}-{m}x{n}-mvec-errorbar.png') 

if __name__ == '__main__':
    main()