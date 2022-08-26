from skecg.apps.apputils import *
from skecg.physionet import *

import pandas as pd
from statsmodels.stats import descriptivestats

@click.command()
def main():
    mit_bih_dir = get_db_dir()
    record_nums = MIT_BIH['record_nums']
    all_stats = []
    for record_num in record_nums:
        click.echo(f'Processing: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0], physical=False)
        signal = np.squeeze(record.d_signal)
        stats = descriptivestats.describe(signal)
        stats = stats.T
        stats.insert(0, 'record', [record_num])
        click.echo(stats)
        all_stats.append(stats)
    # save results after all records
    df = pd.concat(all_stats)
    df.to_csv('mit-bih-stats.csv')


if __name__ == '__main__':
    main()