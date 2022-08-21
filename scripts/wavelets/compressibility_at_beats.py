
from skecg.apps.apputils import *
from skecg.physionet import *

from typing import NamedTuple
import pandas as pd


class Row(NamedTuple):
    record: int
    beat: int
    sym: str
    pos: int
    block: int
    k: int


@click.command()
@click.option('--energy', '-e', default=99, help='Percentage of energy for thresholding.')
@click.option('--wavelet', default='bior3.1', help="Wavelet to use for transform.")
@click.option('--length', default=1024, help='Block length in samples.')
def main(energy, wavelet, length):
    experiment_name = f'beat_sparsity_size_{length}_energy_{energy:.2f}'
    destination = f'{experiment_name}.csv'
    # read the signal from the database
    mit_bih_dir = get_db_dir()
    record_nums = MIT_BIH['record_nums']

    # prepare the wavelet transform operator
    wavelet_name = wavelet
    wavelet = wt.to_wavelet(wavelet)
    family = wavelet.short_name
    max_level = wt.dwt_max_level(length, wavelet.dec_len)
    print(f'{wavelet_name}: {max_level}')
    PsiT = crs.lop.dwt(length, wavelet_name, max_level)
    PsiT = crs.lop.jit(PsiT)

    data = []
    for record_num in record_nums:
        click.echo(f'Processing: {record_num}')
        path = f'{mit_bih_dir}/{record_num}'
        header = wfdb.rdheader(path)
        record = wfdb.rdrecord(path, channels=[0])
        signal = np.squeeze(record.p_signal)

        click.echo('Reading annotations.')
        ann = wfdb.rdann(path, 'atr')
        symbols = ann.symbol
        positions = ann.sample
        click.echo(f'Number of annotations: {len(symbols)}')

        click.echo('Computing wavelet coefficients for each block.')
        # Divide signal into blocks
        X = crn.vec_to_windows(signal, length)
        # Number of blocks
        blocks = X.shape[1]
        A = PsiT.times(X)
        # Sparsity for each block
        ks = jnp.apply_along_axis(
            lambda a : crn.num_largest_coeffs_for_energy_percent(a, energy), 0, A)

        click.echo('Matching annotations to blocks.')
        for i, sym in enumerate(symbols):
            # sample number of beat
            pos = positions[i]
            # block number
            blk = pos // length
            k = ks[blk]
            k = int(k)
            row = Row(record=record_num, beat=i, sym=sym, pos=pos,
                block=blk, k=k)
            # click.echo(row)
            data.append(row)
        # save results after each record
        # df = pd.DataFrame(data, columns=Row._fields)
        # df.to_csv(destination)
    # save results after all records
    df = pd.DataFrame(data, columns=Row._fields)
    df.to_csv(destination)


if __name__ == '__main__':
    main()