from .apputils import *

@click.command()
@click.argument('record_num', type=int)
@click.argument('energy_level', type=float)
@click.option('--length', default=1024, help='Block length in samples.')
@click.option('--wavelet', default='bior3.1', help="Wavelet to use for transform.")
def main(record_num, energy_level, length, wavelet):

    # read the signal from the database
    mit_bih_dir = get_db_dir()
    path = f'{mit_bih_dir}/{record_num}'
    header = wfdb.rdheader(path)
    click.echo(f'Processing record: {record_num}: Sampling rate: {header.fs}')
    record = wfdb.rdrecord(f'{mit_bih_dir}/{record_num}', channels=[0])
    signal = np.squeeze(record.p_signal)

    # prepare the wavelet transform operator
    wavelet_name = wavelet
    wavelet = wt.to_wavelet(wavelet)
    family = wavelet.short_name
    max_level = wt.dwt_max_level(length, wavelet.dec_len)
    print(f'{wavelet_name}: {max_level}')
    PsiT = crs.lop.dwt(length, wavelet_name, max_level)
    PsiT = crs.lop.jit(PsiT)

    # Divide signal into blocks
    X = crn.vec_to_windows(signal, length)
    blocks = X.shape[1]
    A = PsiT.times(X)
    ks = jnp.apply_along_axis(
        lambda a : crn.num_largest_coeffs_for_energy_percent(a, energy_level), 0, A)
    indices = jnp.argsort(ks)
    n = 5
    first_n = indices[:n]
    last_n = indices[-n:]
    click.echo('\n\nEasiest blocks:')
    for ind in first_n:
        start_sample = ind * length
        k = ks[ind]
        click.echo(f'start: {start_sample}, k: {k}')
    click.echo('\n\nHardest blocks:')
    for ind in last_n:
        start_sample = ind * length
        k = ks[ind]
        click.echo(f'start: {start_sample}, k: {k}')