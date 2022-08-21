from .apputils import *
from matplotlib import pyplot as plt


@click.command()
@click.argument('record_num', type=int)
@click.argument('start_sample', type=int)
@click.option('--length', default=1024, help='ECG length in samples.')
@click.option('--transform', "-t", is_flag=True, help="Generate transform coefficients.")
@click.option('--wavelet', default='bior3.1', help="Wavelet to use for transform.")
def analyze(record_num, start_sample, length, 
    transform, wavelet):
    mit_bih_dir = get_db_dir()
    path = f'{mit_bih_dir}/{record_num}'
    header = wfdb.rdheader(path)
    click.echo(f'Processing record: {record_num}: Sampling rate: {header.fs}')
    fs = float(header.fs)
    # we should round to the nearest start of second
    # start_sec = start_sample // fs
    # sampfrom = fs * start_sec
    sampfrom = start_sample
    sampto = sampfrom + length 
    record = wfdb.rdrecord(path, channels=[0],
        sampfrom=sampfrom, sampto=sampto, physical=True)
    signal = np.squeeze(record.p_signal)
    click.echo(f'Start: {sampfrom}, To: {sampto}, Length: {len(signal)}')
    # time in sec
    start_sec = sampfrom / fs
    ts = np.arange(signal.size) / fs + start_sec
    min_ts = np.min(ts)
    max_ts = np.max(ts)
    min_ts = int(min_ts)
    max_ts = round(max_ts)
    major_ticks = np.arange(min_ts, max_ts+1)
    fig = plt.figure(figsize=(16, 9));
    ax = plt.axes();
    ax.plot(ts, signal);
    ax.set_xticks(major_ticks)
    # Turn on the minor ticks on
    ax.minorticks_on()
    # Make the major grid
    ax.grid(which='major', linestyle='-', color='red', linewidth='1.0')
    # Make the minor grid
    ax.grid(which='minor', linestyle=':', color='black', linewidth='0.5')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Amplitude (mV)')
    title = f'Rec ({record_num}) Lead II: Samples {sampfrom}-{sampto}'
    ax.set_title(title)
    identifier = f'{record_num}_{sampfrom}_{sampto}'
    name = f'ecg_signal_{identifier}'
    plt.savefig(f"{name}.png", dpi=150)
    plt.savefig(f"{name}.pdf", dpi=150)
    if not transform:
        return
    click.echo(f'Now computing wavelet coefficients using {wavelet}.')
    wavelet_obj = wt.to_wavelet(wavelet)
    block_len = len(signal)
    max_level = wt.dwt_max_level(block_len, wavelet_obj.dec_len)
    # wavelet transform operator
    PsiT = crs.lop.dwt(block_len, wavelet, max_level)
    # PsiT = crs.lop.jit(PsiT)
    coeffs = PsiT.times(signal)
    energy_levels  = [90, 95, 99, 99.5]
    ks = [crn.num_largest_coeffs_for_energy_percent(coeffs, energy_level)
     for energy_level in energy_levels]
    lks = zip(energy_levels, ks)
    k_text = '\n'.join(f'{l}% in {k} terms' for l, k in lks)
    for l, k in lks:
        click.echo(f'Energy percentage: {l}%, K: {k}')
    fig = plt.figure(figsize=(16, 9));
    ax = plt.axes();
    ax.plot(coeffs);
    title = f'Rec ({record_num}) : Samples {sampfrom}-{sampto} Wavelet: {wavelet}'
    ax.set_xlabel('Wavelet coefficient index')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    name = f'ecg_wavelet_{identifier}'
    plt.text(0.8, 0.2, k_text, horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)
    plt.savefig(f"{name}.png", dpi=150)
    plt.savefig(f"{name}.pdf", dpi=150)
