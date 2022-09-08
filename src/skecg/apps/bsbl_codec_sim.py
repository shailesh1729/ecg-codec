from .apputils import *

import timeit
import wfdb
import constriction

import jax
import numpy as np
import jax.numpy as jnp


# CR-Suite libraries
import cr.nimble as crn
import cr.nimble.dsp as crdsp
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.block.bsbl as bsbl

@click.command()
@click.argument('record_num', type=int)
@click.option('-n', default=1024, help='Window length')
@click.option('-m', default=512, help='Measurement length')
@click.option('-d', default=12, help='Ones in sensing matrix')
@click.option('-b', '--block-size', default=32, help='BSBL block size')
def main(record_num, n, m, d, block_size):

    # read the signal from the database
    mit_bih_dir = get_db_dir()
    path = f'{mit_bih_dir}/{record_num}'
    header = wfdb.rdheader(path)
    click.echo(f'Processing record: {record_num}: Sampling rate: {header.fs}')

    sampfrom=0
    sampto=30*360
    record = wfdb.rdrecord(f'{mit_bih_dir}/{record_num}', channels=[0]
        , sampfrom=sampfrom, sampto=sampto, physical=False)
    ecg = np.squeeze(record.d_signal)

    X = crn.vec_to_windows(ecg, n) - 1024
    n_samples = X.size
    n_windows = X.shape[1]
    print(f'n_samples: {n_samples}, n_windows: {n_windows}')

    Phi = crdict.sparse_binary_mtx(crn.KEY0, 
        m, n, d=d, normalize_atoms=False)

    # Measurements
    Y = Phi @ X


    # Entropy coding
    Y_np = np.array(Y).astype(int)
    Y2 = Y_np.flatten(order='F')
    n_measurements = Y2.size
    max_val = np.max(np.abs(Y2))
    mean_val = np.round(Y2.mean())
    std_val = np.round(Y2.std())

    model = constriction.stream.model.QuantizedGaussian(-max_val, max_val)
    encoder = constriction.stream.stack.AnsCoder()
    means = np.full(n_measurements, mean_val)
    stds = np.full(n_measurements, std_val)
    encoder.encode_reverse(Y2, model, means, stds)

    # Get and print the compressed representation:
    compressed = encoder.get_compressed()
    uncompressed_bits = n_samples * 11
    compressed_bits = len(compressed)*32
    ratio = uncompressed_bits / compressed_bits
    print(f'Uncompressed bits: {uncompressed_bits} Compressed bits: {compressed_bits}, ratio: {ratio:.2f}x')
    print(f'bits per measurement in compressed data: {compressed_bits  / n_measurements}')
    print(f'bits per measurement in cs measurements: {np.round(np.log2(2* max_val + 1))}')


    # Decode the message:
    decoder = constriction.stream.stack.AnsCoder(compressed)
    reconstructed = decoder.decode(model, means, stds)
    reconstructed = reconstructed
    print(f'matched entropy decoding: {np.all(reconstructed == Y2)}')

    # Arrange measurements into column vectors
    RY = crn.vec_to_windows(jnp.asarray(reconstructed, dtype=float), m)

    DPhi = Phi.todense()
    options = bsbl.bsbl_bo_options(max_iters=20)

    for i in range(n_windows):
        x = X[:, i]
        y = RY[:, i]
        start = timeit.default_timer()
        sol = bsbl.bsbl_bo_np_jit(DPhi, y, block_size, options=options)
        stop = timeit.default_timer()
        rtime = stop - start
        x_hat = sol.x
        snr = crn.signal_noise_ratio(x, x_hat)
        prd = crn.percent_rms_diff(x, x_hat)
        nmse = crn.normalized_mse(x, x_hat)
        print(f'[{i}] SNR: {snr:.2f} dB, PRD: {prd:.1f}%, NMSE: {nmse:.5f}, Time: {rtime:.2f} sec, Iters: {sol.iterations}')
