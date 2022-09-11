
# std imports
from typing import NamedTuple, List
import constriction
import timeit

import numpy as np
import jax.numpy as jnp

# CR-Suite libraries
import cr.nimble as crn
import cr.nimble.dsp as crdsp
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.block.bsbl as bsbl


class EncodedData(NamedTuple):
    n_samples: int
    n_windows: int
    n_measurements: int
    mean_val: int
    std_val: int
    compressed: List[int]
    measurements: np.ndarray

class DecodedData(NamedTuple):
    x: np.ndarray
    r_times: np.ndarray
    r_iters: np.ndarray

    @property
    def total_time(self):
        return np.sum(self.r_times)


class CompressionStats(NamedTuple):
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


def compression_stats(ecg, coded_ecg, decoded_ecg):
    n_samples = coded_ecg.n_samples
    n_windows = coded_ecg.n_windows
    n_measurements = coded_ecg.n_measurements
    y = coded_ecg.measurements
    uncompressed_bits = n_samples * 11
    compressed_bits = len(coded_ecg.compressed)*32
    ratio = crn.compression_ratio(uncompressed_bits, compressed_bits)
    pss = crn.percent_space_saving(uncompressed_bits, compressed_bits)
    bpm = compressed_bits  / coded_ecg.n_measurements
    bps = compressed_bits/coded_ecg.n_samples
    y_max = np.max(np.abs(y))
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
    return CompressionStats(
            q_mean=coded_ecg.mean_val,
            q_std=coded_ecg.std_val,
            u_bits=uncompressed_bits,
            c_bits=compressed_bits,
            cr=ratio, pss=pss,
            bpm=bpm, bps=bps,
            snr=float(snr), prd=float(prd), nmse=float(nmse), rtime=rtime
        )

def build_codec(n, m, d, block_size, q_bits):

    n_sigma = 3
    Phi = crdict.sparse_binary_mtx(crn.KEY0, 
        m, n, d=d, normalize_atoms=False)
    DPhi = Phi.todense()

    def ecg_encoder(ecg):
        X = crn.vec_to_windows(ecg, n) - 1024
        n_samples = X.size
        n_windows = X.shape[1]
        print(f'n_samples: {n_samples}, n_windows: {n_windows}')
        # Measurements
        Y = Phi @ X
        # Convert to numpy
        Y_np = np.array(Y).astype(int)
        y = Y_np.flatten(order='F')
        Y2 = y
        # quantization
        if q_bits > 0:
            Y2 = Y2 >> q_bits
        # Entropy coding
        n_measurements = Y2.size
        max_val = np.max(Y2)
        min_val = np.min(Y2)
        mean_val = int(np.round(Y2.mean()))
        std_val = int(np.round(Y2.std()))
        print(f'min: {min_val}, max: {max_val}, mean: {mean_val}, std: {std_val}')
        g_max = max(np.abs(max_val), np.abs(min_val))
        a_min = int(mean_val - n_sigma * std_val)
        a_max = int(mean_val + n_sigma * std_val)
        Y2 = np.clip(Y2, a_min, a_max)

        # model = constriction.stream.model.QuantizedGaussian(min_val, max_val)
        model = constriction.stream.model.QuantizedGaussian(a_min, a_max)
        encoder = constriction.stream.stack.AnsCoder()
        means = np.full(n_measurements, mean_val * 1.)
        stds = np.full(n_measurements, std_val * 1.)
        encoder.encode_reverse(Y2, model, means, stds)

        # Get and print the compressed representation:
        compressed = encoder.get_compressed()
        return  EncodedData(
            n_samples=n_samples, n_windows=n_windows,
            n_measurements=n_measurements,
            mean_val=mean_val, std_val=std_val,
            compressed=compressed, measurements=y)


    def ecg_decoder(coded_ecg, n_windows=None):
        n_samples = coded_ecg.n_samples
        n_windows = coded_ecg.n_windows if n_windows is None else n_windows
        n_measurements = coded_ecg.n_measurements
        mean_val = coded_ecg.mean_val
        std_val = coded_ecg.std_val
        a_min = int(mean_val - n_sigma * std_val)
        a_max = int(mean_val + n_sigma * std_val)
        model = constriction.stream.model.QuantizedGaussian(a_min, a_max)
        compressed = coded_ecg.compressed
        means = np.full(n_measurements, mean_val * 1.)
        stds = np.full(n_measurements, std_val * 1.)
        # Decode the message:
        ans_decoder = constriction.stream.stack.AnsCoder(compressed)
        reconstructed = ans_decoder.decode(model, means, stds)

        # inverse quantization
        if q_bits > 0:
            reconstructed = reconstructed << q_bits
        # Arrange measurements into column vectors
        RY = crn.vec_to_windows(jnp.asarray(reconstructed, dtype=float), m)
        options = bsbl.bsbl_bo_options(max_iters=20)

        X_hat = np.zeros((n, n_windows))
        r_times = np.zeros(n_windows)
        r_iters = np.zeros(n_windows, dtype=int)

        for i in range(n_windows):
            y = RY[:, i]
            start = timeit.default_timer()
            sol = bsbl.bsbl_bo_np_jit(DPhi, y, block_size, options=options)
            stop = timeit.default_timer()
            rtime = stop - start
            x_hat = sol.x
            X_hat[:, i] = x_hat
            r_times[i] = rtime
            r_iters[i] = sol.iterations
            print(f'[{i}/{n_windows}], time: {rtime:.2f} sec')
        x = X_hat.flatten(order='F') + 1024
        return DecodedData(x=x, r_times=r_times, r_iters=r_iters)
    return ecg_encoder, ecg_decoder
