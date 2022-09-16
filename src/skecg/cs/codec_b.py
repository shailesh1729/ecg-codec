"""
This codec supports adaptive quantization
"""


# std imports
import sys
import math
from decimal import Decimal
from typing import NamedTuple, List
import timeit

# NumPy
import jax
import numpy as np
import jax.numpy as jnp


# CR-Suite libraries
import cr.nimble as crn
import cr.nimble.dsp as crdsp
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.block.bsbl as bsbl


# Bitarray
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from cr.nimble.compression import *

# Entropy coding
import constriction

# Constants for Encoder Design
N_BITS = 12
M_BITS = 12
D_BITS = 6
W_BITS = 8
Q_MAX = 6
Q_MIN = 0

SEC_MEAN_BITS = 16
SEC_STD_BITS = 16
SEC_Q_BITS = 3
SEC_RNG_BITS = 3
SEC_WORD_BITS = 16


######################################################################
#                       ENCODER
######################################################################

class EncoderParams(NamedTuple):
    key: jax.numpy.ndarray
    "PRNG Key for generating sensing matrix"
    n: int
    "number of samples per window"
    m : int
    "number of measurements per window"
    d: int
    "Number of ones per column in sparse binary sensing matrix"
    w: int
    "number of windows in each section of the signal"
    adaptive: bool    
    "A flag to indicate if the quantization is adaptive"
    q: int
    "quantization parameter if the quantization is fixed"
    q_nmse_limit: Decimal
    "NMSE limit for the quantization step"
    c_nmse_limit: Decimal
    "NMSE limit for the clipping step"

    @property
    def section_length(self):
        return self.n * self.w

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if not jnp.all(self.key == other.key):
                return False
            if not self.n == other.n:
                return False
            if not self.m == other.m:
                return False
            if not self.d == other.d:
                return False
            if not self.w == other.w:
                return False
            if not self.adaptive == other.adaptive:
                return False
            if self.adaptive:
                if not self.q_nmse_limit == other.q_nmse_limit:
                    return False
            else:
                if not self.q == other.q:
                    return False
            if not self.c_nmse_limit == other.c_nmse_limit:
                return False
            return True
        else:
            return False

class EncodedSection(NamedTuple):
    "Information about each encoded section"
    n_measurements: int
    n_windows: int
    max_val: int
    min_val: int
    mean_val: int
    std_val: int
    q : int
    rng_mult: int
    n_words : int
    q_nmse: float
    c_nmse: float
    qc_nmse: float
    qc_snr: float

class EncodedStream(NamedTuple):
    n_samples: int
    n_windows: int
    n_sections: int
    n_measurements: int
    sections: List[EncodedSection]

class EncodedData(NamedTuple):
    info: EncodedStream
    y: np.ndarray
    bits: bitarray

def serialize_encoder_params(params: EncoderParams):
    a = bitarray()
    key = params.key.to_py()
    a.extend(int2ba(int(key[0]), 32))
    a.extend(int2ba(int(key[1]), 32))
    a.extend(int2ba(params.n, N_BITS))
    a.extend(int2ba(params.m, M_BITS))
    a.extend(int2ba(params.d, D_BITS))
    a.extend(int2ba(params.w, W_BITS))
    a.append(params.adaptive)
    if not params.adaptive:
        a.append(params.q, 4)
    else:
        s, digits, exp = params.q_nmse_limit.as_tuple()
        a.extend(int2ba(digits[0], 4))
        a.extend(int2ba(-exp, 4))
    # encoded the clipping limit
    s, digits, exp = params.c_nmse_limit.as_tuple()
    a.extend(int2ba(digits[0], 4))
    a.extend(int2ba(-exp, 4))
    return a




def build_sensor(params: EncoderParams):
    Phi = crdict.sparse_binary_mtx(params.key, 
        params.m, params.n, d=params.d, normalize_atoms=False)
    return Phi
    
def sense(params, Phi, ecg):
    X = crn.vec_to_windows(ecg, params.n)
    # Measurements
    Y = Phi @ X
    # Convert to numpy
    Y_np = np.array(Y).astype(int)
    y = Y_np.flatten(order='F')
    return y

def encode(params: EncoderParams, ecg: np.ndarray):
    """Encodes ECG data into a bitstream
    """
    stream = bitarray()
    stream.extend(serialize_encoder_params(params))
    # fill to the multiple of 8
    stream.fill()
    # sensing matrix
    Phi = build_sensor(params)
    # measurements
    y = sense(params, Phi, ecg)
    n_measurements = y.size
    n_windows = n_measurements // params.m
    n_samples = params.n * n_windows
    # compute number of sections
    n_sections = math.ceil(n_windows / params.w)
    # length of each section of measurements
    sl = params.m * params.w
    # work section by section
    start = 0
    sections = []
    for i_sec in range(n_sections):
        print(f'Encoding section {i_sec}')
        sec_info, bits = encode_section(params, y[start:start+sl])
        start += sl
        stream.extend(bits)
        sections.append(sec_info)
    info = EncodedStream(n_samples=n_samples, n_windows=n_windows,
        n_sections=n_sections, n_measurements=n_measurements,
        sections=sections)
    data = EncodedData(info=info, y=y, bits=stream)
    return data


def encode_section(params: EncoderParams, y: np.ndarray):
    n_measurements=len(y)
    n_windows = n_measurements // params.m
    q_nmse_limit = float(params.q_nmse_limit)
    c_nmse_limit = float(params.c_nmse_limit)
    q = params.q
    if params.adaptive:
        for q in range(Q_MAX, Q_MIN, -1):
            yq = y >> q
            yhat = yq << q
            q_nmse = crn.normalized_root_mse(y, yhat)
            if q_nmse <= q_nmse_limit:
                # we have achieved the desired quantization
                break
    else:
        yq = y >> q
        q_nmse = crn.normalized_root_mse(y, yhat)
    max_val = np.max(yq)
    min_val = np.min(yq)
    mean_val = int(np.round(yq.mean()))
    std_val = int(np.ceil(yq.std()))
    # make sure that std-val is positive
    std_val = std_val if std_val > 0 else 1
    s_max = max(np.abs(max_val), np.abs(min_val))

    for rng_mult in range(3,8):
        a_min = int(mean_val - rng_mult * std_val)
        a_max = int(mean_val + rng_mult * std_val)
        yc = np.clip(yq, a_min, a_max)
        c_nmse = crn.normalized_root_mse(yq, yc)
        if c_nmse <= c_nmse_limit:
            break
    # Measure the overall SNR
    yhat = yc << q
    qc_nmse = crn.normalized_root_mse(y, yhat)
    qc_snr = crn.signal_noise_ratio(y, yhat)
    model = constriction.stream.model.QuantizedGaussian(a_min, a_max,
        mean=mean_val, std=std_val)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(yc, model)
    compressed = encoder.get_compressed()
    # number of words of compressed bits
    n_words = len(compressed)
    # start encoding
    stream = bitarray()
    stream.extend(int2ba(mean_val, SEC_MEAN_BITS, signed=True))
    stream.extend(int2ba(std_val, SEC_STD_BITS))
    stream.extend(int2ba(q, SEC_Q_BITS))
    stream.extend(int2ba(rng_mult, SEC_RNG_BITS))
    stream.extend(int2ba(n_windows, W_BITS))
    stream.extend(int2ba(n_words, SEC_WORD_BITS))
    stream.fill()
    for word in compressed:
        # print(word)
        stream.extend(int2ba(int(word), 32))
    info = EncodedSection(n_measurements=n_measurements,
        n_windows=n_windows,
        max_val=max_val, min_val=min_val,
        mean_val=mean_val, std_val=std_val,
        q=q, rng_mult=rng_mult, n_words=n_words,
        q_nmse=float(q_nmse), c_nmse=float(c_nmse), 
        qc_nmse=float(qc_nmse), qc_snr=float(qc_snr))
    return info, stream


######################################################################
#                       DECODER
######################################################################

def deserialize_encoder_params(bits: bitarray, pos=0):
    key0 = ba2int(bits[pos:pos+32])
    pos = 32
    key1 = ba2int(bits[pos:pos+32])
    key = jnp.array([key0, key1], dtype=jnp.uint32)
    pos += 32
    n = ba2int(bits[pos:pos+N_BITS])
    pos += N_BITS
    m = ba2int(bits[pos:pos+M_BITS])
    pos += M_BITS
    d = ba2int(bits[pos:pos+D_BITS])
    pos += D_BITS
    w = ba2int(bits[pos:pos+W_BITS])
    pos += W_BITS
    adaptive = bool(bits[pos])
    pos += 1
    q = 0
    if not adaptive:
        q = ba2int(bits[pos:pos+4])
        pos += 4
    else:
        digit = ba2int(bits[pos:pos+4])
        pos += 4
        exp = ba2int(bits[pos:pos+4])
        pos += 4
        q_nmse_limit = Decimal((0, (digit,), -exp))
    digit = ba2int(bits[pos:pos+4])
    pos += 4
    exp = ba2int(bits[pos:pos+4])
    pos += 4
    c_nmse_limit = Decimal((0, (digit,), -exp))
    return EncoderParams(key=key, n=n, m=m, d=d, w=w, adaptive=adaptive,
        q=q, q_nmse_limit=q_nmse_limit, c_nmse_limit=c_nmse_limit), pos

def next_byte_pos(pos):
    return (pos + 7) & (-8)


class DecodedData(NamedTuple):
    x: np.ndarray
    r_times: np.ndarray
    r_iters: np.ndarray

    @property
    def total_time(self):
        return np.sum(self.r_times)

def decode(bits: bitarray, block_size=32):
    # read the parameters
    params, pos = deserialize_encoder_params(bits)
    print(params)
    # extend the pos to next multiple of 8
    pos = next_byte_pos(pos)
    print(pos)
    yhat = read_measurements(params, bits, pos)

    # Arrange measurements into column vectors
    Yhat = crn.vec_to_windows(jnp.asarray(yhat, dtype=float), params.m)
    n_windows = Yhat.shape[1]
    options = bsbl.bsbl_bo_options(max_iters=20)
    X_hat = np.zeros((params.n, n_windows))
    r_times = np.zeros(n_windows)
    r_iters = np.zeros(n_windows, dtype=int)

    # sensing matrix
    Phi = build_sensor(params)
    DPhi = Phi.todense()
    # Start decoding
    for i in range(n_windows):
        y = Yhat[:, i]
        start = timeit.default_timer()
        sol = bsbl.bsbl_bo_np_jit(DPhi, y, block_size, options=options)
        stop = timeit.default_timer()
        rtime = stop - start
        x_hat = sol.x
        X_hat[:, i] = x_hat
        r_times[i] = rtime
        r_iters[i] = sol.iterations
        print(f'[{i}/{n_windows}], time: {rtime:.2f} sec')
    x = X_hat.flatten(order='F')
    return DecodedData(x=x, r_times=r_times, r_iters=r_iters)

def read_measurements(params, bits, pos):
    # total bits
    n_bits = len(bits)
    yhats = []
    while pos < n_bits:
        # decode a section
        # read section header
        mean_val = ba2int(bits[pos:pos+SEC_MEAN_BITS], signed=True)
        pos += SEC_MEAN_BITS
        std_val = ba2int(bits[pos:pos+SEC_STD_BITS])
        pos += SEC_STD_BITS
        q = ba2int(bits[pos:pos+SEC_Q_BITS])
        pos += SEC_Q_BITS
        rng_mult = ba2int(bits[pos:pos+SEC_RNG_BITS])
        pos += SEC_RNG_BITS
        n_windows = ba2int(bits[pos:pos+W_BITS])
        pos += W_BITS
        n_words = ba2int(bits[pos:pos+SEC_WORD_BITS])
        pos += SEC_WORD_BITS
        pos = next_byte_pos(pos)
        # print(mean_val, std_val, q, rng_mult, n_words)
        compressed = []
        for i in range(n_words):
            word = ba2int(bits[pos:pos+32])
            pos += 32
            compressed.append(word)
        a_min = int(mean_val - rng_mult * std_val)
        a_max = int(mean_val + rng_mult * std_val)
        compressed = np.array(compressed, dtype=np.uint32)
        model = constriction.stream.model.QuantizedGaussian(a_min, a_max,
            mean=mean_val, std=std_val)
        # Decode the message:
        ans_decoder = constriction.stream.stack.AnsCoder(compressed)
        # number of measurements in the section
        sl = params.m * n_windows
        yc = ans_decoder.decode(model, sl)
        yhat = yc << q
        yhats.append(yhat)
    return np.concatenate(yhats)


######################################################################
#                       COMPARISON
######################################################################

class CompressionStats(NamedTuple):
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
    info = coded_ecg.info
    n_samples = info.n_samples
    n_windows = info.n_windows
    n_measurements = info.n_measurements
    y = coded_ecg.y
    uncompressed_bits = n_samples * 11
    compressed_bits = len(coded_ecg.bits)
    ratio = crn.compression_ratio(uncompressed_bits, compressed_bits)
    pss = crn.percent_space_saving(uncompressed_bits, compressed_bits)
    bpm = compressed_bits  / n_measurements
    bps = compressed_bits/ n_samples
    y_max = np.max(np.abs(y))
    print(f'Uncompressed bits: {uncompressed_bits} Compressed bits: {compressed_bits}, ratio: {ratio:.2f}x')
    print(f'bits per measurement in compressed data: {bpm:.2f}')
    print(f'bits per measurement in cs measurements: {np.round(np.log2(2* y_max + 1))}')
    print(f'Compressed bits per sample: {bps:.2f}')
    rtime = decoded_ecg.total_time
    x = ecg[:n_samples]
    x_hat = decoded_ecg.x
    snr = crn.signal_noise_ratio(x, x_hat)
    prd = crn.percent_rms_diff(x, x_hat)
    nmse = crn.normalized_mse(x, x_hat)
    print(f'SNR: {snr:.2f} dB, PRD: {prd:.1f}%, NMSE: {nmse:.5f}, Time: {rtime:.2f} sec')
    return CompressionStats(
            u_bits=uncompressed_bits,
            c_bits=compressed_bits,
            cr=ratio, pss=pss,
            bpm=bpm, bps=bps,
            snr=float(snr), prd=float(prd), nmse=float(nmse), rtime=rtime
        )
