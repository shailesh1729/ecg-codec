"""
This module implements the codec
proposed in :cite:`mamaghanian2011compressed`
without inter-packet redundancy removal and
huffman coding.
"""

######################################################################
#                       ENCODER
######################################################################
# std imports
import math
from typing import NamedTuple, List

# NumPy
import jax
import numpy as np
import jax.numpy as jnp

# CR-Suite libraries
import cr.nimble as crn
import cr.nimble.dsp as crdsp
import cr.sparse.dict as crdict
import cr.sparse.lop as crlop
import cr.sparse.cvx.spgl1 as crspgl1

class EncoderParams(NamedTuple):
    """The set of parameters to configure the ECG encoder
    """
    key: jax.numpy.ndarray
    "PRNG Key for generating sensing matrix"
    n: int
    "number of samples per window"
    m : int
    "number of measurements per window"
    d: int


def build_sensor(params: EncoderParams):
    """ Constructs a sparse binary sensing matrix based on the
    encoding parameters.
    """
    Phi = crdict.sparse_binary_mtx(params.key, 
        params.m, params.n, d=params.d, normalize_atoms=False)
    return Phi

def sense(params, Phi, ecg):
    """Performs windowing, compressing sensing and flattening of ECG signal
    """
    X = crn.vec_to_windows(ecg, params.n)
    # Measurements
    Y = Phi @ X
    # Convert to numpy
    Y_np = np.array(Y).astype(int)
    y = Y_np.flatten(order='F')
    return y


class EncodedData(NamedTuple):
    """Encoded bitstream and encoding summary"""
    y: np.ndarray
    "Measurement values array (across all frames)"


def encode(params: EncoderParams, ecg: np.ndarray):
    """Encodes ECG data into measurements
    """
    # sensing matrix
    Phi = build_sensor(params)
    # measurements
    y = sense(params, Phi, ecg)
    return EncodedData(y=y)

class DecodedData(NamedTuple):
    """Decoded ECG signal and decoding summary
    """
    x: np.ndarray
    "Decoded ECG signal"
    y_hat: np.ndarray
    "Decoded measurements (after entropy decoding and inverse quantization)"
    r_times: np.ndarray
    "List of reconstruction times for each frame"
    r_iters: np.ndarray
    "List of number of iterations for reconstruction of each frame"

    @property
    def total_time(self):
        """Total reconstruction time"""
        return np.sum(self.r_times)


def decode(params: EncoderParams, y: np.ndarray):
    d_root = math.sqrt(params.d)
    d_root_inv = 1/d_root

    # sensing matrix
    PhiMat = build_sensor(params)
    PhiOp = crlop.sparse_real_matrix(PhiMat)
    scalar = crlop.scalar_mult(d_root_inv, params.m)
    Phi = crlop.compose(scalar, PhiOp)

    # scale down y
    y_hat = d_root_inv * y 
    # Arrange measurements into column vectors
    Yhat = crn.vec_to_windows(jnp.asarray(y_hat, dtype=float), params.m)
    n_windows = Yhat.shape[1]

    # reconstruction
    X_hat = np.zeros((params.n, n_windows))
    r_times = np.zeros(n_windows)
    r_iters = np.zeros(n_windows, dtype=int)

    options = crspgl1.SPGL1Options()
    # Start decoding
    for i in range(n_windows):
        y = Yhat[:, i]
        start = timeit.default_timer()
        sol = crspgl1.solve_bp_jit(Phi, y, options=options)
        stop = timeit.default_timer()
        rtime = stop - start
        x_hat = sol.x
        X_hat[:, i] = x_hat
        r_times[i] = rtime
        r_iters[i] = sol.iterations
        print(f'[{i}/{n_windows}], time: {rtime:.2f} sec')
    x = X_hat.flatten(order='F')

    return DecodedData(x=x, y_hat=y_hat, r_times=r_times, r_iters=r_iters)
