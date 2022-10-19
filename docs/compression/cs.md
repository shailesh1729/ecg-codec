(sec:cs)=
# Compressive Sensing

Under the digital CS paradigm, we assume that the
signals have passed through the analog-to-digital
converter (ADC) and are represented using signed integers
at a specific resolution.

CS is an emerging signal compression paradigm that relies
on the sparsity of signals (using an appropriate basis)
to compress using incoherent measurements.
The basic model can be expressed as:

$$
\by = \Phi \bx + \be
$$
where $\bx \in \RR^n$ is a signal to compress
(of length $n$). $\Phi \in \RR^{m \times n}$
is a sensing matrix which compresses $\bx$
via linear measurements. Each row of $\Phi$
represents one linear measurement as a linear
functional on $\bx$. $\by$ is a measurement
vector consisting of $m$ distinct measurements
done on $\bx$. By design, $\Phi$ is a full-rank matrix.
Hence every set of $m$ columns of $\Phi$
is linearly independent. $\be \in \RR^m$ is the
error/noise introduced during the measurement process.
In our digital CS paradigm, the noise is introduced
by the quantization step in our encoder.
In our case, $\bx$ is a window of a raw ECG record
from one channel/lead. 
Often $\bx$ is not sparse by itself, but is sparse
in some orthonormal basis $\Psi$ expressed as
$\bx = \Phi \alpha$ and the representation $\alpha$
is sparse. ECG signals exhibit sparsity in wavelet
bases.

Most natural signals have richer structures beyond
sparsity. A common structure is natural signals
is a block/group structure {cite}`eldar2010block`. 
We introduce the block/group structure on $\bx$ as

$$
\bx = \begin{pmatrix}
\bx_1 & \bx_2 & \dots & \bx_g
\end{pmatrix}
$$
where each $\bx_i$ is a block of $b$ values.
The signal $\bx$ consists of $g$ such blocks/groups.
Under the block sparsity model, only a few $k \ll g$
blocks are nonzero (active) in the signal $\bx$
however, the locations of these blocks are unknown.
We can rewrite the sensing equation as:

$$
\by = \sum_{i=1}^g \Phi_i \bx_i + \be
$$
by splitting the sensing matrix into blocks of columns appropriately.

## Block Sparse Bayesian Learning

Under the sparse Bayesian framework, each block
is assumed to satisfy a parametrized multivariate
Gaussian distribution:

$$
\PP(\bx_i ; \gamma_i, \bB_i) = \NNN(\bzero, \gamma_i \bB_i), \Forall i=1,\dots,g.
$$
The covariance matrix $\bB_i$ captures the intra-block correlations.
We further assume that the blocks are mutually uncorrelated.
The prior of $\bx$ can then be written as

$$
\PP(\bx; \{ \gamma_i, \bB_i\}_i ) = \NNN(\bzero, \Sigma_0)
$$
where

$$
\Sigma_0 = \text{diag}\{\gamma_1 \bB_1, \dots, \gamma_g \bB_g \}.
$$
We also model the correlation among the values
within each active block as an AR-1 process.
Under this assumption the matrices
$\bB_i$ take the form of a Toeplitz matrix

$$
\bB = \begin{bmatrix}
1 & r & \dots & r^{b-1}\\
r & 1 & \dots & r^{b-2}\\
\vdots &  & \ddots & \vdots\\
r^{b-1} & r^{b-2} & \dots & 1
\end{bmatrix}
$$
where $r$ is the AR-1 model coefficient.
This constraint significantly reduces the model parameters to be learned.

Measurement error is modeled as independent zero mean Gaussian
noise $\PP(\be; \lambda) \sim \NNN(\bzero, \lambda \bI)$.
BSBL doesn't require us to provide the value of noise variance
as input.
It can estimate $\lambda$ within the algorithm.

The estimate of $\bx$ under the Bayesian learning framework
is given by the posterior mean of $\bx$ given the measurements $\by$.
