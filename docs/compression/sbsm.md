(sec:sbsm)=
# Sparse Binary Sensing Matrices


A (random) sparse binary sensing matrix has a very simple design.
Assume that the signal space is $\RR^n$ and the measurement
space is $\RR^m$.
Every column of a sparse binary sensing matrix has a 1 in
exactly $d$ positions and zeros elsewhere. The indices
at which ones are present are randomly selected for each column.

Following is an example sparse binary matrix with 3 ones in
each column:

$$
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 1 & 1 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 1\\
1 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 1 & 0\\
1 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 1 & 0\\
0 & 1 & 1 & 0 & 1 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 1\\
0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix}
$$

From the perspective of algorithm design, we often require that
the sensing matrix have unit norm columns. This can be easily
attained for sparse binary matrices by scaling them with
$\frac{1}{\sqrt{d}}$. The normalization can be skipped during
the sensing process as it allows us to implement the sensing
process entirely using simple lookup and addition operations.
The normalization by $\frac{1}{\sqrt{d}}$ will necessarily be
used during sparse reconstruction if the reconstruction algorithm
demands that the columns of the sensing matrix have unit norms.


## Mutual Coherence

The mutual coherence of a sensing matrix with an orthonormal basis
is a good indicator of robust and efficient recovery of signals
with a sparse representation in the given basis.

Let $\Phi$ be a given $m \times n$ sensing matrix and
$\Psi$ be an $n \times n$ a sparsifying orthonormal basis.
Then mutual coherence is given by

$$
\max_{1 \leq i \leq m, 1 \leq j \leq n} | \langle \phi_i, \psi_j \rangle |
$$
where $\phi_i$ are the rows of $\Phi$ and $\psi_j$ are the columns of $\Psi$.
{numref}`fig-sbsm-db10-coherence` shows the variation of mutual coherence of
sparse binary sensing matrices (averaged over 10 trials each) as $d$ increases.

```{figure} images/sbsm/coherence-w_db10-m_256-n_512-t_10-dmax_75.png
---
name: fig-sbsm-db10-coherence
---
Variation of mutual coherence of sparse binary sensing matrices with
a db10 wavelet basis as number of ones per column increases.
```
