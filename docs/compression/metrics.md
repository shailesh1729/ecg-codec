(sec:codec:metrics)=
# Performance Metrics

In our encoder, ECG signal is split into windows
of $n$ samples each which can be multiplied
with a sensing matrix. Each window of $n$
samples generates $m$ measurements by the
sensing equation $\by = \Phi \bx$.
Assume that we are encoding $s$ ECG samples where
$s = n w$ and $w$ is the number of signal windows
being encoded.
Let the ECG signal be sampled by the ADC device
at a resolution of $r$ bits per sample.
For MIT-BIH Arrhythmia database, $r=11$.
Then the number of uncompressed bits is given by
$\bits_u = r s$.

## Compression Ratio

Let the total number of compressed
bits corresponding to the $s$ ECG samples be
$\bits_c$.
This includes the overhead bits required
for the stream header and frame headers to be explained later.
Then the **compression ratio** ($\compr$) is defined as

$$
\compr \triangleq \frac{\bits_u}{\bits_c}.
$$
**Percentage space saving** ($\pss$) is defined as

$$
\pss \triangleq \frac{\bits_u - \bits_c}{\bits_u} \times 100.
$$
Note that often in literature, $\pss$ is defined as compression ratio
(e.g., {cite}`mamaghanian2011compressed`).
Several papers ignore the bitstream formation aspect
and report $\frac{m}{n} \times 100$
(e.g., {cite}`zhang2016comparison`)
or $\frac{n - m}{n} \times 100$ (e.g., {cite}`zhang2021csnet`)
as the compression ratio
which measures the reduction in number of measurements
compared to the number of samples in each window.
We shall call this metric as \emph{percentage measurement
saving} ($\pms$):

$$
\pms \triangleq \frac{n - m}{n} \times 100.
$$
The ratio $m/n$ will be called as the **measurement ratio**:

$$
\mathrm{MR}= \frac{m}{n}.
$$

The measurement ratio $\frac{m}{n}$ is not a
good indicator of compression ratio.
If the sensing matrix $\Phi$ is Gaussian,
then the measurement values are real valued.
In literature using Gaussian sensing matrices
(e.g., {cite}`zhang2016comparison`),
it is unclear how many bits are
being used to represent each floating point measurement value
for transmission.
Under standard 32-bit IEEE floating point format,
each value would require 32-bits.
Then for MIT-BIH data the compression ratio in bits
would be $\frac{11 \times n}{32 \times m}$.
The only way the ratio $\frac{m}{n}$ would make sense
if the measurements are also quantized at 11 bits
resolution. However the impact of such quantization
is not considered in the simulations.

Now consider the case of a sparse binary sensing
matrix. Since it consists of only zeros and ones,
hence for integer inputs, it generates integer
outputs. Thus, we can say that output of a sparse
binary sensor are quantized by design.
However, the range of values changes.
Assume that the sensing matrix has $d$ ones per column.
Then it has a total of $n d$ ones. Thus, each row
will have on average $\frac{n d}{m}$ ones.
Since the ones are randomly placed, hence
we won't have same number of ones in each row.
If we assume the input data to be in the range
of $[-1024, 1023]$ (under 11-bit), then in the
worst case, the range of output values may go
up to$[-\frac{n d}{m} \times 1024, \frac{n d}{m} \times 1023]$.
For a simple case where $n = 2m$ and $d=4$, we will require
14 bits to represent each measurement value.
To achieve $\frac{m}{n}$ as the compression ratio, we will
have to quantize the measurements in 11 bits. If we do so,
we shall need to provide some way to communicate the quantization
parameters to the decoder as well as study the impact of
quantization noise.
This issue seems to be ignored in {cite}`zhang2012compressed`.

Another way of looking at the compressibility is how
many bits per sample ($\bps$) are needed on average in the compressed
bitstream. We define $\bps$ as:

$$
\bps \triangleq \frac{\bits_c}{s}.
$$
Since the entropy coder is coding the measurements rather than
the samples directly, hence it is also useful to see how
many bits are needed to code each measurement. We
denote this as bits per measurement ($\bpm$):

$$
\bpm \triangleq \frac{\bits_c}{m w}.
$$

## Reconstruction Quality

The **normalized root mean square error** is defined as

$$
\nrmse (\bx, \tilde{\bx}) \triangleq \frac{\| \bx - \tilde{\bx}\|_2}{\| \bx \|_2}
$$
where $\bx$ is the original ECG signal and $\tilde{\bx}$
is the reconstructed signal.
A popular metric to measure the quality of reconstruction
of ECG signals is
**percentage root mean square difference** ($\prd$):

$$
\prd(\bx, \tilde{\bx}) \triangleq \nrmse(\bx, \tilde{\bx}) \times 100
$$
The **signal to noise ratio** ($\snr$) is related to $\prd$ as

$$
\snr \triangleq -20 \log_{10}(0.01 \prd).
$$
As one desires higher compression ratios and lower
$\prd$, one can define a combined **quality score** (QS) as

$$
\text{QS} = \frac{\compr}{\prd} \times 100.
$$


Zigel et al. {cite}`zigel2000weighted` established 
a link between the diagnostic distortion and
the easy to measure $\prd$ metric.
\Cref{tbl:quality:prd:snr} shows the classified quality
and corresponding SNR (signal to noise ratio) and PRD ranges.

```{list-table} Quality of Reconstruction
:header-rows: 1
:name: tbl:quality:prd:snr

* - Quality 
  - PRD 
  - SNR 
* - Very good 
  - $<$ 2\% 
  - $>$ 33 dB 
* - Good 
  - 2-9\% 
  - 20-33 dB 
* - Undetermined 
  - $\geq$ 9\% 
  - $\leq$ 20 dB
```
