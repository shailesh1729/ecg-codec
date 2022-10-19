(sec:history)=
# History

This section provides a historical review of some
of the popular compression techniques for ECG data
compression. This review is by no means complete.
The focus is on compressive sensing-based techniques
where the encoders are simple and most of the
complexity lies in the decoding.

In wireless body area networks (WBAN)
based telemonitoring networks{cite}`cao2009enabling`,
the energy consumption on sensor nodes is
a primary design constraint {cite}`milenkovic2006wireless`.
The wearable sensor nodes are often battery-operated.
It is necessary to reduce energy consumption as
much as possible.

However, long-term ECG monitoring
can generate a large amount of uncompressed data.
For example, each half-hour 2 lead recording in the
MIT-BIH Arrhythmia database {cite}`moody2001impact`
requires 1.9MB of storage. As shown in {cite}`mamaghanian2011compressed`,
in a real-time telemonitoring sensor node, the wireless
transmission of data consumes most of the energy.
The real-time compression of ECG data by a low-complexity
encoder has received significant attention
in the past decade.

ECG signal compression has been an active area
of interest for several decades. Extensive surveys
can be found in {cite}`singh2015review,rajankar2019electrocardiogram`.
Compressive sensing (CS) based techniques for ECG
data compression have been reviewed in {cite}`craven2014compressed,kumar2022review`.

## Transform Domain Techniques

Transform domain techniques
(e.g., Discrete Cosine Transform {cite}`al1995dynamic`,
Discrete Cosine Transform {cite}`batista2001compression,bendifallah2011improved`,
Discrete Wavelet Transform {cite}`djohan1995ecg,lu2000wavelet,pooyan2004wavelet,kim2006wavelet`) are popular in ECG compression
and achieve high compression ratios (CR) at clinically
acceptable quality.
However, they require computationally intensive sparsifying
transforms on all data samples and are thus not suitable
for WBAN sensor nodes {cite}`craven2014compressed`.

## Compressive Sensing Approaches

Compressive sensing {cite}`donoho2006compressed,baraniuk2007compressive,candes2006compressive,candes2008introduction,candes2006near`
uses a sub-Nyquist sampling method by acquiring a small number
of incoherent measurements which are sufficient to reconstruct
the signal if the signal is sufficiently sparse in some
basis. For a sparse signal $\bx \in \RR^n$, one would make
$m$ linear measurements where $m \ll n$ which can be
mathematically represented by a sensing operation

$$
\by = \Phi \bx
$$
where $\Phi \in \RR^{m \times n}$ is a matrix
representation of the sensing process and $\by \in \RR^m$
the set of $m$ measurements collected for $\bx$.
A suitable reconstruction algorithm can recover $\bx$
from $\by$.

Ideally, the sensing process should be implemented at the
hardware level in the analog-to-digital conversion (ADC) process.
However, much of the use of CS in ECG follows
a digital CS paradigm {cite}`mamaghanian2011compressed` where
the ECG samples are acquired first by the ADC circuit on the
device and then they are translated into incoherent
measurements via the multiplication of a digital sensing matrix.
These measurements are then transmitted
to remote telemonitoring servers. A suitable reconstruction
algorithm is used on the server to recover the original
ECG signal from the compressive measurements.
Reconstruction algorithms for ECG signals include:
greedy algorithms 
{cite}`polania2011compressed` (simultaneous orthogonal matching pursuit),
optimization-based algorithms {cite}`zhang2014energy`,
{cite}`mamaghanian2011compressed` (SPG-L1),
Bayesian learning based algorithms
{cite}`zhang2012compressed,zhang2014spatiotemporal,zhang2013extension`,
deep learning based algorithms {cite}`zhang2021csnet`.

To keep the sensing matrix multiplication
simple and efficient, sparse binary sensing matrices
are a popular choice {cite}`mamaghanian2011compressed,zhang2012compressed`.

### Entropy Coding

The literature on the use of CS for ECG compression is mostly
focused on the design of the specific *sensing matrix*,
*sparsifying dictionary*, or *reconstruction algorithm*
for the high-quality reconstruction of the ECG signal from the
compressive measurements.
To the best of our knowledge, (digital) quantization and entropy coding of
the compressive measurements of ECG data hasn't received
much attention in the past.

Mamaghanian et al.{cite}`mamaghanian2011compressed` use a Huffman codebook
which is deployed inside the sensor device. They don't
use any quantization of the measurements.
However, they don't
provide much detail on how the codebook was designed or how should
it be adapted for variations in ECG signals.
They clearly define the compression ratio in terms
of a ratio between the uncompressed bits $\bits_u$
and the compressed bits $\bits_c$. They define it
as $\frac{\bits_u - \bits_c}{\bits_u} \times 100$.
This is often known in the literature as
*percentage space savings*.

Simulation-based studies often send the real-valued compressive
measurements to the decoder modules and don't consider the issue
of the number of bits required to encode each measurement.
Zhang et al. {cite}`zhang2012compressed` and
Zhang et al. {cite}`zhang2021csnet` define
$\frac{n - m}{n} \times 100$ as compression ratio. 
Mangia et al. {cite}`mangia2020deep` define $\frac{n}{m}$
as the compression ratio.
Polania et al. {cite}`polania2018compressed` define
$\frac{m}{n}$ as compression ratio.
Picariello et al. {cite}`picariello2021novel` use
a non-random sensing matrix.
They infer a circulant binary sensing matrix
directly from the ECG signal being compressed.
The sensing matrix is adapted as and when
there is a significant change in the signal
statistics.
They represent both the ECG samples and
compressive measurements with the same number of
bits per sample/measurement. However,
in their case, there is the additional
overhead of sending the sensing matrix updates.
Their compression ratio is slightly lower than
$\frac{n}{m}$ where they call $\frac{n}{m}$ as
the *under-sampling ratio*.


Huffman codes have frequently been used in ECG data
compression for non-CS methods.
Luo et al. in {cite}`luo2014dynamic` proposed a dynamic
compression scheme for ECG signals which consisted
of a digital integrate and fire sampler followed
by an entropy coder. They used Huffman codes for
entropy coding of the timestamps associated with
the impulse train.
Chouakri et al. in {cite}`chouakri2013wavelet` compute
the DWT of ECG signal by Db6 wavelet, select coefficients
using higher order statistics thresholding, then perform
linear prediction coding and Huffman coding of the selected
coefficients.  


Asymmetric Numeral Systems (ANS) ({cite}`duda2013asymmetric`)
based entropy coding schemes have seen much success
in recent years for lossless data compression.
The *zstd* library by Facebook Inc. {cite}`zstd`
and *LZFSE* by Apple Inc. are popular lossless
data compression formats based on ANS.
To the best of our knowledge, ANS-type stream codes
have not been considered in the past for the entropy
coding of compressive measurements.
