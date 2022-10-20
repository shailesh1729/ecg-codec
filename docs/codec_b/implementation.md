(sec:codec:b:implementation)=
# Implementation


This section provides some notes and a walk-through of the implementation
of the codec.

* The codec is implemented in the file `src/skecg/cs/codec_b.py`.
* We use `bitarray` for working with bit arrays efficiently.
* We use `constriction` for ANS (Asymmetric Numeral Systems) encoding
  and decoding.
* We use `JAX` for most of the numerical computing. This makes the codec
  portable for GPU hardware.
* We use `cr-nimble` for some basic array processing, SNR calculation etc.
* We use `cr-sparse` for compressive sensing and sparse recovery.


