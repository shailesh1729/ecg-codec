(sec:datasets)=
# Datasets

We use the MIT-BIH Arrhythmia Database {cite}`moody2001impact`
from PhysioNet {cite}`goldberger2000physiobank`.
The database contains 48 half-hour excerpts of two-channel
ambulatory ECG recordings from 47 subjects.
The recordings were digitized at 360 samples per second
for both channels with 11-bit resolution over a 10mV range.
The samples can be read in both digital (integer) form or
as physical values (floating point) via the software provided
by PhysioNet.
We use the integer values in our experiments since our
encoder is designed for integer arithmetic.
We use the MLII signal (first channel)
from each recording in our experiments.
