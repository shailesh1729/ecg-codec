import os
import sys
import click
from dotenv import load_dotenv

import numpy as np
import scipy as sp
import wfdb
import wfdb.processing
import constriction

import jax
from jax import random
import jax.numpy as jnp

import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.dict
import cr.sparse.data
import cr.sparse.lop
import cr.biosignals as crb

load_dotenv()

# Some keys for generating random numbers
key = random.PRNGKey(0)
keys = random.split(key, 4)

@click.group()
def main():
    pass


# Number of measurements
M = 512
# Ambient dimension
N = 1024
SCALE_FACTOR = 16
MAX_Y_VAL = 15
Y_STD = 4.


def get_db_dir():
    db_dir = os.getenv('MIT_BIH_DIR')
    if not db_dir:
        click.echo('ERROR: Please configure the environment variable MIT_BIH_DIR.')
        sys.exit(1)
    return db_dir



@click.command()
@click.argument('name', type=str)
def build():
    pass

@click.command()
@click.argument('record_num', type=int)
def encode(record_num):
    mit_bih_dir = get_db_dir()
    click.echo(f'Processing record: {record_num}:')
    record = wfdb.rdrecord(f'{mit_bih_dir}/{record_num}', channels=[0], physical=False)
    ann = wfdb.rdann(f'{mit_bih_dir}/{record_num}','atr')
    signal = np.squeeze(record.d_signal)
    mean_val = int(np.mean(signal))
    signal = signal - mean_val
    fs = record.fs
    click.echo(f'Length: {len(signal)} samples, Frequency {fs} Hz, Duration: {len(signal) / fs / 60.} minutes.')
    # prepare batches
    X = crn.vec_to_windows(signal, N)
    click.echo('Preparing sensing matrix.')
    Phi = crs.lop.gaussian_dict(keys[0], M, N, normalize_atoms=True)
    # perform compressive sensing
    click.echo('Performing compressive sensing.')
    Y = Phi.times(X)
    click.echo('Flattening measurements for entropy coding.')
    # flatten the values
    Y2 = np.array(Y.flatten(order='F')) / SCALE_FACTOR
    # clip within acceptable range
    Y3 = np.clip(Y2, -MAX_Y_VAL, MAX_Y_VAL)
    # convert to integer
    Y4 = np.around(Y3).astype(int)
    # Number of values to encode
    n = len(Y4)
    click.echo('Building entropy coding model.')
    # Data distribution model
    model = constriction.stream.model.QuantizedGaussian(-MAX_Y_VAL, MAX_Y_VAL)
    # Parameters for the data distribution model
    means = np.zeros(n)
    stds = np.full(n, Y_STD)
    click.echo('Building entropy coder.')
    encoder = constriction.stream.stack.AnsCoder()
    # start encoding
    click.echo('Running entropy coder.')
    encoder.encode_reverse(Y4, model, means, stds)
    # compressed representation
    click.echo('Extracting compressed data.')
    compressed = encoder.get_compressed()
    uncompressed_bits = len(signal) * 11
    compressed_bits = len(compressed)*32
    ratio = uncompressed_bits * 1. / compressed_bits
    click.echo(f'Uncompressed bits: {uncompressed_bits} Compressed bits: {compressed_bits}, ratio: {ratio}')
    click.echo(f'bits per sample in compressed data: {compressed_bits  / n}')
    click.echo(f'bits per sample in cs measurements: {np.round(np.log2(2* MAX_Y_VAL + 1))}')
    click.echo('Writing compressed data to file.')
    if sys.byteorder != 'little':
        # Let's use the convention that we always save data in little-endian byte order.
        compressed.byteswap(inplace=True)
    dst_file = f'{record_num}.bin'
    compressed.tofile(dst_file)
    click.echo(f'Compressed data saved to file: {dst_file}.')


@click.command()
@click.argument('record_num', type=int)
def decode(record_num):
    pass


@click.command()
@click.argument('record_num', type=int)
def assess(record_num):
    pass


main.add_command(build)
main.add_command(encode)
main.add_command(decode)
main.add_command(assess)

