import os
import sys
import click
from dotenv import load_dotenv

import numpy as np
import scipy as sp

import wfdb
import wfdb.processing

# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

import jax
from jax import random
import jax.numpy as jnp
# Some keys for generating random numbers
key = random.PRNGKey(0)
keys = random.split(key, 4)

import cr.nimble as crn
import cr.wavelets as wt
import cr.sparse as crs
import cr.sparse.lop

load_dotenv()

def get_db_dir():
    db_dir = os.getenv('MIT_BIH_DIR')
    if not db_dir:
        click.echo('ERROR: Please configure the environment variable MIT_BIH_DIR.')
        sys.exit(1)
    return db_dir

