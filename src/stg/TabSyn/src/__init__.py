import torch
import os
from icecream import install

# Allow configurable threading via environment variable
# Default to 1 for TabSyn compatibility, but can be overridden
num_threads = int(os.environ.get('TABSYN_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1')))
torch.set_num_threads(num_threads)
install()

from . import env  # noqa
from .data import *  # noqa
from .deep import *  # noqa
from .env import *  # noqa
from .metrics import *  # noqa
from .util import *  # noqa