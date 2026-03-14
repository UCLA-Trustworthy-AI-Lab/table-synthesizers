import torch
import os

try:
    from icecream import install
except ImportError:
    install = None

# Allow configurable threading via environment variable
# Default to 1 for TabSyn compatibility, but can be overridden
num_threads = int(os.environ.get('TABSYN_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1')))
torch.set_num_threads(num_threads)
if install is not None:
    install()

from . import env  # noqa
from .data import *  # noqa
from .deep import *  # noqa
from .env import *  # noqa
from .metrics import *  # noqa
from .util import *  # noqa
