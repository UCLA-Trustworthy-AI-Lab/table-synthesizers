"""
Workaround module to replace libzero==0.0.8 functionality.
This module provides minimal implementations of the zero functions used in the codebase.
"""

import random as python_random
import numpy as np
import torch


def improve_reproducibility(seed, mode = 'default'):
    """
    Improve reproducibility by setting seeds for various random number generators.
    This is a workaround for zero.improve_reproducibility()

    Args:
        seed: Random seed value
        mode: 'default' (all), 'np' (numpy only), 'torch' (pytorch only), 'cuda' (cuda only)
    """
    if mode == 'default':
        # Set all seeds for maximum reproducibility
        python_random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Set deterministic algorithms for PyTorch (optional, may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif mode == 'np':
        np.random.seed(seed)
    elif mode == 'torch':
        torch.manual_seed(seed)
    elif mode == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


class Random:
    """implementation of zero.random functionality"""

    @staticmethod
    def get_state():
        """Get the current random state"""
        return {
            'python': python_random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }

    @staticmethod
    def set_state(state):
        """Set the random state"""
        if isinstance(state, dict):
            if 'python' in state:
                python_random.setstate(state['python'])
            if 'numpy' in state:
                np.random.set_state(state['numpy'])
            if 'torch' in state:
                torch.set_rng_state(state['torch'])
            if 'torch_cuda' in state and state['torch_cuda'] is not None:
                torch.cuda.set_rng_state_all(state['torch_cuda'])


def iter_batches(data, chunk_size):
    """
    Iterate over data in chunks.
    This is a workaround for zero.iter_batches()
    """
    if hasattr(data, '__len__'):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    else:
        # For iterables without len()
        chunk = []
        for item in data:
            chunk.append(item)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:  # yield remaining items
            yield chunk


class Hardware:
    """implementation of zero.hardware functionality"""

    @staticmethod
    def get_gpus_info():
        """Get GPU information"""
        try:
            import torch
            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        'id': i,
                        'name': props.name,
                        'memory_total': props.total_memory,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                return gpus
            else:
                return []
        except Exception:
            return []


# Create module-level objects to match zero's API
random = Random()
hardware = Hardware()
