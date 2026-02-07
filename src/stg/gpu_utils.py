"""
GPU Utilities for Table Synthesizers

This module provides utility functions for GPU detection, device selection,
and optimization for deep learning-based table synthesizers.
"""

import torch
import logging
from typing import Optional, Dict, Tuple


def detect_best_device() -> torch.device:
    """
    Automatically detect the best available device for training.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: Best available device

    Examples:
        >>> device = detect_best_device()
        >>> print(device)
        device(type='cuda', index=0)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"Detected CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Detected Apple Silicon GPU (MPS)")
        return device
    else:
        device = torch.device("cpu")
        logging.info("No GPU detected, using CPU")
        return device


def get_device_info() -> Dict[str, any]:
    """
    Get detailed information about available compute devices.

    Returns:
        dict: Dictionary containing device information:
            - 'type': 'cuda', 'mps', or 'cpu'
            - 'name': Device name (for CUDA)
            - 'memory_gb': Total memory in GB (for CUDA)
            - 'compute_capability': (major, minor) tuple (for CUDA)
            - 'device_count': Number of GPUs (for CUDA)

    Examples:
        >>> info = get_device_info()
        >>> print(f"Device: {info['type']}, Memory: {info.get('memory_gb', 'N/A')} GB")
        Device: cuda, Memory: 119.7 GB
    """
    info = {}

    if torch.cuda.is_available():
        info['type'] = 'cuda'
        info['device_count'] = torch.cuda.device_count()
        info['name'] = torch.cuda.get_device_name(0)

        props = torch.cuda.get_device_properties(0)
        info['memory_gb'] = props.total_memory / 1024**3
        info['compute_capability'] = (props.major, props.minor)
        info['multi_processor_count'] = props.multi_processor_count

        # Check CUDA version
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['type'] = 'mps'
        info['name'] = 'Apple Silicon GPU'

    else:
        info['type'] = 'cpu'
        info['name'] = 'CPU'

    return info


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available.

    Returns:
        bool: True if GPU is available, False otherwise

    Examples:
        >>> if is_gpu_available():
        ...     print("GPU training enabled")
        GPU training enabled
    """
    return torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())


def get_optimal_batch_size(
    dataset_size: int,
    model_memory_per_sample_mb: float = 1.0,
    default_batch_size: int = 128,
    device: Optional[torch.device] = None
) -> int:
    """
    Calculate optimal batch size based on GPU memory and dataset size.

    Args:
        dataset_size (int): Number of samples in the dataset
        model_memory_per_sample_mb (float): Estimated memory per sample in MB (default: 1.0)
        default_batch_size (int): Default batch size if GPU not available (default: 128)
        device (torch.device, optional): Device to use. If None, auto-detect.

    Returns:
        int: Recommended batch size

    Examples:
        >>> batch_size = get_optimal_batch_size(10000, model_memory_per_sample_mb=2.0)
        >>> print(f"Recommended batch size: {batch_size}")
        Recommended batch size: 256

    Note:
        This is a heuristic. Actual optimal batch size depends on:
        - Model architecture complexity
        - Input dimension
        - GPU memory bandwidth
        - Dataset characteristics
    """
    if device is None:
        device = detect_best_device()

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = props.total_memory / 1024**3

        # Reserve 20% for model parameters and other overhead
        available_memory_mb = (gpu_memory_gb * 1024) * 0.8

        # Calculate max batch size based on memory
        max_batch_size = int(available_memory_mb / model_memory_per_sample_mb)

        # Heuristic adjustments based on GPU memory tier
        if gpu_memory_gb >= 80:  # High-end GPUs (A100, H100, GB10, etc.)
            recommended_batch_size = min(512, max_batch_size, dataset_size // 2)
        elif gpu_memory_gb >= 40:  # Mid-high GPUs (A6000, RTX 6000, etc.)
            recommended_batch_size = min(256, max_batch_size, dataset_size // 4)
        elif gpu_memory_gb >= 16:  # Consumer high-end (RTX 4090, RTX 3090, etc.)
            recommended_batch_size = min(128, max_batch_size, dataset_size // 8)
        elif gpu_memory_gb >= 8:   # Consumer mid-range (RTX 4070, RTX 3070, etc.)
            recommended_batch_size = min(64, max_batch_size, dataset_size // 10)
        else:                       # Entry-level GPUs
            recommended_batch_size = min(32, max_batch_size, dataset_size // 10)

        logging.info(
            f"GPU memory: {gpu_memory_gb:.1f}GB, "
            f"Available: {available_memory_mb:.0f}MB, "
            f"Recommended batch size: {recommended_batch_size}"
        )
        return recommended_batch_size

    else:
        # CPU or MPS - use conservative batch size
        return min(default_batch_size, dataset_size // 10)


def get_gpu_models_supported() -> Dict[str, Dict[str, any]]:
    """
    Get information about which table synthesizer models support GPU acceleration.

    Returns:
        dict: Dictionary mapping model names to GPU support information:
            - 'gpu_support': bool - Whether model supports GPU
            - 'speedup': tuple - (min, max) speedup range vs CPU
            - 'recommended_batch_size': tuple - (min, max) batch size range
            - 'memory_usage_gb': tuple - (min, max) typical memory usage

    Examples:
        >>> models = get_gpu_models_supported()
        >>> if models['CTGAN']['gpu_support']:
        ...     print("CTGAN supports GPU acceleration")
        CTGAN supports GPU acceleration
    """
    return {
        'CTGAN': {
            'gpu_support': True,
            'speedup': (15, 30),
            'recommended_batch_size': (100, 500),
            'memory_usage_gb': (2, 8),
            'description': 'Conditional GAN - Best balance of speed and quality'
        },
        'TVAE': {
            'gpu_support': True,
            'speedup': (10, 20),
            'recommended_batch_size': (100, 500),
            'memory_usage_gb': (2, 6),
            'description': 'VAE - Fastest training'
        },
        'TabDDPM': {
            'gpu_support': True,
            'speedup': (20, 40),
            'recommended_batch_size': (128, 512),
            'memory_usage_gb': (4, 12),
            'description': 'Diffusion model - Highest quality'
        },
        'PATE-CTGAN': {
            'gpu_support': True,
            'speedup': (10, 25),
            'recommended_batch_size': (50, 200),
            'memory_usage_gb': (4, 10),
            'description': 'Privacy-preserving GAN'
        },
        'AutoDiff': {
            'gpu_support': True,
            'speedup': (15, 30),
            'recommended_batch_size': (100, 300),
            'memory_usage_gb': (3, 8),
            'description': 'Automatic differential privacy'
        },
        'NFlow': {
            'gpu_support': True,
            'speedup': (10, 20),
            'recommended_batch_size': (100, 400),
            'memory_usage_gb': (3, 7),
            'description': 'Normalizing flows'
        },
        'GReat': {
            'gpu_support': True,
            'speedup': (20, 35),
            'recommended_batch_size': (64, 256),
            'memory_usage_gb': (6, 16),
            'description': 'Transformer-based generation'
        },
        'TabSyn': {
            'gpu_support': True,
            'speedup': (5, 15),
            'recommended_batch_size': (32, 128),
            'memory_usage_gb': (4, 10),
            'description': 'VAE + Diffusion (has compatibility issues)'
        },
        # Non-GPU models
        'Identity': {
            'gpu_support': False,
            'speedup': (1, 1),
            'recommended_batch_size': None,
            'memory_usage_gb': None,
            'description': 'Pass-through (no training)'
        },
        'CART': {
            'gpu_support': False,
            'speedup': (1, 1),
            'recommended_batch_size': None,
            'memory_usage_gb': None,
            'description': 'Decision tree (scikit-learn, CPU-only)'
        },
        'DPCART': {
            'gpu_support': False,
            'speedup': (1, 1),
            'recommended_batch_size': None,
            'memory_usage_gb': None,
            'description': 'Differential privacy tree'
        },
        'SMOTE': {
            'gpu_support': False,
            'speedup': (1, 1),
            'recommended_batch_size': None,
            'memory_usage_gb': None,
            'description': 'Oversampling (scikit-learn, CPU-only)'
        },
        'BayesianNetwork': {
            'gpu_support': False,
            'speedup': (1, 1),
            'recommended_batch_size': None,
            'memory_usage_gb': None,
            'description': 'Probabilistic model'
        },
        'AIM': {
            'gpu_support': False,
            'speedup': (1, 1),
            'recommended_batch_size': None,
            'memory_usage_gb': None,
            'description': 'Statistical model'
        },
        'ARF': {
            'gpu_support': False,
            'speedup': (1, 1),
            'recommended_batch_size': None,
            'memory_usage_gb': None,
            'description': 'Random forest'
        },
    }


def print_gpu_info():
    """
    Print detailed GPU information to console.

    This is useful for debugging and verifying GPU setup.

    Examples:
        >>> print_gpu_info()
        ============================================================
        GPU Information
        ============================================================
        Device Type: cuda
        GPU Name: NVIDIA GB10
        Total Memory: 119.7 GB
        Compute Capability: 12.1
        Multi Processors: 48
        CUDA Version: 13.0
        cuDNN Version: 91701
        ============================================================
    """
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)

    info = get_device_info()

    print(f"Device Type: {info['type']}")

    if info['type'] == 'cuda':
        print(f"GPU Name: {info['name']}")
        print(f"Total Memory: {info['memory_gb']:.1f} GB")
        print(f"Compute Capability: {info['compute_capability'][0]}.{info['compute_capability'][1]}")
        print(f"Multi Processors: {info['multi_processor_count']}")
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"Number of GPUs: {info['device_count']}")

    elif info['type'] == 'mps':
        print(f"Device: {info['name']}")

    else:
        print("No GPU available")

    print("=" * 60)


def validate_gpu_setup() -> Tuple[bool, str]:
    """
    Validate GPU setup and return status with message.

    Returns:
        tuple: (success: bool, message: str)

    Examples:
        >>> success, message = validate_gpu_setup()
        >>> print(message)
        ✅ GPU setup validated: NVIDIA GB10 (SM 12.1) with CUDA 13.0
    """
    if not is_gpu_available():
        return False, "❌ No GPU detected. Install PyTorch with CUDA support."

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = (props.major, props.minor)
        cuda_version = torch.version.cuda

        # Check CUDA version compatibility
        if compute_cap[0] == 12 and compute_cap[1] == 1:
            # Blackwell architecture (SM 12.1)
            if cuda_version < "13.0":
                return False, (
                    f"❌ GPU {gpu_name} (SM {compute_cap[0]}.{compute_cap[1]}) requires CUDA 13.0+, "
                    f"but found CUDA {cuda_version}. Please upgrade PyTorch."
                )

        # Test CUDA operations
        try:
            x = torch.rand(100, 100, device='cuda')
            y = torch.rand(100, 100, device='cuda')
            z = torch.matmul(x, y)
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            return False, f"❌ CUDA operations failed: {e}"

        return True, (
            f"✅ GPU setup validated: {gpu_name} "
            f"(SM {compute_cap[0]}.{compute_cap[1]}) with CUDA {cuda_version}"
        )

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS operations
        try:
            x = torch.rand(100, 100, device='mps')
            y = torch.rand(100, 100, device='mps')
            z = torch.matmul(x, y)
            del x, y, z
        except Exception as e:
            return False, f"❌ MPS operations failed: {e}"

        return True, "✅ Apple Silicon GPU (MPS) setup validated"

    return False, "❌ Unknown error in GPU validation"


if __name__ == "__main__":
    # Demo usage
    print_gpu_info()

    print("\nValidating GPU setup...")
    success, message = validate_gpu_setup()
    print(message)

    if success:
        print("\nTesting GPU operations...")
        device = detect_best_device()
        print(f"Best device: {device}")

        # Test batch size calculation
        batch_size = get_optimal_batch_size(10000)
        print(f"Optimal batch size for 10K samples: {batch_size}")

        # Show supported models
        print("\nGPU-Accelerated Models:")
        models = get_gpu_models_supported()
        for name, info in models.items():
            if info['gpu_support']:
                print(f"  ✅ {name}: {info['description']}")
                print(f"     Speedup: {info['speedup'][0]}-{info['speedup'][1]}x")
