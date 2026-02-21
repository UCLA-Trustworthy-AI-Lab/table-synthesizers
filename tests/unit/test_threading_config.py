"""Tests for CPU threading configuration and multi-core utilization.

Validates that PyTorch threading is configurable and that importing
stg modules doesn't silently reset thread counts to 1.
"""

import multiprocessing

import torch


def test_cpu_count_positive():
    assert multiprocessing.cpu_count() > 0


def test_torch_num_threads_settable():
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        assert torch.get_num_threads() == 2
        torch.set_num_threads(4)
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(original)


def test_imports_preserve_thread_count():
    """Importing stg.tableSynthesizer must not reset threads to 1."""
    cores = multiprocessing.cpu_count()
    torch.set_num_threads(cores)
    assert torch.get_num_threads() == cores

    from stg.tableSynthesizer import TableSynthesizer  # noqa: F401

    assert torch.get_num_threads() >= cores


def test_matrix_multiply_runs():
    """Sanity-check that CPU matmul works (uses threading internally)."""
    x = torch.randn(1000, 1000)
    y = torch.mm(x, x)
    assert y.shape == (1000, 1000)
    assert torch.isfinite(y).all()


def test_interop_threads_positive():
    assert torch.get_num_interop_threads() >= 1
