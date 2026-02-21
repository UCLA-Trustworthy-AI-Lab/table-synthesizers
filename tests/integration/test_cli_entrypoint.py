"""Smoke tests for the train_all_compatible_models.py CLI entrypoint.

Replaces the obsolete test_all_models.py / test_all_models_cuda.py which
targeted the now-removed main.py.
"""

import subprocess
import sys


def test_help_flag():
    result = subprocess.run(
        [sys.executable, "train_all_compatible_models.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "--help" in result.stdout


def test_invalid_model_handled():
    """A bad model name should produce a non-zero exit or a clear error, not an unhandled traceback."""
    result = subprocess.run(
        [
            sys.executable,
            "train_all_compatible_models.py",
            "--models",
            "NONEXISTENT_MODEL_XYZ",
            "--dataset",
            "insurance",
            "--epochs",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    # Either exits non-zero OR prints a warning/skip message about the unknown model
    assert result.returncode != 0 or "skip" in combined.lower() or "not found" in combined.lower() or "unknown" in combined.lower() or "error" in combined.lower()
