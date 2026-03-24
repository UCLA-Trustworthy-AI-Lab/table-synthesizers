import pytest
import os
import pandas as pd
import numpy as np
import torch
from stg.TabSyn.tabsyn_synthesizer import TabSynSynthesizer, TABSYN_AVAILABLE

pytestmark = pytest.mark.gpu

@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_tabsyn_initialization():
    model = TabSynSynthesizer(epochs=1)
    assert model.epochs == 1

@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_tabsyn_fit_and_sample(sample_data):
    # Use epochs=1 to trigger fast path (mock training)
    model = TabSynSynthesizer(epochs=1)

    model.fit(sample_data)

    # Should use stored data bootstrap
    samples = model.sample(10, return_dataframe=True)

    assert len(samples) == 10
    assert samples.shape[1] == sample_data.shape[1]


# ------------------------------------------------------------------
# Cleanup lifecycle tests
# ------------------------------------------------------------------

@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_cleanup_callable_on_untrained_model():
    """cleanup() should be safe to call on a freshly-created (untrained) model."""
    model = TabSynSynthesizer(epochs=1)
    model.cleanup()  # should not raise
    assert model._cleaned_up is True
    assert model.trained is False


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_cleanup_is_idempotent():
    """Calling cleanup() multiple times should not raise."""
    model = TabSynSynthesizer(epochs=1)
    model.cleanup()
    model.cleanup()  # second call should be a no-op
    assert model._cleaned_up is True


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_context_manager_protocol(sample_data):
    """Using TabSynSynthesizer as a context manager should call cleanup on exit."""
    with TabSynSynthesizer(epochs=1) as synth:
        synth.fit(sample_data)
        assert synth.trained is True
        samples = synth.sample(5, return_dataframe=True)
        assert len(samples) == 5
    # After exiting the context manager, cleanup should have run
    assert synth._cleaned_up is True
    assert synth.trained is False


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_sample_raises_after_cleanup(sample_data):
    """After cleanup(), sample() should raise RuntimeError."""
    model = TabSynSynthesizer(epochs=1)
    model.fit(sample_data)
    assert model.trained is True

    model.cleanup()
    assert model.trained is False

    with pytest.raises(RuntimeError, match="trained"):
        model.sample(10)


@pytest.mark.skipif(not TABSYN_AVAILABLE, reason="TabSyn dependencies not installed")
def test_cleanup_removes_tracked_dirs(tmp_path):
    """cleanup() should remove directories and files it tracks."""
    model = TabSynSynthesizer(epochs=1)

    # Simulate tracked artifacts by creating real temp dirs/files
    fake_dir = tmp_path / "fake_ckpt"
    fake_dir.mkdir()
    (fake_dir / "model.pt").write_text("fake")

    fake_file = tmp_path / "info.json"
    fake_file.write_text("{}")

    model._cleanup_dirs = [str(fake_dir)]
    model._cleanup_files = [str(fake_file)]
    model._cleaned_up = False

    model.cleanup()

    assert not fake_dir.exists()
    assert not fake_file.exists()
    assert model._cleaned_up is True
