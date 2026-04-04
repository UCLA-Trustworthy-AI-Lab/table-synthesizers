import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

try:
    from stg.CLLM.cllm_synthesizer import CLLMSynthesizer, OPENAI_AVAILABLE
except ImportError:
    OPENAI_AVAILABLE = False


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def cllm_model():
    """Pre-configured CLLMSynthesizer with test API key."""
    if not OPENAI_AVAILABLE:
        pytest.skip("openai not installed")
    return CLLMSynthesizer(api_key="test-key-12345", model="gpt-5-nano-2025-08-07")


@pytest.fixture
def trained_cllm(cllm_model, sample_data):
    """CLLMSynthesizer that has been fit on sample_data."""
    cllm_model.fit(sample_data)
    return cllm_model


def _make_csv_response(sample_data, n_rows=5):
    """Build a plausible CSV response mimicking LLM output."""
    rng = np.random.RandomState(42)
    rows = []
    for _ in range(n_rows):
        f1 = rng.randn()
        f2 = rng.randint(0, 10)
        t = rng.choice(["A", "B"])
        rows.append(f"{f1:.4f},{f2},{t}")
    return "\n".join(rows)


@pytest.fixture
def mock_openai_client():
    """Patch openai.OpenAI to return a mock client that produces CSV rows."""
    csv_text = _make_csv_response(None, n_rows=10)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = csv_text

    with patch("stg.CLLM.cllm_synthesizer.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client
        yield mock_cls, mock_client


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------
def test_cllm_availability():
    assert isinstance(OPENAI_AVAILABLE, bool)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_initialization():
    model = CLLMSynthesizer(api_key="test-key")
    assert model.api_key == "test-key"
    assert model.model_name == "gpt-5-nano-2025-08-07"
    assert model.temperature == 0.7
    assert model.max_tokens == 4096
    assert model.n_examples == 5
    assert model.batch_size_per_call == 10
    assert model.max_retries == 3
    assert model.stored_data is None


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_custom_params():
    model = CLLMSynthesizer(
        api_key="k",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2048,
        n_examples=3,
        batch_size_per_call=5,
        max_retries=1,
    )
    assert model.model_name == "gpt-4o"
    assert model.temperature == 0.3
    assert model.max_tokens == 2048
    assert model.n_examples == 3
    assert model.batch_size_per_call == 5
    assert model.max_retries == 1


# ------------------------------------------------------------------
# Training / Fit
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_fit_stores_data(trained_cllm, sample_data):
    assert trained_cllm.stored_data is not None
    assert list(trained_cllm._column_names) == list(sample_data.columns)
    assert len(trained_cllm._dtypes) == len(sample_data.columns)
    assert len(trained_cllm._column_stats) == len(sample_data.columns)
    assert trained_cllm._example_rows is not None
    assert len(trained_cllm._example_rows) <= trained_cllm.n_examples


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_rejects_dataloader():
    model = CLLMSynthesizer(api_key="test")
    tensor_data = torch.randn(100, 3)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    with pytest.raises(ValueError, match="DataFrame"):
        model.fit(loader)


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_build_system_prompt(trained_cllm):
    prompt = trained_cllm._build_system_prompt()
    assert "synthetic" in prompt.lower()
    assert "CSV" in prompt


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_build_user_prompt(trained_cllm, sample_data):
    prompt = trained_cllm._build_user_prompt(5)
    # Should contain column names
    for col in sample_data.columns:
        assert col in prompt
    # Should contain example rows
    assert "example" in prompt.lower()
    # Should request 5 rows
    assert "5" in prompt


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_parse_response_valid_csv(trained_cllm):
    csv_text = "0.5,3,A\n-1.2,7,B\n0.8,1,A"
    df = trained_cllm._parse_response(csv_text, 3)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == trained_cllm._column_names


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_parse_response_with_code_fences(trained_cllm):
    csv_text = "```csv\n0.5,3,A\n-1.2,7,B\n```"
    df = trained_cllm._parse_response(csv_text, 2)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_parse_response_malformed(trained_cllm):
    garbage = "Sorry, I cannot generate data for you."
    df = trained_cllm._parse_response(garbage, 5)
    assert isinstance(df, pd.DataFrame)
    # Should return empty or partial DataFrame gracefully
    assert len(df) <= 5


# ------------------------------------------------------------------
# Generation (mocked API)
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_generate_mocked(trained_cllm, mock_openai_client):
    samples = trained_cllm._generate(5)
    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 5
    assert list(samples.columns) == trained_cllm._column_names


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_generate_batched(trained_cllm, mock_openai_client):
    # batch_size_per_call is 10, requesting 25 should make 3 calls
    trained_cllm.batch_size_per_call = 10
    samples = trained_cllm._generate(25)
    _, mock_client = mock_openai_client
    assert mock_client.chat.completions.create.call_count >= 2
    assert isinstance(samples, pd.DataFrame)


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_sample_dataframe(trained_cllm, mock_openai_client):
    samples = trained_cllm.sample(5, return_dataframe=True)
    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 5


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_sample_tensor(trained_cllm, mock_openai_client):
    tensor_out = trained_cllm.sample(5, return_dataframe=False)
    assert isinstance(tensor_out, torch.Tensor)
    assert tensor_out.shape[0] == 5


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_get_state_excludes_api_key(trained_cllm):
    state = trained_cllm.get_state()
    assert "api_key" not in state
    # Verify useful data IS present
    assert "stored_data" in state
    assert "column_names" in state
    assert "dtypes" in state
    assert "column_stats" in state


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_load_state(trained_cllm):
    state = trained_cllm.get_state()

    model2 = CLLMSynthesizer(api_key="new-key")
    model2.load_state(state)
    assert model2._column_names == trained_cllm._column_names
    assert model2._dtypes == trained_cllm._dtypes
    # api_key should NOT be overwritten from checkpoint
    assert model2.api_key == "new-key"


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_no_api_key_raises(sample_data):
    model = CLLMSynthesizer(api_key=None)
    model.fit(sample_data)
    with pytest.raises(ValueError, match="API key"):
        model._call_api("system", "user")


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_generate_before_fit():
    model = CLLMSynthesizer(api_key="test")
    with pytest.raises(RuntimeError, match="trained"):
        model._generate(5)


# ------------------------------------------------------------------
# Factory registration
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_factory(sample_data, mock_openai_client):
    from stg.tableSynthesizer import TableSynthesizer, DEFAULT_MODELS

    if "CLLM" not in DEFAULT_MODELS:
        pytest.skip("CLLM not registered in factory")

    ts = TableSynthesizer("CLLM", config={"api_key": "test-key"})
    ts.fit(sample_data)
    samples = ts.sample(5, return_dataframe=True)
    assert samples.shape == (5, sample_data.shape[1])


# ------------------------------------------------------------------
# Numeric-only data
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_numeric_only_data(mock_openai_client):
    np.random.seed(42)
    df = pd.DataFrame({
        "x": np.random.randn(50),
        "y": np.random.rand(50) * 10,
    })
    model = CLLMSynthesizer(api_key="test")
    model.fit(df)

    # Override mock to return numeric CSV
    _, mock_client = mock_openai_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "0.5,3.2\n-1.2,7.1\n0.8,1.5\n0.1,4.0\n-0.3,6.6"
    mock_client.chat.completions.create.return_value = mock_response

    samples = model._generate(5)
    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 5
    assert list(samples.columns) == ["x", "y"]


# ------------------------------------------------------------------
# Categorical handling in prompts
# ------------------------------------------------------------------
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai not installed")
def test_cllm_categorical_in_prompt(trained_cllm):
    prompt = trained_cllm._build_user_prompt(3)
    # The categorical column "target" stats should mention possible values
    assert "categorical" in prompt.lower()
    # Should list A and B as possible values
    target_stats = trained_cllm._column_stats["target"]
    assert target_stats["type"] == "categorical"
    assert "A" in target_stats["unique_values"]
    assert "B" in target_stats["unique_values"]
