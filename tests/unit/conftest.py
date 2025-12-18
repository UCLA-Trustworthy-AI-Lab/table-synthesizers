import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Provides a simple DataFrame for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randint(0, 10, 100),
        'target': np.random.choice(['A', 'B'], 100)
    })
    return data

@pytest.fixture
def sample_categorical_data():
    """Provides a DataFrame with only categorical/discrete columns for models like AIM."""
    np.random.seed(42)
    data = pd.DataFrame({
        'col1': np.random.randint(0, 5, 100),
        'col2': np.random.randint(0, 3, 100),
        'col3': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    return data
