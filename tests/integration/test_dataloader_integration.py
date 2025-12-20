import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.tableSynthesizer import TableSynthesizer

class TestDataLoaderIntegration:
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC']
        })
    
    def test_train_from_csv(self, sample_df):
        """Test training from CSV file"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            sample_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
            
        try:
            # Use Identity synthesizer for quick test
            synthesizer = TableSynthesizer('Identity')
            synthesizer.train_from_csv(tmp_path)
            
            # Verify synthesizer was trained
            assert synthesizer.model.model_loaded or hasattr(synthesizer.model, 'encoders')
            
            # Generate samples
            synthetic = synthesizer.sample(n=3, return_dataframe=True)
            assert len(synthetic) == 3
            assert list(synthetic.columns) == list(sample_df.columns)
            
        finally:
            Path(tmp_path).unlink()

    def test_train_from_csv_with_optimization(self, sample_df):
        """Test training from CSV with memory optimization"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            sample_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
            
        try:
            synthesizer = TableSynthesizer('Identity')
            synthesizer.train_from_csv(tmp_path, optimize_memory=True)
            
            # Verify synthesizer was trained
            assert synthesizer.model.model_loaded or hasattr(synthesizer.model, 'encoders')
            
            # Generate samples
            synthetic = synthesizer.sample(n=3, return_dataframe=True)
            assert len(synthetic) == 3
            
        finally:
            Path(tmp_path).unlink()

    def test_train_from_parquet(self, sample_df):
        """Test training from Parquet file"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            pass
        
        tmp_path = tmp.name
        sample_df.to_parquet(tmp_path, index=False)
            
        try:
            synthesizer = TableSynthesizer('Identity')
            synthesizer.train_from_parquet(tmp_path)
            
            # Verify synthesizer was trained
            assert synthesizer.model.model_loaded or hasattr(synthesizer.model, 'encoders')
            
            # Generate samples
            synthetic = synthesizer.sample(n=3, return_dataframe=True)
            assert len(synthetic) == 3
            assert list(synthetic.columns) == list(sample_df.columns)
            
        finally:
            Path(tmp_path).unlink()

    def test_train_from_parquet_with_optimization(self, sample_df):
        """Test training from Parquet with memory optimization"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            pass
        
        tmp_path = tmp.name
        sample_df.to_parquet(tmp_path, index=False)
            
        try:
            synthesizer = TableSynthesizer('Identity')
            synthesizer.train_from_parquet(tmp_path, optimize_memory=True)
            
            # Verify synthesizer was trained
            assert synthesizer.model.model_loaded or hasattr(synthesizer.model, 'encoders')
            
            # Generate samples
            synthetic = synthesizer.sample(n=3, return_dataframe=True)
            assert len(synthetic) == 3
            
        finally:
            Path(tmp_path).unlink()

    def test_train_from_csv_file_not_found(self):
        """Test error handling for non-existent CSV file"""
        synthesizer = TableSynthesizer('Identity')
        
        with pytest.raises(FileNotFoundError):
            synthesizer.train_from_csv("non_existent_file.csv")

    def test_train_from_parquet_file_not_found(self):
        """Test error handling for non-existent Parquet file"""
        synthesizer = TableSynthesizer('Identity')
        
        with pytest.raises(FileNotFoundError):
            synthesizer.train_from_parquet("non_existent_file.parquet")
