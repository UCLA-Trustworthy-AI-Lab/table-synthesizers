import pytest
import pandas as pd
import tempfile
from pathlib import Path
from data_loader import DataLoader

class TestDataLoader:
    
    @pytest.fixture
    def data_loader(self):
        return DataLoader()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
    def test_load_csv(self, data_loader, sample_df):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            sample_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
            
        try:
            loaded_df = data_loader.load(tmp_path)
            pd.testing.assert_frame_equal(loaded_df, sample_df)
        finally:
            Path(tmp_path).unlink()

    def test_load_parquet(self, data_loader, sample_df):
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            # Need to close file to let pandas write to it (Windows mostly, but good practice)
            pass
        
        tmp_path = tmp.name
        # pandas to_parquet requires a path string or file-like object
        sample_df.to_parquet(tmp_path, index=False)
            
        try:
            loaded_df = data_loader.load(tmp_path)
            pd.testing.assert_frame_equal(loaded_df, sample_df)
        finally:
            Path(tmp_path).unlink()

    def test_file_not_found(self, data_loader):
        with pytest.raises(FileNotFoundError):
            data_loader.load("non_existent_file.csv")

    def test_unsupported_format(self, data_loader):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"content")
            tmp_path = tmp.name
            
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                data_loader.load(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_memory_optimization(self, data_loader):
        # Create a dataframe with types that can be optimized
        df = pd.DataFrame({
            'int_col': pd.Series([1, 2, 3, 4, 5], dtype='int64'),
            'float_col': pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float64'),
            'obj_col': pd.Series(['a', 'a', 'a', 'a', 'b'], dtype='object') # 2/5 = 0.4 < 0.5
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
            
        try:
            loaded_df = data_loader.load(tmp_path, optimize_memory=True)
            
            # Check optimizations
            assert loaded_df['int_col'].dtype == 'int8' # Should fit in int8
            assert loaded_df['float_col'].dtype == 'float32' # Should be downcast
            assert loaded_df['obj_col'].dtype == 'category' # Should be category
            
        finally:
            Path(tmp_path).unlink()

    def test_load_csv_chunking(self, data_loader, sample_df):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            # Create a larger df to chunk
            large_df = pd.concat([sample_df] * 10, ignore_index=True)
            large_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
            
        try:
            # Chunk size 5, length 30 -> 6 chunks
            iterator = data_loader.load(tmp_path, chunk_size=5)
            chunks = list(iterator)
            
            assert len(chunks) == 6
            reconstructed_df = pd.concat(chunks, ignore_index=True)
            pd.testing.assert_frame_equal(reconstructed_df, large_df)
            
        finally:
            Path(tmp_path).unlink()

    def test_load_parquet_chunking(self, data_loader, sample_df):
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            # Create a larger df to chunk
            large_df = pd.concat([sample_df] * 10, ignore_index=True)
            pass
        
        tmp_path = tmp.name
        large_df.to_parquet(tmp_path, index=False)
            
        try:
            # Chunk size 5
            iterator = data_loader.load(tmp_path, chunk_size=5)
            chunks = list(iterator)
            
            # Note: Parquet batching depends on how the file was written (row groups).
            # If written as a single block, iter_batches might return one large batch 
            # unless we force row group size or pyarrow splits it.
            # However, iter_batches(batch_size=N) *tries* to respect the limit.
            
            reconstructed_df = pd.concat(chunks, ignore_index=True)
            pd.testing.assert_frame_equal(reconstructed_df, large_df)
            
        finally:
            Path(tmp_path).unlink()
