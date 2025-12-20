import pandas as pd
from pathlib import Path
from typing import Optional, Union, Any

class DataLoader:
    """
    Intelligent Data Loader for CSV and Parquet files.
    """
    
    def __init__(self):
        pass

    def load(self, file_path: Union[str, Path], optimize_memory: bool = False, chunk_size: Optional[int] = None) -> Union[pd.DataFrame, Any]:
        """
        Load a file into a pandas DataFrame with optional memory optimization and chunking.
        
        Args:
            file_path: Path to the file to load (CSV or Parquet)
            optimize_memory: If True, downcast numerical columns and convert objects to categories
            chunk_size: If specified, return an iterator of DataFrames (chunked reading)
            
        Returns:
            pd.DataFrame or Iterator[pd.DataFrame]: Loaded data
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if path.suffix.lower() == '.csv':
            if chunk_size:
                iterator = pd.read_csv(path, chunksize=chunk_size)
                return (self._optimize_dtypes(chunk) if optimize_memory else chunk for chunk in iterator)
            else:
                df = pd.read_csv(path)
                return self._optimize_dtypes(df) if optimize_memory else df
                
        elif path.suffix.lower() == '.parquet':
            # PyArrow generic optimization is handled by the backend partially, 
            # but we can apply further optimization.
            # Chunked reading for parquet is slightly different.
             if chunk_size:
                 # ParquetFile from pyarrow is needed for true chunked streaming, 
                 # but pandas doesn't expose a simple chunksize for read_parquet.
                 # We'll use pyarrow to stream batches and convert to pandas.
                 import pyarrow.parquet as pq
                 parquet_file = pq.ParquetFile(path)
                 
                 def parquet_generator():
                     for batch in parquet_file.iter_batches(batch_size=chunk_size):
                         df_chunk = batch.to_pandas()
                         yield self._optimize_dtypes(df_chunk) if optimize_memory else df_chunk
                         
                 return parquet_generator()
             else:
                df = pd.read_parquet(path)
                return self._optimize_dtypes(df) if optimize_memory else df
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats: .csv, .parquet")

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage by downcasting numeric types and converting objects to categories.
        """
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                if 'int' in str(col_type):
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if c_min > -128 and c_max < 127:
                        df[col] = df[col].astype('int8')
                    elif c_min > -32768 and c_max < 32767:
                        df[col] = df[col].astype('int16')
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df[col] = df[col].astype('int32')
                    else:
                        df[col] = df[col].astype('int64')
                elif 'float' in str(col_type):
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if c_min > -3.4028235e+38 and c_max < 3.4028235e+38:
                         # Check if precision loss is acceptable? For now just downcast to float32
                        df[col] = df[col].astype('float32')
            else:
                # Convert object to category if cardinality is low
                num_unique = len(df[col].unique())
                num_total = len(df[col])
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
                    
        return df
