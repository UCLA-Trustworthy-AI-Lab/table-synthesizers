#!/usr/bin/env python3
"""
Comprehensive Synthesizer Benchmark for Preprocessed Sandbox Data

This script runs all synthesizers on all 4 preprocessed datasets with realistic
hyperparameters and proper error handling for data quality issues.
"""
import sys
import os
import pandas as pd
import time
import traceback
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add the src directory to the path
sys.path.insert(0, '../src')
from stg.tableSynthesizer import TableSynthesizer

# Synthesizers to test with realistic hyperparameters for full datasets
SYNTHESIZERS_CONFIG = {
    'AutoDiff': {
        'n_epochs': 50,          # Full training epochs
        'diff_n_epochs': 20,     # Full diffusion epochs  
        'batch_size': 512        # Full batch size for efficiency
    },
    'TabSyn': {
        'epochs': 50             # Full training epochs
    },
    'GReaT': {
        'epochs': 30,            # Realistic for full dataset (was 25)
        'batch_size': 32,        # Standard batch size
        'max_length': 2048       # Full context length
    },
    'CART': {},                  # No hyperparameters needed
    'DPCART': {
        'epsilon': 1.0,
        'max_depth': 10          # Full depth for better quality
    },
    'SMOTE': {
        'k_neighbors': 5,
        'frac_samples': 1.0
    },
    'CTGAN': {
        'epochs': 100,           # Realistic epochs for full data
        'batch_size': 500,
        'generator_dim': (256, 256),
        'discriminator_dim': (256, 256)
    },
    'TVAE': {
        'epochs': 100,           # Realistic epochs for full data
        'batch_size': 500,
        'compress_dims': (128, 128),
        'decompress_dims': (128, 128)
    }
}

# Dataset files in sandbox_preprocessed
DATASETS = [
    'preprocessed_conversionsall8125.csv',         # 5,101 rows, 25 cols
    'preprocessed_amazonattributedeventsbytraffictime72925.csv',  # 4,947 rows, 166 cols  
    'preprocessed_sponsoredadstraffic72925.csv',  # 10,983 rows, 50 cols
    'preprocessed_dspimpressions72925.csv',       # 32k rows, 183 cols
]

def get_sample_size(dataset_name, total_rows, generate_full=True):
    """Determine appropriate sample size - now generates full dataset by default"""
    if generate_full:
        return total_rows  # Generate same size as training data
    else:
        # Fallback to 5% sample for very large datasets
        base_sample = max(50, min(1000, int(total_rows * 0.05)))
        
        # Adjust based on dataset characteristics  
        if 'impressions' in dataset_name:
            return min(800, base_sample)  # Large dataset, cap at 800
        elif 'amazon' in dataset_name:
            return min(400, base_sample)  # High-dimensional, cap at 400
        else:
            return base_sample

def preprocess_dataset(df, dataset_name):
    """Apply preprocessing to handle data quality issues"""
    print(f"  📊 Original shape: {df.shape}")
    print(f"  🔍 Data types: {df.dtypes.value_counts().to_dict()}")
    
    # Check for NaN values
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        print(f"  ⚠️  Columns with NaN: {len(nan_cols)} columns")
        
        # Fill NaN values appropriately
        for col in nan_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Fill numeric columns with median
                df[col] = df[col].fillna(df[col].median())
            else:
                # Fill categorical columns with mode or 'Unknown'
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_val)
    
    # Check for problematic columns with all NaN or constant values
    problematic_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            problematic_cols.append(col)
    
    if problematic_cols:
        print(f"  ⚠️  Removing {len(problematic_cols)} constant/problematic columns")
        df = df.drop(columns=problematic_cols)
    
    # Ensure all object columns are strings
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    print(f"  ✅ Preprocessed shape: {df.shape}")
    return df

def test_synthesizer(synth_name, config, df, n_samples, dataset_name, output_dir, test_idx=0):
    """Test a single synthesizer with train_test_split and CSV output"""
    print(f"\n{'='*60}")
    print(f"Testing {synth_name} on {dataset_name}")
    print(f"Configuration: {config}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Train/test split with seed=42, test_size=0.2
        print(f"🔄 Splitting dataset (80/20 train/test, seed=42)...")
        try:
            # Try stratified split for classification data
            stratify_col = None
            if df.select_dtypes(include=['object', 'category']).shape[1] > 0:
                # Use the first categorical column for stratification
                cat_col = df.select_dtypes(include=['object', 'category']).columns[0]
                if df[cat_col].value_counts().min() >= 2:  # At least 2 samples per class
                    stratify_col = df[cat_col]
                    
            train_df, test_df = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42,
                stratify=stratify_col
            )
        except Exception as e:
            print(f"  ⚠️  Stratified split failed ({e}), using random split")
            train_df, test_df = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42
            )
        
        print(f"  📊 Train: {train_df.shape[0]} samples, Test: {test_df.shape[0]} samples")
        
        synthesizer = TableSynthesizer(synth_name, config)
        
        # Fit the model on training data only
        fit_start = time.time()
        print(f"🔄 Training {synth_name} on {train_df.shape[0]} training samples...")
        synthesizer.fit(train_df)
        fit_time = time.time() - fit_start
        print(f"✅ Training completed in {fit_time:.2f}s")
        
        # Generate samples (same size as training data)
        gen_start = time.time()
        print(f"🔄 Generating {n_samples} samples...")
        synthetic_data = synthesizer.sample(n=n_samples, return_dataframe=True)
        gen_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
        print(f"✅ {synth_name} SUCCESS!")
        print(f"  Training data: {train_df.shape}")
        print(f"  Test data: {test_df.shape}")
        print(f"  Synthetic data: {synthetic_data.shape}")
        print(f"  Training time: {fit_time:.2f}s")
        print(f"  Generation time: {gen_time:.2f}s") 
        print(f"  Total time: {total_time:.2f}s")
        
        # Validate synthetic data
        if synthetic_data.shape[0] == 0:
            raise RuntimeError("Generated 0 samples")
        
        # Save synthetic data as CSV in output directory
        # Remove 'preprocessed_' prefix and '.csv' suffix from dataset_name
        real_name = dataset_name.replace('preprocessed_', '').replace('.csv', '')
        # Format: {REAL_CSV_NAME}_{SYNTHESIZER_NAME}_{TRANSFORMER_TYPE}_{TEST_IDX}.csv
        csv_filename = f"{real_name}_{synth_name}_preprocessed_{test_idx}.csv"
        csv_path = os.path.join('..', output_dir, csv_filename)  # Go back to parent dir, then to output_dir
        synthetic_data.to_csv(csv_path, index=False)
        print(f"  💾 Saved synthetic data: {csv_path}")
        
        return {
            'status': 'success',
            'fit_time': fit_time,
            'generation_time': gen_time,
            'total_time': total_time,
            'original_shape': list(df.shape),
            'train_shape': list(train_df.shape),
            'test_shape': list(test_df.shape),
            'synthetic_shape': list(synthetic_data.shape),
            'synthetic_csv_file': csv_filename,
            'config': config
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = str(e)
        print(f"❌ {synth_name} FAILED after {error_time:.2f}s")
        print(f"  Error: {error_msg}")
        print(f"  Traceback: {traceback.format_exc()}")
        
        return {
            'status': 'failed',
            'error': error_msg,
            'error_time': error_time,
            'traceback': traceback.format_exc(),
            'config': config
        }

def main(output_dir=None):
    """Run comprehensive benchmark on all datasets and synthesizers"""
    print("🚀 COMPREHENSIVE SYNTHESIZER BENCHMARK")
    print("=====================================")
    print(f"📊 Datasets: {len(DATASETS)}")
    print(f"🧬 Synthesizers: {len(SYNTHESIZERS_CONFIG)}")
    print(f"🕐 Start time: {datetime.now()}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"comprehensive_results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print("")
    
    # Change to the sandbox_preprocessed directory
    os.chdir('sandbox_preprocessed')
    
    # Results storage
    all_results = {}
    
    for dataset_idx, dataset_file in enumerate(DATASETS, 1):
        print(f"\n{'='*80}")
        print(f"[{dataset_idx}/{len(DATASETS)}] Processing: {dataset_file}")
        print(f"{'='*80}")
        
        try:
            # Load dataset
            df = pd.read_csv(dataset_file)
            print(f"📥 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Preprocess dataset
            df = preprocess_dataset(df, dataset_file)
            
            # Determine sample size (80% of original for training, generate same size)
            train_size = int(df.shape[0] * 0.8)  # 80% will be used for training
            n_samples = get_sample_size(dataset_file, train_size, generate_full=True)
            print(f"🎯 Will train on ~{train_size} samples, generate {n_samples} synthetic samples")
            
            dataset_results = {}
            
            # Test each synthesizer
            for synth_name, config in SYNTHESIZERS_CONFIG.items():
                print(f"\n🔄 Preparing to test {synth_name}...")
                
                result = test_synthesizer(synth_name, config, df, n_samples, dataset_file, output_dir, test_idx=0)
                dataset_results[synth_name] = result
                
                # Save intermediate results
                all_results[dataset_file] = dataset_results
                with open(f'../{output_dir}/comprehensive_benchmark_results.json', 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
            
        except Exception as e:
            print(f"❌ Failed to process {dataset_file}: {e}")
            all_results[dataset_file] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    # Generate final summary
    print(f"\n{'='*80}")
    print("FINAL COMPREHENSIVE SUMMARY")
    print(f"{'='*80}")
    
    total_tests = 0
    successful_tests = 0
    skipped_tests = 0
    
    for dataset, dataset_results in all_results.items():
        print(f"\n📊 {dataset}:")
        
        if 'error' in dataset_results:
            print(f"  ❌ Dataset failed to load: {dataset_results['error']}")
            continue
            
        for synth, result in dataset_results.items():
            total_tests += 1
            if result['status'] == 'success':
                successful_tests += 1
                print(f"  ✅ {synth}: {result['total_time']:.1f}s")
            elif result['status'] == 'skipped':
                skipped_tests += 1
                print(f"  ⏭️  {synth}: {result['reason']}")
            else:
                print(f"  ❌ {synth}: {result['error'][:100]}...")
    
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total tests: {total_tests}")
    print(f"  Successful: {successful_tests} ({success_rate:.1%})")
    print(f"  Skipped: {skipped_tests}")
    print(f"  Failed: {total_tests - successful_tests - skipped_tests}")
    
    # Save final results
    final_summary = {
        'comprehensive_benchmark_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_datasets': len(DATASETS),
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'skipped_tests': skipped_tests,
            'failed_tests': total_tests - successful_tests - skipped_tests,
            'success_rate': success_rate,
            'synthesizers_config': SYNTHESIZERS_CONFIG
        },
        'detailed_results': all_results
    }
    
    with open(f'../{output_dir}/comprehensive_benchmark_final.json', 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_dir}/:")
    print(f"  - comprehensive_benchmark_results.json (intermediate)")  
    print(f"  - comprehensive_benchmark_final.json (final summary)")
    print(f"  - synthetic_*.csv files for each successful synthesizer")
    print(f"\n🎉 Comprehensive benchmark completed!")

if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(output_dir)