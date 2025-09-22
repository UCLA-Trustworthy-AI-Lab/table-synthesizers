#!/usr/bin/env python3
"""
Simple Synthesizer Demo

Quick demonstration of table synthesizers with small sample data.
Perfect for testing and understanding basic functionality.

Usage:
    python simple_demo.py [--synthesizer NAME] [--samples N]

Example:
    python simple_demo.py --synthesizer CTGAN --samples 100
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import time

# Add src to the path
sys.path.insert(0, os.path.abspath('../src'))

from stg.tableSynthesizer import TableSynthesizer

def create_demo_data(n_rows=200):
    """Create diverse demonstration dataset."""
    np.random.seed(42)
    
    # Create realistic mixed data
    ages = np.random.normal(35, 12, n_rows).astype(int)
    ages = np.clip(ages, 18, 80)
    
    incomes = np.random.lognormal(10.5, 0.6, n_rows)
    
    education_levels = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'], 
        n_rows, p=[0.4, 0.35, 0.2, 0.05]
    )
    
    cities = np.random.choice(
        ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Seattle'], 
        n_rows, p=[0.2, 0.18, 0.15, 0.12, 0.1, 0.25]
    )
    
    married = np.random.choice([True, False], n_rows, p=[0.6, 0.4])
    
    # Add some correlations
    education_multiplier = {
        'High School': 0.8, 'Bachelor': 1.0, 
        'Master': 1.4, 'PhD': 1.8
    }
    incomes *= [education_multiplier[ed] for ed in education_levels]
    
    # Age affects income somewhat  
    incomes *= (ages / 40) ** 0.3
    
    credit_scores = np.random.normal(650, 100, n_rows).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)
    
    # Higher income -> better credit
    credit_scores += ((incomes - incomes.mean()) / incomes.std() * 30).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)
    
    df = pd.DataFrame({
        'age': ages,
        'income': incomes,
        'education': education_levels,
        'city': cities,
        'married': married,
        'credit_score': credit_scores
    })
    
    return df

def run_demo(synthesizer_name, n_samples):
    """Run a quick demonstration."""
    print(f"🚀 Table Synthesizer Demo")
    print(f"🔧 Synthesizer: {synthesizer_name}")
    print(f"🔢 Target samples: {n_samples}")
    print("-" * 50)
    
    # Create demo data
    print("📊 Creating demonstration dataset...")
    df = create_demo_data()
    print(f"✅ Created dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📋 Columns: {list(df.columns)}")
    print()
    
    # Show sample of original data
    print("🔍 Original data sample:")
    print(df.head())
    print()
    
    # Initialize synthesizer with reasonable config for demo
    configs = {
        'Identity': {},
        'CTGAN': {'epochs': 50, 'batch_size': 100},
        'TVAE': {'epochs': 50, 'batch_size': 100},
        'TabDDPM': {'num_epochs': 100},
        'CART': {'max_depth': 10, 'random_state': 42},
        'SMOTE': {'k_neighbors': 5, 'random_state': 42},
        'TabSyn': {'epochs': 20}  # Reduced for demo
    }
    
    config = configs.get(synthesizer_name, {})
    
    try:
        print(f"⚙️ Initializing {synthesizer_name}...")
        synthesizer = TableSynthesizer(synthesizer_name, config)
        
        # Training
        print("🎯 Training synthesizer...")
        start_time = time.time()
        synthesizer.fit(df)
        fit_time = time.time() - start_time
        print(f"✅ Training completed in {fit_time:.1f}s")
        
        # Generate synthetic data
        print(f"🔮 Generating {n_samples} synthetic samples...")
        start_time = time.time()
        synthetic_df = synthesizer.sample(n=n_samples, return_dataframe=True)
        gen_time = time.time() - start_time
        print(f"✅ Generation completed in {gen_time:.1f}s")
        
        # Show results
        print()
        print("🎉 RESULTS:")
        print(f"📊 Synthetic data shape: {synthetic_df.shape}")
        print(f"⏱️ Total time: {fit_time + gen_time:.1f}s")
        print()
        
        print("🔍 Synthetic data sample:")
        print(synthetic_df.head())
        print()
        
        # Basic comparison
        print("📈 QUICK COMPARISON:")
        for col in df.select_dtypes(include=[np.number]).columns:
            orig_mean = df[col].mean()
            synth_mean = synthetic_df[col].mean()
            print(f"  {col}: Original μ={orig_mean:.1f}, Synthetic μ={synth_mean:.1f}")
        
        print()
        for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
            orig_unique = df[col].nunique()
            synth_unique = synthetic_df[col].nunique()
            print(f"  {col}: Original {orig_unique} unique, Synthetic {synth_unique} unique")
        
        # Save results
        output_file = f"{synthesizer_name.lower()}_demo_output.csv"
        synthetic_df.to_csv(output_file, index=False)
        print(f"\n💾 Synthetic data saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Try a different synthesizer or check your environment")
        return 1
    
    print("\n🎯 Demo completed successfully!")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Simple synthesizer demonstration")
    parser.add_argument("--synthesizer", default="CTGAN",
                       choices=['Identity', 'CTGAN', 'TVAE', 'TabDDPM', 'CART', 'SMOTE', 'TabSyn'],
                       help="Synthesizer to demonstrate (default: CTGAN)")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of synthetic samples (default: 100)")
    
    args = parser.parse_args()
    
    return run_demo(args.synthesizer, args.samples)

if __name__ == "__main__":
    sys.exit(main())