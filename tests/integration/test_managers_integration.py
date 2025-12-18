#!/usr/bin/env python3
"""
Integration Test for DataManager, ConfigManager, and WandbManager
==================================================================

Tests the unified management system for table synthesizers.

Usage:
    python test/test_managers_integration.py
    python test/test_managers_integration.py --with-wandb  # Test wandb integration
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg.data_manager import DataManager
from stg.config_manager import ConfigManager
from stg.wandb_manager import WandbManager


def test_data_manager():
    """Test DataManager functionality."""
    print("\n" + "="*80)
    print("TESTING DataManager")
    print("="*80)

    # Initialize DataManager
    dm = DataManager('TVAE', base_data_dir='test_data_temp')
    print(f"\n✓ DataManager initialized: {dm}")

    # Test 1: Save and load preprocessed data
    print("\n[Test 1] Preprocessed Data Storage")
    test_data = {
        'features': np.random.randn(100, 10),
        'labels': np.random.randint(0, 2, 100)
    }
    metadata = {'num_features': 10, 'dataset': 'test'}

    saved_path = dm.save_preprocessed_data(test_data, 'test_dataset', metadata)
    print(f"  ✓ Saved preprocessed data: {saved_path}")

    loaded_data = dm.load_preprocessed_data('test_dataset')
    assert loaded_data is not None, "Failed to load preprocessed data"
    assert 'data' in loaded_data and 'metadata' in loaded_data
    print(f"  ✓ Loaded preprocessed data with metadata: {loaded_data['metadata']}")

    datasets = dm.list_preprocessed_datasets()
    print(f"  ✓ Available datasets: {datasets}")

    # Test 2: Save and load checkpoints
    print("\n[Test 2] Checkpoint Management")
    checkpoint_data = {
        'model_state': torch.randn(100, 50),
        'optimizer_state': {'lr': 0.001},
        'epoch': 10
    }
    checkpoint_metadata = {'loss': 0.15, 'epoch': 10}

    saved_checkpoint = dm.save_checkpoint(checkpoint_data, 'epoch_10', checkpoint_metadata)
    print(f"  ✓ Saved checkpoint: {saved_checkpoint}")

    loaded_checkpoint = dm.load_checkpoint('epoch_10')
    assert loaded_checkpoint is not None, "Failed to load checkpoint"
    assert loaded_checkpoint['metadata']['epoch'] == 10
    print(f"  ✓ Loaded checkpoint: epoch={loaded_checkpoint['metadata']['epoch']}, loss={loaded_checkpoint['metadata']['loss']}")

    checkpoints = dm.list_checkpoints()
    print(f"  ✓ Available checkpoints: {checkpoints}")

    # Test 3: Save and load samples
    print("\n[Test 3] Sample Storage")
    # Test CSV format
    samples_df = pd.DataFrame({
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'category': np.random.choice(['A', 'B', 'C'], 50)
    })
    sample_metadata = {'num_samples': 50, 'generation_params': {'temperature': 1.0}}

    saved_csv = dm.save_samples(samples_df, 'synthetic_csv', format='csv', metadata=sample_metadata)
    print(f"  ✓ Saved CSV samples: {saved_csv}")

    loaded_csv = dm.load_samples('synthetic_csv', format='csv')
    assert loaded_csv is not None and len(loaded_csv) == 50
    print(f"  ✓ Loaded CSV samples: {loaded_csv.shape}")

    # Test pickle format
    saved_pkl = dm.save_samples(samples_df, 'synthetic_pkl', format='pickle')
    loaded_pkl = dm.load_samples('synthetic_pkl', format='pickle')
    assert loaded_pkl is not None
    print(f"  ✓ Saved and loaded pickle samples")

    samples = dm.list_samples()
    print(f"  ✓ Available sample sets: {samples}")

    # Test 4: Temporary directory
    print("\n[Test 4] Temporary File Management")
    temp_dir = dm.get_temp_dir()
    print(f"  ✓ Temp directory: {temp_dir}")

    # Create a temp file
    temp_file = temp_dir / 'test_temp.txt'
    temp_file.write_text('temporary data')
    assert temp_file.exists()
    print(f"  ✓ Created temp file: {temp_file}")

    dm.cleanup_temp_files()
    assert not temp_file.exists()
    print(f"  ✓ Cleaned temp files")

    # Test 5: Storage info
    print("\n[Test 5] Storage Information")
    storage_info = dm.get_storage_info()
    print(f"  ✓ Storage info:")
    print(f"    - Model: {storage_info['model_name']}")
    print(f"    - Preprocessed: {storage_info['storage']['preprocessed']['count']} datasets, {storage_info['storage']['preprocessed']['size_mb']:.2f} MB")
    print(f"    - Checkpoints: {storage_info['storage']['checkpoints']['count']} files, {storage_info['storage']['checkpoints']['size_mb']:.2f} MB")
    print(f"    - Samples: {storage_info['storage']['samples']['count']} sets, {storage_info['storage']['samples']['size_mb']:.2f} MB")

    # Cleanup test directory
    import shutil
    shutil.rmtree('test_data_temp')
    print("\n✓ All DataManager tests passed!")


def test_config_manager():
    """Test ConfigManager functionality."""
    print("\n" + "="*80)
    print("TESTING ConfigManager")
    print("="*80)

    # Initialize ConfigManager
    cm = ConfigManager()
    print(f"\n✓ ConfigManager initialized: {cm}")

    # Test 1: Load configuration with profile
    print("\n[Test 1] Load Configuration with Profiles")
    default_config = cm.load_config('TVAE', profile='default')
    print(f"  ✓ Default config: {default_config['training']}")

    quick_config = cm.load_config('TVAE', profile='quick')
    print(f"  ✓ Quick config: {quick_config['training']}")

    production_config = cm.load_config('TVAE', profile='production')
    print(f"  ✓ Production config: {production_config['training']}")

    # Verify profile differences
    assert default_config['training']['epochs'] == 100
    assert quick_config['training']['epochs'] == 2
    assert production_config['training']['epochs'] == 300
    print(f"  ✓ Profiles correctly differentiated")

    # Test 2: List available profiles
    print("\n[Test 2] Available Profiles")
    profiles = cm.get_available_profiles('TVAE')
    print(f"  ✓ TVAE profiles: {profiles}")

    # Test 3: Load and apply template
    print("\n[Test 3] Configuration Templates")
    quick_test_config = cm.load_config_with_template('TVAE', 'quick_test')
    assert quick_test_config['training']['epochs'] == 2
    print(f"  ✓ Applied 'quick_test' template: epochs={quick_test_config['training']['epochs']}")

    production_template = cm.load_config_with_template('TVAE', 'production')
    assert production_template['training']['epochs'] == 300
    print(f"  ✓ Applied 'production' template: epochs={production_template['training']['epochs']}")

    # Test 4: Environment variable overrides
    print("\n[Test 4] Environment Variable Overrides")
    # Set environment variables
    os.environ['TABLE_SYNTH_EPOCHS'] = '150'
    os.environ['TABLE_SYNTH_TVAE_EMBEDDING_DIM'] = '256'

    env_config = cm.load_config('TVAE', profile='default')
    print(f"  ✓ Config with env overrides:")
    print(f"    - Epochs (from TABLE_SYNTH_EPOCHS): {env_config['training'].get('epochs', 'not set')}")
    print(f"    - Embedding dim (from TABLE_SYNTH_TVAE_EMBEDDING_DIM): {env_config['architecture'].get('embedding_dim', 'not set')}")

    # Cleanup env vars
    del os.environ['TABLE_SYNTH_EPOCHS']
    del os.environ['TABLE_SYNTH_TVAE_EMBEDDING_DIM']

    # Test 5: Create custom profile
    print("\n[Test 5] Custom Profile Creation")
    cm.create_custom_profile(
        'TVAE',
        base_profile='default',
        custom_profile='test_custom',
        overrides={'training': {'epochs': 50, 'batch_size': 64}}
    )
    custom_config = cm.load_config('TVAE', profile='test_custom')
    assert custom_config['training']['epochs'] == 50
    assert custom_config['training']['batch_size'] == 64
    print(f"  ✓ Created custom profile: epochs={custom_config['training']['epochs']}, batch_size={custom_config['training']['batch_size']}")

    # Test 6: Configuration validation
    print("\n[Test 6] Configuration Validation")
    valid_config = {
        'training': {
            'epochs': 100,
            'batch_size': 32
        }
    }
    is_valid, errors = cm.validate_config('TVAE', valid_config)
    assert is_valid
    print(f"  ✓ Valid config validated successfully")

    invalid_config = {
        'training': {
            'epochs': 100
            # Missing batch_size
        }
    }
    is_valid, errors = cm.validate_config('TVAE', invalid_config)
    assert not is_valid
    print(f"  ✓ Invalid config detected: {errors}")

    print("\n✓ All ConfigManager tests passed!")


def test_wandb_manager(enable_wandb=False):
    """Test WandbManager functionality."""
    print("\n" + "="*80)
    print("TESTING WandbManager")
    print("="*80)

    if not enable_wandb:
        print("\n⚠️  WandB testing disabled (use --with-wandb to enable)")
        print("   Testing configuration loading only...")

    # Initialize WandbManager
    wm = WandbManager('TVAE', config_dir='config/wandb')
    print(f"\n✓ WandbManager initialized for TVAE")

    # Test 1: Configuration loading
    print("\n[Test 1] Configuration Loading")
    print(f"  ✓ Global config: {wm.global_config.get('project_name', 'N/A')}")
    print(f"  ✓ Model config metrics: {list(wm.model_config.get('metrics', {}).keys())}")
    print(f"  ✓ Hyperparameters: {wm.model_config.get('hyperparameters', [])}")

    if enable_wandb:
        # Test 2: Initialize experiment
        print("\n[Test 2] Initialize Experiment")
        config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'embedding_dim': 128
        }
        run = wm.init_experiment(config, tags=['test', 'integration'], notes='Integration test run')
        print(f"  ✓ Initialized wandb run: {run.name}")

        # Test 3: Log metrics
        print("\n[Test 3] Log Metrics")
        for epoch in range(3):
            metrics = {
                'train_loss': 0.5 - epoch * 0.1,
                'epoch_time': 10.5
            }
            wm.log_metrics(metrics, step=epoch, phase='training')
            print(f"  ✓ Logged epoch {epoch}: loss={metrics['train_loss']:.2f}")

        # Test 4: Finish experiment
        print("\n[Test 4] Finish Experiment")
        wm.finish_experiment()
        print(f"  ✓ Finished wandb run")

    print("\n✓ All WandbManager tests passed!")


def test_integrated_workflow():
    """Test integrated workflow with all managers."""
    print("\n" + "="*80)
    print("TESTING Integrated Workflow")
    print("="*80)

    # Initialize all managers
    dm = DataManager('TVAE', base_data_dir='test_data_temp')
    cm = ConfigManager()

    print("\n[Integrated Workflow] Training Simulation")

    # Step 1: Load configuration
    config = cm.load_config('TVAE', profile='quick')
    print(f"  1. ✓ Loaded config: epochs={config['training']['epochs']}, batch_size={config['training']['batch_size']}")

    # Step 2: Prepare and save data
    train_data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    dm.save_preprocessed_data({'train': train_data}, 'training_data', metadata={'size': len(train_data)})
    print(f"  2. ✓ Saved preprocessed training data: {len(train_data)} samples")

    # Step 3: Simulate training epochs
    for epoch in range(config['training']['epochs']):
        # Simulate training
        loss = 1.0 - epoch * 0.3

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': torch.randn(10, 10),
            'loss': loss
        }
        dm.save_checkpoint(checkpoint, f'epoch_{epoch}', metadata={'epoch': epoch, 'loss': loss})
        print(f"  3. ✓ Epoch {epoch}: loss={loss:.3f}, saved checkpoint")

    # Step 4: Generate and save samples
    synthetic_samples = pd.DataFrame({
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'category': np.random.choice(['A', 'B', 'C'], 50)
    })
    dm.save_samples(synthetic_samples, 'final_synthetic', format='csv',
                   metadata={'num_samples': 50, 'model': 'TVAE'})
    print(f"  4. ✓ Generated and saved {len(synthetic_samples)} synthetic samples")

    # Step 5: Get storage summary
    storage_info = dm.get_storage_info()
    print(f"  5. ✓ Storage summary:")
    print(f"     - Preprocessed: {storage_info['storage']['preprocessed']['count']} datasets")
    print(f"     - Checkpoints: {storage_info['storage']['checkpoints']['count']} files")
    print(f"     - Samples: {storage_info['storage']['samples']['count']} sets")

    # Cleanup
    import shutil
    shutil.rmtree('test_data_temp')
    print("\n✓ Integrated workflow test passed!")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Test DataManager, ConfigManager, and WandbManager')
    parser.add_argument('--with-wandb', action='store_true', help='Enable WandB integration tests')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("TABLE SYNTHESIZERS - MANAGEMENT SYSTEM INTEGRATION TESTS")
    print("="*80)

    try:
        # Run individual component tests
        test_data_manager()
        test_config_manager()
        test_wandb_manager(enable_wandb=args.with_wandb)

        # Run integrated workflow test
        test_integrated_workflow()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nManagement System Summary:")
        print("  ✓ DataManager: Unified data storage (preprocessed/checkpoints/samples)")
        print("  ✓ ConfigManager: Configuration profiles and templates")
        print("  ✓ WandbManager: Experiment tracking integration")
        print("  ✓ Integrated workflow: All managers working together")
        print("\nReady for model integration!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
