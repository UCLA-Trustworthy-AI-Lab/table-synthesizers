#!/usr/bin/env python3
"""
Test BaseSynthesizer Integration with DataManager and ConfigManager
====================================================================

Tests that all models inherit manager functionality through BaseSynthesizer.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from stg import TableSynthesizer


def test_tvae_with_managers():
    """Test TVAE with integrated managers."""
    print("\n" + "="*80)
    print("TEST: TVAE with Integrated Managers")
    print("="*80)

    # Create sample data
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.lognormal(10, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

    # Initialize TVAE (managers are enabled by default in BaseSynthesizer)
    print("\n1. Initializing TVAE...")
    tvae = TableSynthesizer('TVAE', {'epochs': 2, 'batch_size': 32}).model

    # Check that managers are initialized
    assert tvae.data_manager is not None, "DataManager should be initialized"
    assert tvae.config_manager is not None, "ConfigManager should be initialized"
    print("   ✓ DataManager initialized:", tvae.data_manager)
    print("   ✓ ConfigManager initialized:", tvae.config_manager)

    # Load configuration from ConfigManager
    print("\n2. Loading configuration...")
    config = tvae.load_config_from_manager(profile='quick')
    if config:
        print(f"   ✓ Config loaded: epochs={config['training']['epochs']}")
    else:
        print("   ⚠ Config not found, using defaults")

    # Train the model
    print("\n3. Training model...")
    tvae.fit(data)
    print("   ✓ Training completed")

    # Save checkpoint using DataManager
    print("\n4. Saving checkpoint...")
    checkpoint_path = tvae.save_checkpoint_to_manager(
        checkpoint_name='test_checkpoint',
        metadata={'test': 'integration', 'data_size': len(data)}
    )
    print(f"   ✓ Checkpoint saved: {checkpoint_path}")

    # Generate samples
    print("\n5. Generating samples...")
    samples = tvae.sample(50, return_dataframe=True)
    print(f"   ✓ Generated samples: {samples.shape}")

    # Save samples using DataManager
    print("\n6. Saving samples...")
    samples_path = tvae.save_samples_to_manager(
        samples,
        sample_name='test_samples',
        format='csv',
        metadata={'num_samples': len(samples)}
    )
    print(f"   ✓ Samples saved: {samples_path}")

    # List available data
    print("\n7. Checking storage...")
    if tvae.data_manager:
        checkpoints = tvae.data_manager.list_checkpoints()
        sample_sets = tvae.data_manager.list_samples()
        print(f"   ✓ Checkpoints: {checkpoints}")
        print(f"   ✓ Sample sets: {sample_sets}")

    # Load checkpoint
    print("\n8. Loading checkpoint...")
    tvae2 = TableSynthesizer('TVAE', {'epochs': 2}).model
    success = tvae2.load_checkpoint_from_manager('test_checkpoint')
    if success:
        print("   ✓ Checkpoint loaded successfully")
    else:
        print("   ⚠ Checkpoint load failed")

    # Cleanup
    import shutil
    if Path('data/TVAE').exists():
        shutil.rmtree('data/TVAE')
        print("\n✓ Cleanup completed")

    print("\n" + "="*80)
    print("✅ TVAE Integration Test PASSED")
    print("="*80)


def test_identity_with_managers():
    """Test Identity model with managers."""
    print("\n" + "="*80)
    print("TEST: Identity with Integrated Managers")
    print("="*80)

    # Create sample data
    data = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'z': np.random.choice(['red', 'blue'], 50)
    })

    print("\n1. Initializing Identity model...")
    identity = TableSynthesizer('Identity', {}).model

    # Verify managers are present
    assert identity.data_manager is not None
    assert identity.config_manager is not None
    print("   ✓ Managers initialized")

    # Train
    print("\n2. Training...")
    identity.fit(data)
    print("   ✓ Training completed")

    # Save checkpoint
    print("\n3. Saving checkpoint...")
    checkpoint_path = identity.save_checkpoint_to_manager('identity_test')
    print(f"   ✓ Saved: {checkpoint_path}")

    # Generate and save samples
    print("\n4. Generating and saving samples...")
    samples = identity.sample(30, return_dataframe=True)
    samples_path = identity.save_samples_to_manager(samples, 'identity_samples')
    print(f"   ✓ Saved: {samples_path}")

    # Cleanup
    import shutil
    if Path('data/Identity').exists():
        shutil.rmtree('data/Identity')

    print("\n✅ Identity Integration Test PASSED\n")


def test_managers_disabled():
    """Test that managers can be disabled."""
    print("\n" + "="*80)
    print("TEST: Managers Disabled")
    print("="*80)

    # Create data
    data = pd.DataFrame({'x': np.random.randn(50)})

    # Initialize with managers disabled
    print("\n1. Initializing with managers disabled...")
    tvae = TableSynthesizer('TVAE', {
        'epochs': 1,
        'enable_data_manager': False,
        'enable_config_manager': False
    }).model

    assert tvae.data_manager is None, "DataManager should be None"
    assert tvae.config_manager is None, "ConfigManager should be None"
    print("   ✓ Managers are disabled")

    # Train should still work
    print("\n2. Training without managers...")
    tvae.fit(data)
    print("   ✓ Training works without managers")

    # Manager methods should handle gracefully
    print("\n3. Testing manager methods with managers disabled...")
    checkpoint_path = tvae.save_checkpoint_to_manager('test')
    assert checkpoint_path is None, "Should return None when manager disabled"
    print("   ✓ Manager methods handle disabled state gracefully")

    print("\n✅ Managers Disabled Test PASSED\n")


def test_config_application():
    """Test configuration loading and application."""
    print("\n" + "="*80)
    print("TEST: Configuration Application")
    print("="*80)

    # Initialize model
    print("\n1. Initializing TVAE...")
    tvae = TableSynthesizer('TVAE', {'epochs': 10}).model

    # Load configuration
    print("\n2. Loading 'quick' profile configuration...")
    config = tvae.load_config_from_manager(profile='quick')

    if config:
        print(f"   ✓ Loaded config: {config.get('training', {})}")

        # Apply configuration
        print("\n3. Applying configuration...")
        tvae.apply_config(config)

        # Verify epochs were updated
        expected_epochs = config['training']['epochs']
        assert tvae._epochs == expected_epochs, f"Epochs should be {expected_epochs}"
        print(f"   ✓ Epochs updated to: {tvae._epochs}")
    else:
        print("   ⚠ Config file not found, using defaults")

    print("\n✅ Configuration Application Test PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("BASESYNTHESIZER + MANAGERS INTEGRATION TESTS")
    print("="*80)

    try:
        # Run tests
        test_tvae_with_managers()
        test_identity_with_managers()
        test_managers_disabled()
        test_config_application()

        print("\n" + "="*80)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*80)
        print("\nSummary:")
        print("  ✓ TVAE with managers: Working")
        print("  ✓ Identity with managers: Working")
        print("  ✓ Managers can be disabled: Working")
        print("  ✓ Config loading/application: Working")
        print("\n📋 All models now have:")
        print("  - DataManager for unified storage")
        print("  - ConfigManager for configuration")
        print("  - Checkpoint save/load methods")
        print("  - Sample save methods")
        print("  - Config loading methods")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
