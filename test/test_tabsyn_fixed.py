#!/usr/bin/env python3
"""
Test TabSyn with downgraded scipy version
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def apply_zero_workaround():
    """Apply the zero workaround"""
    import zero_workaround as zero
    sys.modules['zero'] = zero
    return zero

def test_tabsyn_with_fixed_scipy():
    """Test TabSyn with scipy 1.11.4"""
    print("🔧 Testing TabSyn with scipy 1.11.4 (has _lazywhere)")
    print("=" * 50)

    # Check scipy version
    import scipy
    print(f"📦 Scipy version: {scipy.__version__}")

    # Check _lazywhere availability
    try:
        from scipy._lib._util import _lazywhere
        print("✅ _lazywhere available")
    except ImportError as e:
        print(f"❌ _lazywhere not available: {e}")
        return False

    # Apply workaround
    zero = apply_zero_workaround()
    print("✅ Zero workaround applied")

    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'num': np.random.normal(0, 1, 30),
        'cat': np.random.choice(['A', 'B', 'C'], 30),
        'bin': np.random.choice([0, 1], 30)
    })
    print(f"📊 Test data: {test_data.shape}")

    try:
        from stg.tableSynthesizer import TableSynthesizer

        config = {'epochs': 1, 'batch_size': 8}  # Minimal config for quick test
        print(f"📦 Creating TabSyn with config: {config}")

        model = TableSynthesizer('TabSyn', config)
        print("✅ Model created")

        print(f"🏋️ Training...")
        model.model.fit(test_data)
        print("✅ Training completed")

        print(f"🎲 Sampling...")
        samples = model.model.sample(3, return_dataframe=True)
        print("✅ Sampling completed")

        if isinstance(samples, pd.DataFrame):
            print(f"✅ SUCCESS: Generated {samples.shape}")
            print(f"   Columns match: {set(samples.columns) == set(test_data.columns)}")
            print(f"   Sample data preview:")
            print(samples.head())
            return True
        else:
            print(f"❌ OUTPUT ISSUE: {type(samples)}")
            return False

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tabsyn_with_fixed_scipy()
    print(f"\n📊 Result: {'SUCCESS' if success else 'FAILED'}")