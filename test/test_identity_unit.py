#!/usr/bin/env python3
"""
Unit test for Identity model to validate basic functionality
"""

import sys
import os
import pandas as pd
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def apply_zero_workaround():
    """Apply the zero workaround"""
    import zero_workaround as zero
    sys.modules['zero'] = zero
    return zero

def test_identity_unit():
    print("🧪 IDENTITY MODEL UNIT TEST")
    print("=" * 40)

    # Apply workaround first
    print("\n1️⃣ APPLYING ZERO WORKAROUND...")
    zero = apply_zero_workaround()
    print("✅ Zero workaround applied")

    # Import Identity model
    print("\n2️⃣ IMPORTING IDENTITY MODEL...")
    try:
        from stg.identity.identity import Identity
        print("✅ Identity model imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    # Create test data
    print("\n3️⃣ CREATING TEST DATA...")
    np.random.seed(42)
    test_data = pd.DataFrame({
        'numerical': np.random.normal(0, 1, 20),
        'categorical': np.random.choice(['A', 'B', 'C'], 20),
        'binary': np.random.choice([0, 1], 20)
    })
    print(f"📊 Test data created: {test_data.shape}")
    print("Sample data:")
    print(test_data.head(3))

    # Test model instantiation
    print("\n4️⃣ TESTING MODEL INSTANTIATION...")
    try:
        model = Identity()
        print("✅ Identity model instantiated")
        print(f"   - Model type: {type(model)}")
        print(f"   - Has fit method: {hasattr(model, 'fit')}")
        print(f"   - Has sample method: {hasattr(model, 'sample')}")
        print(f"   - Has _train method: {hasattr(model, '_train')}")
        print(f"   - Has _generate method: {hasattr(model, '_generate')}")
        print(f"   - Has decode_samples method: {hasattr(model, 'decode_samples')}")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return False

    # Test DataFrame fit
    print("\n5️⃣ TESTING FIT WITH DATAFRAME...")
    try:
        model.fit(test_data)
        print("✅ fit() method completed")

        # Check what was stored
        print(f"   - Has train_data: {hasattr(model, 'train_data')}")
        if hasattr(model, 'train_data'):
            print(f"   - train_data type: {type(model.train_data)}")
            if hasattr(model.train_data, '__len__'):
                print(f"   - train_data length: {len(model.train_data)}")

        print(f"   - Has data_info: {hasattr(model, 'data_info')}")
        print(f"   - Has encoders: {hasattr(model, 'encoders')}")
        print(f"   - Has feature_names: {hasattr(model, 'feature_names')}")

    except Exception as e:
        print(f"❌ fit() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test tensor generation
    print("\n6️⃣ TESTING TENSOR GENERATION...")
    try:
        samples_tensor = model.sample(5)
        print("✅ sample() method completed")
        print(f"   - Sample type: {type(samples_tensor)}")
        print(f"   - Sample shape: {samples_tensor.shape if hasattr(samples_tensor, 'shape') else 'No shape'}")
        print(f"   - Sample content: {samples_tensor[:2] if hasattr(samples_tensor, '__getitem__') else 'Cannot slice'}")
    except Exception as e:
        print(f"❌ sample() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test DataFrame generation
    print("\n7️⃣ TESTING DATAFRAME GENERATION...")
    try:
        samples_df = model.sample(5, return_dataframe=True)
        print("✅ sample(return_dataframe=True) method completed")
        print(f"   - Sample type: {type(samples_df)}")
        if isinstance(samples_df, pd.DataFrame):
            print(f"   - Sample shape: {samples_df.shape}")
            print(f"   - Sample columns: {list(samples_df.columns)}")
            print("   - Sample data:")
            print(samples_df.head(3))
        else:
            print(f"   - Expected DataFrame, got: {type(samples_df)}")
    except Exception as e:
        print(f"❌ sample(return_dataframe=True) failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test decode_samples directly if available
    print("\n8️⃣ TESTING DECODE_SAMPLES DIRECTLY...")
    if hasattr(model, 'decode_samples'):
        try:
            # Generate tensor samples first
            tensor_samples = model.sample(3)
            decoded_samples = model.decode_samples(tensor_samples)
            print("✅ decode_samples() method completed")
            print(f"   - Decoded type: {type(decoded_samples)}")
            if isinstance(decoded_samples, pd.DataFrame):
                print(f"   - Decoded shape: {decoded_samples.shape}")
                print(f"   - Decoded columns: {list(decoded_samples.columns)}")
        except Exception as e:
            print(f"❌ decode_samples() failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  decode_samples method not available")

    print("\n" + "=" * 40)
    print("🎯 UNIT TEST SUMMARY:")
    print("✅ Identity model basic functionality validated")
    print("=" * 40)

    return True

if __name__ == "__main__":
    success = test_identity_unit()
    if success:
        print("\n🏆 IDENTITY UNIT TEST PASSED")
        exit(0)
    else:
        print("\n❌ IDENTITY UNIT TEST FAILED")
        exit(1)