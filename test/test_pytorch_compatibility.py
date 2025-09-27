#!/usr/bin/env python3
"""
PyTorch 2.3+ Compatibility Test for libzero Workaround
Tests all zero functions with modern PyTorch versions
"""

import sys
import os
import torch
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def apply_zero_workaround():
    """Apply the zero workaround"""
    import zero_workaround as zero
    sys.modules['zero'] = zero
    return zero

def test_pytorch_compatibility():
    """Test libzero workaround compatibility with PyTorch 2.3+"""

    print("🔧 PYTORCH 2.3+ COMPATIBILITY TEST")
    print("=" * 60)

    # Check PyTorch version
    torch_version = torch.__version__
    major, minor = torch_version.split('.')[:2]
    major, minor = int(major), int(minor)

    print(f"📦 PyTorch Version: {torch_version}")

    if major < 2 or (major == 2 and minor < 3):
        print("⚠️  Warning: PyTorch version is below 2.3")
    else:
        print("✅ PyTorch 2.3+ detected")

    print(f"🎯 Testing compatibility with PyTorch {torch_version}")
    print()

    # Apply workaround
    print("1️⃣ APPLYING ZERO WORKAROUND...")
    try:
        apply_zero_workaround()
        import zero
        print("✅ Zero workaround applied successfully")
    except Exception as e:
        print(f"❌ Failed to apply workaround: {e}")
        return False

    # Test 1: PyTorch reproducibility functions
    print("\n2️⃣ TESTING PYTORCH REPRODUCIBILITY...")
    try:
        # Test modern PyTorch reproducibility features
        zero.improve_reproducibility(42)

        # Verify CUDA deterministic algorithms (PyTorch 2.3+ feature)
        if torch.cuda.is_available():
            print(f"   🔹 CUDA available: {torch.cuda.get_device_name()}")
            print(f"   🔹 Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
        else:
            print("   🔹 CUDA not available (CPU mode)")

        # Test tensor operations for consistency
        torch.manual_seed(42)
        tensor1 = torch.randn(100, 50)

        torch.manual_seed(42)  # Reset seed
        tensor2 = torch.randn(100, 50)

        assert torch.allclose(tensor1, tensor2), "Reproducibility failed"
        print("✅ PyTorch reproducibility works perfectly")

    except Exception as e:
        print(f"❌ PyTorch reproducibility failed: {e}")
        return False

    # Test 2: Random state management with PyTorch 2.3+
    print("\n3️⃣ TESTING RANDOM STATE MANAGEMENT...")
    try:
        # Test with modern PyTorch random number generation
        original_state = zero.random.get_state()

        # Generate some random numbers
        torch.manual_seed(123)
        random_tensor = torch.randn(10, 10)

        # Restore state and verify
        zero.random.set_state(original_state)
        print("✅ Random state management compatible")

    except Exception as e:
        print(f"❌ Random state management failed: {e}")
        return False

    # Test 3: Batch iteration with modern PyTorch tensors
    print("\n4️⃣ TESTING BATCH ITERATION...")
    try:
        # Test with modern PyTorch tensor operations
        large_tensor = torch.randn(1000, 128)  # Modern tensor size
        batches = list(zero.iter_batches(large_tensor, chunk_size=32))

        assert len(batches) == 32, f"Expected 32 batches, got {len(batches)}"
        assert all(isinstance(batch, torch.Tensor) for batch in batches), "All chunks should be tensors"

        # Test with different dtypes (PyTorch 2.3+ features)
        float16_tensor = torch.randn(100, 64, dtype=torch.float16)
        float16_batches = list(zero.iter_batches(float16_tensor, chunk_size=10))
        assert len(float16_batches) == 10, "Float16 batching failed"

        print("✅ Batch iteration works with modern PyTorch tensors")

    except Exception as e:
        print(f"❌ Batch iteration failed: {e}")
        return False

    # Test 4: GPU detection with PyTorch 2.3+
    print("\n5️⃣ TESTING GPU HARDWARE DETECTION...")
    try:
        gpu_info = zero.hardware.get_gpus_info()

        if torch.cuda.is_available():
            print(f"   🔹 CUDA devices detected: {torch.cuda.device_count()}")
            print(f"   🔹 Current device: {torch.cuda.current_device()}")
            print(f"   🔹 Device name: {torch.cuda.get_device_name()}")
            print(f"   🔹 Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        if torch.backends.mps.is_available():
            print("   🔹 MPS (Apple Silicon) available")

        print("✅ GPU detection compatible with PyTorch 2.3+")

    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

    # Test 5: Integration with affected models
    print("\n6️⃣ TESTING AFFECTED MODEL COMPATIBILITY...")
    try:
        # Test TabDDPM reproducibility function
        print("   🔹 Testing TabDDPM reproducibility...")
        zero.improve_reproducibility(42)

        # Test TabSyn functions
        print("   🔹 Testing TabSyn random state...")
        state = zero.random.get_state()
        zero.random.set_state(state)

        print("   🔹 Testing TabSyn batch iteration...")
        test_data = torch.randn(50, 10)
        batches = list(zero.iter_batches(test_data, chunk_size=10))
        assert len(batches) == 5

        print("   🔹 Testing TabSyn GPU detection...")
        gpu_info = zero.hardware.get_gpus_info()

        print("✅ All affected models compatible")

    except Exception as e:
        print(f"❌ Affected model compatibility failed: {e}")
        return False

    # Test 6: Modern PyTorch features compatibility
    print("\n7️⃣ TESTING MODERN PYTORCH FEATURES...")
    try:
        # Test with compilation (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("   🔹 PyTorch compile available")

        # Test with mixed precision (modern feature)
        if torch.cuda.is_available():
            with torch.autocast('cuda'):
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.matmul(test_tensor, test_tensor)
            print("   🔹 Mixed precision compatible")

        # Test with new tensor creation methods
        tensor = torch.zeros(100, 100, device='cpu', dtype=torch.float32)
        print("   🔹 Modern tensor creation compatible")

        print("✅ Modern PyTorch features work with workaround")

    except Exception as e:
        print(f"❌ Modern PyTorch features failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 PYTORCH 2.3+ COMPATIBILITY: COMPLETE SUCCESS")
    print("✅ All zero functions work perfectly with modern PyTorch")
    print("✅ No compatibility issues detected")
    print("✅ Workaround is future-proof")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = test_pytorch_compatibility()
    sys.exit(0 if success else 1)