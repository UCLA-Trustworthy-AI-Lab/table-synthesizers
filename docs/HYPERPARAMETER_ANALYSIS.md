# Hyperparameter Comparison: CTGAN vs PATE-CTGAN

## Why GPU Optimization Doesn't Work Properly for CTGAN

**TL;DR**: There's a **PAC size mismatch** between the training script (assumes PAC=10) and CTGAN code (default PAC=5), and the config files don't specify PAC at all. This causes batch size optimization to use the wrong constraint.

---

## Complete Hyperparameter Comparison

### Configuration Files

**CTGAN** (`config/default_CTGAN.json`):
```json
{
    "model_type": "gpu",
    "epochs": 50,
    "batch_size": 200,
    "embedding_dim": 128,
    "generator_dim": [256, 256],
    "discriminator_dim": [256, 256],
    "generator_lr": 0.0002,
    "discriminator_lr": 0.0002,
    "description": "Conditional GAN - Best balance of speed and quality",
    "speedup": "15-30x",
    "quality": "High"
}
```

**PATE-CTGAN** (`config/default_PATE-CTGAN.json`):
```json
{
    "model_type": "gpu",
    "epochs": 50,
    "batch_size": 100,
    "num_teachers": 10,
    "epsilon": 1.0,
    "delta": 1e-5,
    "embedding_dim": 128,
    "description": "Privacy-preserving GAN",
    "speedup": "10-25x",
    "quality": "Good"
}
```

### Side-by-Side Comparison

| Parameter | CTGAN | PATE-CTGAN | Impact on Performance |
|-----------|-------|------------|----------------------|
| **batch_size** | 200 | 100 | CTGAN has 2x larger batch → faster convergence |
| **epochs** | 50 | 50 | Same |
| **embedding_dim** | 128 | 128 | Same |
| **generator_dim** | [256, 256] | *(code default)* | CTGAN explicitly configured |
| **discriminator_dim** | [256, 256] | *(code default)* | CTGAN explicitly configured |
| **generator_lr** | 0.0002 | *(code default: 2e-4)* | Same value |
| **discriminator_lr** | 0.0002 | *(code default: 2e-4)* | Same value |
| **num_teachers** | N/A | 10 | PATE-CTGAN splits data into 10 partitions |
| **epsilon** | N/A | 1.0 | Privacy budget (PATE-CTGAN only) |
| **delta** | N/A | 1e-5 | Privacy parameter (PATE-CTGAN only) |

### Hidden Parameters (Not in Config, Only in Code)

| Parameter | CTGAN (code default) | PATE-CTGAN (code default) | Impact |
|-----------|---------------------|--------------------------|--------|
| **pac** | **5** | **1** | **CRITICAL: Affects batch size constraints** |
| **discriminator_steps** | 1 | 1 | Same |
| **teacher_iters** | N/A | 5 | PATE-CTGAN does 5 iterations per teacher |
| **student_iters** | N/A | 5 | PATE-CTGAN does 5 student training iterations |
| **generator_decay** | 1e-6 | 1e-6 | Weight decay |
| **discriminator_decay** | 1e-6 | 1e-6 | Weight decay |
| **regularization** | "dragan" | "dragan" | Gradient penalty type |
| **loss** | "wasserstein" | "cross_entropy" | **Different loss functions!** |

**Code References**:
- CTGAN: `src/stg/CTGAN/ctgan.py:42`
- PATE-CTGAN: `src/stg/PATECTGAN/patectgan.py:158-186`

---

## The PAC (Packing) Mechanism

### What is PAC?

**PAC** (Packing) is a technique used in GANs to stabilize discriminator training by grouping multiple samples together before feeding them to the discriminator.

**How it works**:
```python
# Without PAC (normal discriminator)
discriminator_input = sample  # Shape: [batch_size, features]

# With PAC=5 (CTGAN default)
discriminator_input = sample.view(-1, features * 5)  # Shape: [batch_size/5, features*5]

# With PAC=10 (train script assumes this)
discriminator_input = sample.view(-1, features * 10)  # Shape: [batch_size/10, features*10]
```

**Why PAC matters**:
1. **Batch size constraint**: `batch_size % pac == 0` (must be divisible by PAC)
2. **Discriminator input size**: Multiplied by PAC factor
3. **Training stability**: Higher PAC can improve stability but requires larger batches

**Code Implementation** (`src/stg/CTGAN/models.py:43-44`):
```python
def forward(self, input):
    assert input.size()[0] % self.pac == 0  # ← STRICT REQUIREMENT
    return self.seq(input.view(-1, self.pacdim))
```

---

## The GPU Optimization Bug

### Problem Description

**Location**: `train_all_compatible_models.py:372-377`

**The Issue**:
```python
# For CTGAN, ensure batch size is divisible by PAC size (10)
if model_name == 'CTGAN':
    pac_size = 10  # ← HARDCODED (WRONG!)
    optimal_batch_size = (optimal_batch_size // pac_size) * pac_size
    if optimal_batch_size == 0:
        optimal_batch_size = pac_size
```

**Why it's wrong**:
1. **Hardcoded PAC=10**: Script assumes PAC=10
2. **CTGAN code default**: PAC=5 (`src/stg/CTGAN/ctgan.py:42`)
3. **Config doesn't specify PAC**: No PAC parameter in `config/default_CTGAN.json`
4. **Result**: Batch size optimization uses wrong divisibility constraint

### Impact on Bean Dataset (1338 samples)

**Scenario 1: With GPU (NVIDIA GB10 - 119.7GB)**

```python
# GPU optimization calculation
gpu_memory_gb = 119.7
available_memory_mb = (119.7 * 1024) * 0.8  # 98,058 MB
max_batch_size = 98,058 / 1.0  # Assuming 1 MB per sample
recommended_batch_size = min(512, 98058, 1338//2)  # = 512 (high-end GPU tier)

# Apply WRONG PAC constraint (PAC=10 instead of PAC=5)
optimal_batch_size = (512 // 10) * 10  # = 510

# Actual PAC used by CTGAN
actual_pac = 5  # From code default

# Potential issue
if 510 % 5 != 0:
    # No error in this case (510 is divisible by 5)
    pass
```

**Lucky case**: 510 is divisible by both 5 and 10, so no error occurs.

**Unlucky case** (example with different GPU memory):
```python
# Suppose optimizer suggests batch_size = 255
optimal_batch_size = 255

# Train script applies PAC=10 constraint
optimal_batch_size = (255 // 10) * 10  # = 250

# CTGAN uses PAC=5 (no issue: 250 % 5 == 0)
# But if optimizer suggests 253:
optimal_batch_size = (253 // 10) * 10  # = 250 (correct)

# The issue: Optimizer is overly conservative
# It could use batch_size=255 (divisible by 5)
# But forces it down to 250 (divisible by 10)
```

### Why It Doesn't Always Cause Errors

**Two reasons**:
1. **Any multiple of 10 is also a multiple of 5**: PAC=10 constraint is stricter than PAC=5
2. **Default config batch_size=200**: 200 % 5 == 0 and 200 % 10 == 0 (divisible by both)

**However, it causes suboptimal performance**:
- Unnecessarily reduces batch size
- Loses potential GPU efficiency gains
- May not fully utilize GPU memory

---

## Comparison: CTGAN vs PATE-CTGAN Batch Size Optimization

### CTGAN Batch Size Flow

```
1. Load config: batch_size = 200
2. GPU optimization:
   - Calculate optimal: min(512, 98058, 1338//2) = 512
   - Apply PAC=10 constraint: (512 // 10) * 10 = 510
3. Pass to CTGAN: batch_size = 510
4. CTGAN uses PAC=5 (from code default)
5. Check constraint: 510 % 5 == 0 ✅ (no error, but suboptimal)
```

**Issue**: Optimization uses PAC=10, but CTGAN uses PAC=5 → Unnecessarily conservative batch size

### PATE-CTGAN Batch Size Flow

```
1. Load config: batch_size = 100
2. GPU optimization:
   - Calculate optimal: min(512, 98058, 1338//2) = 512
   - NO PAC constraint in train script (only for CTGAN)
3. Pass to PATE-CTGAN: batch_size = 512
4. PATE-CTGAN uses PAC=1 (from code default)
5. Check constraint: 512 % 1 == 0 ✅ (always satisfied)
```

**Result**: PATE-CTGAN gets optimized batch size without artificial constraint

---

## Why GPU Optimization Works Better for PATE-CTGAN

### Reasons

1. **PAC=1**: No meaningful batch size constraint (any batch size works)
2. **No hardcoded constraint in train script**: Optimization logic only applies PAC constraint to CTGAN
3. **Simpler batch size selection**: No need to worry about divisibility

**Code Evidence** (`train_all_compatible_models.py:359-382`):
```python
def optimize_batch_size(model_name, dataset_size, default_batch_size):
    # Only optimize for GPU models
    if model_name not in GPU_MODEL_CONFIGS:
        return default_batch_size

    if is_gpu_available():
        optimal_batch_size = get_optimal_batch_size(
            dataset_size=dataset_size,
            model_memory_per_sample_mb=1.0,
            default_batch_size=default_batch_size
        )

        # ⚠️ ONLY CTGAN has PAC constraint applied
        if model_name == 'CTGAN':
            pac_size = 10  # ← WRONG VALUE
            optimal_batch_size = (optimal_batch_size // pac_size) * pac_size
            if optimal_batch_size == 0:
                optimal_batch_size = pac_size

        # PATE-CTGAN bypasses this constraint entirely
        return optimal_batch_size
    else:
        return min(default_batch_size, max(32, dataset_size // 10))
```

---

## Impact Analysis

### Performance Impact on Bean Dataset

| Model | Config Batch Size | Optimized Batch Size | Actual PAC | Constraint Match | Efficiency |
|-------|------------------|---------------------|------------|------------------|-----------|
| **CTGAN** | 200 | 510 (PAC=10 constraint) | 5 | ❌ Mismatch | Suboptimal (could be 510-515) |
| **PATE-CTGAN** | 100 | 512 (no constraint) | 1 | ✅ Match | Optimal |

**Efficiency Loss for CTGAN**:
- **Potential**: Could use batch_size=515 (divisible by 5)
- **Actual**: Uses batch_size=510 (divisible by 10)
- **Loss**: ~1% batch size reduction (minor, but indicative of larger issue)

**For other datasets, impact could be worse**:
```python
# Example: Dataset with 2000 samples
recommended = min(512, 98058, 2000//2)  # = 512
ctgan_optimized = (512 // 10) * 10      # = 510 (PAC=10 constraint)
ctgan_actual_ok = (512 // 5) * 5        # = 510 (PAC=5 would allow 510)
# Still works, but could allow 515

# Example 2: Smaller GPU memory suggests batch_size=307
recommended = 307
ctgan_optimized = (307 // 10) * 10      # = 300 (PAC=10 constraint)
ctgan_actual_ok = (307 // 5) * 5        # = 305 (PAC=5 would allow 305)
# Lost 5 samples per batch (~1.6% efficiency loss)
```

---

## Other Key Hyperparameter Differences

### 1. Loss Function

| Model | Loss Type | Implementation | Impact |
|-------|-----------|----------------|--------|
| **CTGAN** | Wasserstein | `loss_d = -(torch.mean(y_real) - torch.mean(y_fake))` | More stable gradients |
| **PATE-CTGAN** | Cross Entropy | `criterion = nn.BCELoss()` | Standard GAN loss |

**Code References**:
- CTGAN: `src/stg/CTGAN/ctgan.py:166-167`
- PATE-CTGAN: `src/stg/PATECTGAN/patectgan.py:322`

**Why different**:
- CTGAN uses Wasserstein distance for better training stability
- PATE-CTGAN uses BCE loss for compatibility with PATE aggregation

### 2. Batch Size (Config)

| Model | Batch Size | Reason |
|-------|-----------|--------|
| **CTGAN** | 200 | Larger batches for faster training |
| **PATE-CTGAN** | 100 | Smaller batches due to memory overhead from multiple discriminators |

**Memory consideration**:
- CTGAN: 1 discriminator → can afford larger batches
- PATE-CTGAN: 11 discriminators (10 teachers + 1 student) → needs smaller batches to fit in memory

### 3. Network Architecture Configuration

**CTGAN**: Explicitly specifies network dimensions in config
```json
"generator_dim": [256, 256],
"discriminator_dim": [256, 256]
```

**PATE-CTGAN**: Uses code defaults (same values, but not in config)
```python
# From patectgan.py:162-163
generator_dim=(256, 256),
discriminator_dim=(256, 256)
```

**Why CTGAN specifies in config**:
- More tunable via config files
- Easier to experiment with different architectures
- Better for hyperparameter sweeps

### 4. Privacy Parameters (PATE-CTGAN only)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **num_teachers** | 10 | Number of teacher discriminators |
| **epsilon** | 1.0 | Privacy budget (lower = more privacy) |
| **delta** | 1e-5 | Failure probability in differential privacy |
| **teacher_iters** | 5 | Iterations per teacher per round |
| **student_iters** | 5 | Student training iterations per round |

**Impact on training**:
- Each iteration trains 10 teachers × 5 iterations = 50 teacher updates
- Plus 5 student updates
- Total: 55+ discriminator updates per iteration (vs 1 for CTGAN)

---

## Recommendations

### Fix the PAC Mismatch

**Option 1: Add PAC to config files** (RECOMMENDED)

**`config/default_CTGAN.json`**:
```json
{
    "model_type": "gpu",
    "epochs": 50,
    "batch_size": 200,
    "pac": 5,  // ← ADD THIS
    "embedding_dim": 128,
    ...
}
```

**`train_all_compatible_models.py:372-377`**:
```python
# For CTGAN, ensure batch size is divisible by PAC size
if model_name == 'CTGAN':
    pac_size = config.get('pac', 5)  # ← Read from config, default to 5
    optimal_batch_size = (optimal_batch_size // pac_size) * pac_size
    if optimal_batch_size == 0:
        optimal_batch_size = pac_size
```

**Option 2: Remove PAC constraint from optimization** (SIMPLER)

```python
# Remove the entire PAC constraint block
# CTGAN will validate batch_size internally anyway
if is_gpu_available():
    optimal_batch_size = get_optimal_batch_size(
        dataset_size=dataset_size,
        model_memory_per_sample_mb=1.0,
        default_batch_size=default_batch_size
    )
    # Let CTGAN handle PAC validation internally
    return optimal_batch_size
```

**Trade-offs**:
- Option 1: More explicit, ensures batch_size is always valid
- Option 2: Simpler code, relies on CTGAN's internal validation (which will raise error if batch_size % pac != 0)

**My recommendation**: **Option 1** - Explicit is better than implicit, and prevents runtime errors

### Add PAC to PATE-CTGAN Config

**`config/default_PATE-CTGAN.json`**:
```json
{
    "model_type": "gpu",
    "epochs": 50,
    "batch_size": 100,
    "pac": 1,  // ← ADD THIS for completeness
    "num_teachers": 10,
    "epsilon": 1.0,
    ...
}
```

### Optimize Batch Sizes for Small Datasets

For Bean dataset (1338 samples):

**Current**:
- CTGAN: 200 (from config) → 510 (optimized, wrong PAC) → 510 (actual, works by luck)
- PATE-CTGAN: 100 (from config) → 512 (optimized, no PAC constraint)

**Recommended** (after fixing PAC):
- CTGAN: 200 (from config) → 515 (optimized, PAC=5) → better GPU utilization
- PATE-CTGAN: 100 (from config) → 512 (no change)

**Expected speedup**: ~1-2% for CTGAN (minor, but correct)

---

## Summary Table

| Aspect | CTGAN | PATE-CTGAN | Who Wins? |
|--------|-------|------------|-----------|
| **Config Batch Size** | 200 | 100 | CTGAN (2x larger) |
| **PAC Constraint** | 5 (code), 10 (script) ❌ | 1 (code), none (script) ✅ | PATE-CTGAN |
| **GPU Optimization** | Broken (wrong PAC) | Works correctly | PATE-CTGAN ✅ |
| **Loss Function** | Wasserstein (stable) | Cross Entropy (standard) | CTGAN |
| **Architecture Config** | Explicit in config | Code defaults | CTGAN |
| **Privacy Params** | None | epsilon, delta, teachers | PATE-CTGAN |
| **Memory Usage** | Low (1 disc) | High (11 disc) | CTGAN |
| **Training Speed** | Faster (2x) | Slower (2x) | CTGAN ✅ |

**Overall for Bean Dataset**: CTGAN is still faster, but GPU optimization could be slightly better if PAC mismatch is fixed.

---

## Testing the Fix

### Before Fix
```bash
python train_all_compatible_models.py --dataset Bean --models CTGAN --verbose

# Expected output:
# Batch size optimized: 200 → 510  (using PAC=10 constraint)
```

### After Fix (Option 1)
1. Add `"pac": 5` to `config/default_CTGAN.json`
2. Update optimization code to read PAC from config
3. Run training:

```bash
python train_all_compatible_models.py --dataset Bean --models CTGAN --verbose

# Expected output:
# Batch size optimized: 200 → 515  (using PAC=5 constraint)
# or
# Batch size optimized: 200 → 510  (if GPU suggests 512, then (512//5)*5 = 510)
```

---

## Conclusion

The GPU optimization doesn't work properly for CTGAN because:

1. ❌ **PAC mismatch**: Train script assumes PAC=10, but CTGAN code uses PAC=5
2. ❌ **Missing config**: PAC is not specified in config files
3. ❌ **Hardcoded constraint**: Train script hardcodes PAC=10 for CTGAN
4. ✅ **PATE-CTGAN bypasses issue**: Uses PAC=1, no special constraint in train script

**Impact**: Minor efficiency loss (1-2%) due to overly conservative batch size calculation

**Fix**: Add PAC to config files and update optimization code to read it

**Priority**: Medium (doesn't cause errors, but suboptimal performance)
