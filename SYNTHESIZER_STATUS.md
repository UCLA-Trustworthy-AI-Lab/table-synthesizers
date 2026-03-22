# Synthesizer Status Checklist

Status of synthesizers requested in `FULL_SYNTHESIZERS.md`, tested on 2026-03-22.

---

## Summary

| Synthesizer | Status | Tests | Action Taken |
|---|---|---|---|
| TabDiff | Working | 16/16 pass | Fixed checkpoint bug, expanded tests |
| TabPFGen | Working | 12/12 pass | Expanded tests |
| TabPFN Unsupervised | New - Working | 11/11 pass (1 skipped) | Implemented from scratch |
| CLLM | New - Working | 21/21 pass | Implemented from scratch |
| AIM | Stub (runs, low quality) | 3/3 pass | No action - documented |

---

## Detailed Status

### 1. TabDiff - WORKING

- **Location**: `src/stg/TabDiff/tabdiff_synthesizer.py`
- **Implementation**: Complete MLP-DDPM with sinusoidal timestep embeddings, linear beta schedule, forward/reverse diffusion, early stopping
- **Note**: Not architecturally faithful to the TabDiff paper (uses MLP not transformer, no per-column diffusion). Functions as a generic tabular DDPM.
- **Bug fixed**: `_decode_tensor()` crashed after `load_state()` because `stored_data` was None. Fixed by saving `column_order` in checkpoint.
- **Factory fix**: Changed from hard import to try/except pattern
- **Tests**: 16 tests covering init, fit+sample, edit, hyperparams, numeric-only, categorical-only, dtypes, tensors, reproducibility, save/load, single-column, conditions, factory, DataLoader rejection, large samples

### 2. TabPFGen - WORKING

- **Location**: `src/stg/TabPFGen/tabpfgen_synthesizer.py`
- **Implementation**: SGLD energy-based sampler. No traditional model training; uses nearest-neighbor energy function with optional TabPFN refinement for target prediction.
- **Factory fix**: Changed from hard import to try/except pattern
- **Tests**: 12 tests covering init, fit+sample, edit, tensor output, numeric-only, explicit target, regression, reproducibility, custom SGLD params, factory, single-feature, large samples

### 3. TabPFN Unsupervised - NEW, WORKING

- **Location**: `src/stg/TabPFNUnsupervised/tabpfn_unsupervised_synthesizer.py`
- **Implementation**: Wraps `tabpfn_extensions.unsupervised.TabPFNUnsupervisedModel`. Handles categorical columns via label encoding/decoding. Fully unsupervised (no target column required).
- **Dependencies**: `tabpfn>=2.0.0`, `tabpfn-extensions>=0.1.0`
- **Tests**: 12 tests covering availability, init, custom params, fit+sample, numeric-only, dtypes, tensors, single-column, factory, DataLoader rejection, decode roundtrip

### 4. CLLM - NEW, WORKING

- **Location**: `src/stg/CLLM/cllm_synthesizer.py`
- **Implementation**: LLM API-based generation using OpenAI. Stores training data statistics, builds few-shot prompts, parses CSV responses. Handles code fences, malformed output, batching, retries.
- **Dependencies**: `openai>=1.0.0`
- **API Key**: Passed via constructor `api_key` argument. Never serialized in checkpoints.
- **Model**: Default `gpt-5-nano-2025-08-07` (configurable)
- **Tests**: 21 tests (all API calls mocked). Covers init, fit, prompts, parsing (valid CSV, code fences, malformed), mocked generation, batching, tensor/DataFrame output, checkpoint safety, error handling, factory

### 5. AIM - STUB (runs but low quality)

- **Location**: `src/stg/AIM/AIM.py`
- **Implementation**: Just adds Laplace noise to discretized data. `AIM_MBI_AVAILABLE = False` hardcoded. Missing the real AIM algorithm (iterative matrix inference, workload optimization).
- **Tests**: 3/3 pass (init, fit+sample, save/load)
- **Decision**: Per `FULL_SYNTHESIZERS.md`: "Keep AIM dropped. Same reasoning as TabDiff."

---

## Changes Made

### New Files
- [x] `src/stg/TabPFNUnsupervised/__init__.py`
- [x] `src/stg/TabPFNUnsupervised/tabpfn_unsupervised_synthesizer.py`
- [x] `src/stg/CLLM/__init__.py`
- [x] `src/stg/CLLM/cllm_synthesizer.py`
- [x] `tests/unit/test_tabpfn_unsupervised.py`
- [x] `tests/unit/test_cllm.py`
- [x] `config/templates/tabpfn_unsupervised_config.json`
- [x] `config/templates/cllm_config.json`

### Modified Files
- [x] `src/stg/tableSynthesizer.py` - Safe imports for TabDiff/TabPFGen, registered TabPFNUnsupervised + CLLM
- [x] `src/stg/TabDiff/tabdiff_synthesizer.py` - Fixed `_decode_tensor` crash after `load_state`, saved column_order in checkpoint
- [x] `tests/unit/test_tabdiff.py` - Expanded from 3 to 16 tests
- [x] `tests/unit/test_tabpfgen.py` - Expanded from 4 to 12 tests
- [x] `requirements-optional.txt` - Added tabpfn, tabpfn-extensions, openai

### Test Results (2026-03-22)
```
tests/unit/test_tabdiff.py            16 passed
tests/unit/test_tabpfgen.py           12 passed
tests/unit/test_tabpfn_unsupervised.py 11 passed, 1 skipped
tests/unit/test_cllm.py               21 passed
                                      ─────────────────
                                      60 passed, 1 skipped
```
