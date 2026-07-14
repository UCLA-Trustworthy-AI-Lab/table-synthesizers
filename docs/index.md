# table-synthesizers

Unified synthetic tabular data generation with 16 models and a single DataFrame interface.

## Quick Start

```python
import pandas as pd
from stg import TableSynthesizer

df = pd.read_csv("data.csv")

ts = TableSynthesizer("CTGAN", config={"epochs": 300, "seed": 42})
ts.fit(df)
synthetic = ts.sample(1000, return_dataframe=True)
```

## Installation

```bash
# Base (CPU-only models)
pip install -r requirements.txt

# GPU models (CUDA 13.x for Blackwell / SM 12.1+)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-gpu.txt

# Synthcity models (ARF, BayesianNetwork, GREAT, NFlow)
pip install -r requirements-synthcity.txt
```

## Model Groups

| Group | Models |
|-------|--------|
| **Base** | Identity, CART, DPCART, SMOTE, AIM |
| **GAN** | CTGAN, PATECTGAN |
| **VAE / Diffusion** | TVAE, TabDDPM, TabSyn, AutoDiff, LTM_VAE |
| **Transformer** | TabDiff, TabPFGen, GREAT |
| **Probabilistic** | GaussianCopula, BayesianNetwork, ARF, NFlow |
