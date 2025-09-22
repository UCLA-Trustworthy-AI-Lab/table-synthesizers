import os
import sys
import pandas as pd
import torch


def main() -> None:
    # Hardcoded absolute path to CSV. Replace with your dataset path.
    # Tip: keep a small CSV for quick tests; large files will train slower.
    CSV_PATH = "/work/nvme/bdqh/tfm/table_generation/table-synthesizers/example/sandbox/adult.csv"

    # Ensure the repository's `src/` is on PYTHONPATH so `from stg import TableSynthesizer` works
    REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if REPO_SRC not in sys.path:
        sys.path.insert(0, REPO_SRC)

    from stg import TableSynthesizer

    # Select one synthesizer to run. Recommended stable choices for DataFrame input: 'TVAE', 'TabDDPM'.
    # Known caveat: 'CTGAN' and 'PATECTGAN' may have DataFrame issues per README; prefer legacy DataLoader for them.
    MODEL_NAME = "AutoDiff"  # e.g., 'TVAE', 'CTGAN', 'AutoDiff', 'Identity'

    # Minimal model config. Adjust per model; values here are safe defaults for a sanity check.
    # To adapt: add/remove keys according to each model's parameters (see README/tests).
    MODEL_CONFIG = {
        "epochs": 5,        # keep tiny for a quick run; increase for quality
        "diff_n_epochs": 5,
        "batch_size": 64,   # fit() can also override this via its own batch_size arg
    }

    # If GPU is available, nudge models that accept a 'cuda' flag
    cuda_available = torch.cuda.is_available()
    if MODEL_NAME in ("TVAE", "PATECTGAN"):
        MODEL_CONFIG["cuda"] = bool(cuda_available)

    # How many rows to generate; set return_dataframe=True to get a decoded pandas.DataFrame
    N_SAMPLES = 10

    # 1) Load data
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV with shape: {df.shape}")

    # 2) Create synthesizer
    synthesizer = TableSynthesizer(MODEL_NAME, MODEL_CONFIG)
    print(f"Initialized synthesizer: {MODEL_NAME}")
    # Print CUDA/GPU status and intended device
    print(f"CUDA available: {cuda_available}")
    try:
        print(f"torch.__version__={torch.__version__}, torch.version.cuda={getattr(torch.version,'cuda',None)}")
        print(f"CUDA device_count={torch.cuda.device_count()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU[{i}]: {torch.cuda.get_device_name(i)}")
    except Exception:
        pass
    try:
        init_dev = getattr(synthesizer.model, "_device", getattr(synthesizer.model, "device", None))
        if init_dev is not None:
            print(f"Selected device (pre-train): {init_dev}")
    except Exception:
        pass

    # 3) Fit model on DataFrame (automatic encoding handled internally)
    synthesizer.fit(df)
    # Confirm runtime device
    try:
        run_dev = getattr(synthesizer.model, "_device", getattr(synthesizer.model, "device", None))
        print(f"Training complete on device: {run_dev}")
    except Exception:
        print("Training complete")

    # 4) Sample synthetic data
    synth_df = synthesizer.sample(n=N_SAMPLES, return_dataframe=True)
    print(f"\nSynthetic sample (n={N_SAMPLES}) head:\n{synth_df.head()}")

    # Optional: if you want a tensor instead, omit return_dataframe or set to False
    # synth_tensor = synthesizer.sample(n=N_SAMPLES)
    # print(f"Tensor shape: {synth_tensor.shape}")

    # Notes to adapt:
    # - Use a different CSV: change CSV_PATH to your file.
    # - Use a different model: set MODEL_NAME to any available in stg.tableSynthesizer.DEFAULT_MODELS.
    #   Examples: 'TVAE', 'TabDDPM', 'Identity', 'CART', 'DPCART', 'SMOTE' (if deps installed), etc.
    # - Tune training: modify MODEL_CONFIG keys per model (see README and test files for references).


if __name__ == "__main__":
    main()
