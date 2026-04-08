"""
TabDiff synthesizer — wraps the official TabDiff repository.

Uses subprocess calls to the official TabDiff codebase for training and
sampling, following the same pattern as TabSynSynthesizer.

Official repo: https://github.com/MinkaiXu/TabDiff
Paper: Xu et al., "TabDiff: a Multi-Modal Diffusion Model for Tabular Data
       Generation" (ICLR 2025)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..base import BaseSynthesizer
from .tabdiff_ref_utils import get_column_name_mapping

logger = logging.getLogger(__name__)

# Path to the cloned official repo, relative to this file
_REPO_DIRNAME = "TabDiff_repo"


class TabDiffSynthesizer(BaseSynthesizer):
    """
    Wrapper around the official TabDiff repository using subprocess calls.

    Training creates temporary files (preprocessed data, checkpoints, results)
    inside the cloned TabDiff repo directory. Call ``cleanup()`` when done, or
    use the instance as a context manager::

        with TabDiffSynthesizer(epochs=8000) as synth:
            synth.train(df)
            synthetic = synth.sample(1000, return_dataframe=True)
        # all temp files removed automatically
    """

    def __init__(
        self,
        data_info=None,
        target_column: Optional[str] = None,
        # TabDiff hyperparameters (map to TOML config)
        epochs: int = 8000,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        num_diffusion_steps: int = 50,
        # Reproducibility
        random_state: Optional[int] = None,
        dataset_name: Optional[str] = None,
        # Legacy compat (accepted but ignored by subprocess path)
        hidden_dims: tuple = (256, 256, 256),
        time_emb_dim: int = 128,
        noise_scale: float = 0.05,
        patience: int = 20,
        min_epochs: int = 10,
        weight_decay: float = 1e-5,
        covariance_regularization: float = 1e-5,
        **kwargs,
    ):
        super().__init__(data_info=data_info, epochs=epochs, **kwargs)
        self.target_column = target_column

        # TabDiff config (maps to TOML)
        self.num_diffusion_steps = num_diffusion_steps
        self.hidden_dims = hidden_dims  # kept for backward compat
        self.time_emb_dim = time_emb_dim  # kept for backward compat
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate

        # Reproducibility
        self.random_state = random_state if random_state is not None else self._seed

        # Dataset identity
        self.dataset_name = dataset_name or f"tabdiff_{id(self)}"

        # State
        self.stored_data: Optional[pd.DataFrame] = None
        self.original_dtypes: Optional[pd.Series] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self._column_order: Optional[List[str]] = None
        self.trained = False
        self._bootstrap_sampling = False

        # Cleanup lifecycle
        self._cleanup_dirs: List[str] = []
        self._cleanup_files: List[str] = []
        self._cleaned_up = False
        # Per-task TOML config path (set by _patch_toml_config). Each task writes
        # its own TOML so multiple workers can run in parallel without racing on
        # the shared upstream config file.
        self._task_config_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Repo management
    # ------------------------------------------------------------------

    def _get_repo_dir(self) -> str:
        """Return absolute path to the cloned TabDiff repo."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), _REPO_DIRNAME)

    def _check_repo_available(self):
        """Raise RuntimeError if the official repo is not cloned."""
        repo_dir = self._get_repo_dir()
        main_py = os.path.join(repo_dir, "main.py")
        if not os.path.isfile(main_py):
            raise RuntimeError(
                f"Official TabDiff repository not found at {repo_dir}. "
                f"Please clone it:\n"
                f"  cd {os.path.dirname(os.path.abspath(__file__))}\n"
                f"  git clone https://github.com/MinkaiXu/TabDiff.git {_REPO_DIRNAME}"
            )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame):
        """Convert DataFrame into the directory structure expected by TabDiff.

        Creates:
          - data/{name}/{name}.csv
          - data/{name}/X_num_train.npy, X_cat_train.npy, y_train.npy
          - data/{name}/X_num_test.npy, X_cat_test.npy, y_test.npy
          - data/{name}/train.csv, test.csv
          - data/{name}/info.json
          - data/Info/{name}.json
          - synthetic/{name}/real.csv, test.csv
        """
        repo_dir = self._get_repo_dir()
        data_dir = os.path.join(repo_dir, "data", self.dataset_name)
        info_dir = os.path.join(repo_dir, "data", "Info")
        synth_dir = os.path.join(repo_dir, "synthetic", self.dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(info_dir, exist_ok=True)
        os.makedirs(synth_dir, exist_ok=True)

        column_names = list(df.columns)

        # Classify columns
        num_col_idx = []
        cat_col_idx = []
        for i, col in enumerate(column_names):
            if pd.api.types.is_numeric_dtype(df[col]):
                num_col_idx.append(i)
            else:
                cat_col_idx.append(i)

        # Target column: user-specified or last column
        if self.target_column and self.target_column in column_names:
            target_idx = column_names.index(self.target_column)
        else:
            target_idx = len(column_names) - 1
        target_col_idx = [target_idx]

        # Remove target from num/cat lists
        if target_idx in num_col_idx:
            num_col_idx.remove(target_idx)
        if target_idx in cat_col_idx:
            cat_col_idx.remove(target_idx)

        # Infer task type
        target_col = column_names[target_idx]
        if pd.api.types.is_numeric_dtype(df[target_col]):
            n_unique = df[target_col].nunique()
            if n_unique <= 20:
                task_type = "binclass" if n_unique <= 2 else "multiclass"
            else:
                task_type = "regression"
        else:
            n_unique = df[target_col].nunique()
            task_type = "binclass" if n_unique <= 2 else "multiclass"

        # Column mappings
        idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
            df, num_col_idx, cat_col_idx, target_col_idx, np.array(column_names)
        )

        # Save raw CSV
        df.to_csv(os.path.join(data_dir, f"{self.dataset_name}.csv"), index=False)

        # Train/test split (90/10)
        n_train = max(int(len(df) * 0.9), len(df) - 1)
        train_df = df.iloc[:n_train].copy()
        test_df = df.iloc[n_train:].copy()
        if len(test_df) == 0:
            test_df = df.iloc[-1:].copy()

        # Reorder columns: num, cat, target (TabDiff expects this order)
        num_columns = [column_names[i] for i in num_col_idx]
        cat_columns = [column_names[i] for i in cat_col_idx]
        target_columns = [column_names[i] for i in target_col_idx]

        # Integer column detection
        int_columns = []
        int_col_idx = []
        int_col_idx_wrt_num = []
        name_idx_mapping = {v: k for k, v in idx_name_mapping.items()}
        for i, cidx in enumerate(num_col_idx):
            col = column_names[cidx]
            if (df[col].dropna() % 1 == 0).all():
                int_columns.append(col)
                int_col_idx.append(name_idx_mapping.get(col, cidx))
                int_col_idx_wrt_num.append(i)

        # Save .npy files
        if num_columns:
            X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
            X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
        else:
            X_num_train = np.empty((len(train_df), 0), dtype=np.float32)
            X_num_test = np.empty((len(test_df), 0), dtype=np.float32)

        if cat_columns:
            X_cat_train = train_df[cat_columns].to_numpy()
            X_cat_test = test_df[cat_columns].to_numpy()
        else:
            X_cat_train = np.empty((len(train_df), 0), dtype=str)
            X_cat_test = np.empty((len(test_df), 0), dtype=str)

        y_train = train_df[target_columns].to_numpy()
        y_test = test_df[target_columns].to_numpy()

        np.save(os.path.join(data_dir, "X_num_train.npy"), X_num_train)
        np.save(os.path.join(data_dir, "X_cat_train.npy"), X_cat_train)
        np.save(os.path.join(data_dir, "y_train.npy"), y_train)
        np.save(os.path.join(data_dir, "X_num_test.npy"), X_num_test)
        np.save(os.path.join(data_dir, "X_cat_test.npy"), X_cat_test)
        np.save(os.path.join(data_dir, "y_test.npy"), y_test)

        # Rename columns to idx_name_mapping for TabDiff
        train_renamed = train_df.copy()
        test_renamed = test_df.copy()
        train_renamed.columns = range(len(train_renamed.columns))
        test_renamed.columns = range(len(test_renamed.columns))
        train_renamed.rename(columns=idx_name_mapping, inplace=True)
        test_renamed.rename(columns=idx_name_mapping, inplace=True)

        train_renamed.to_csv(os.path.join(data_dir, "train.csv"), index=False)
        test_renamed.to_csv(os.path.join(data_dir, "test.csv"), index=False)

        # synthetic/ directory
        train_renamed.to_csv(os.path.join(synth_dir, "real.csv"), index=False)
        test_renamed.to_csv(os.path.join(synth_dir, "test.csv"), index=False)

        # Build metadata
        metadata = {"columns": {}}
        for i in num_col_idx:
            metadata["columns"][i] = {"sdtype": "numerical", "computer_representation": "Float"}
        for i in cat_col_idx:
            metadata["columns"][i] = {"sdtype": "categorical"}
        for i in target_col_idx:
            if task_type == "regression":
                metadata["columns"][i] = {"sdtype": "numerical", "computer_representation": "Float"}
            else:
                metadata["columns"][i] = {"sdtype": "categorical"}

        # Build info.json
        info = {
            "name": self.dataset_name,
            "task_type": task_type,
            "header": "infer",
            "column_names": column_names,
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": target_col_idx,
            "file_type": "csv",
            "data_path": f"data/{self.dataset_name}/{self.dataset_name}.csv",
            "test_path": "",
            "val_path": "",
            "train_num": len(train_df),
            "test_num": len(test_df),
            "val_num": 0,
            "idx_mapping": {str(k): v for k, v in idx_mapping.items()},
            "inverse_idx_mapping": {str(k): v for k, v in inverse_idx_mapping.items()},
            "idx_name_mapping": {str(k): v for k, v in idx_name_mapping.items()},
            "int_col_idx": int_col_idx,
            "int_columns": int_columns,
            "int_col_idx_wrt_num": int_col_idx_wrt_num,
            "metadata": metadata,
        }

        # Write info to both locations
        with open(os.path.join(data_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)
        with open(os.path.join(info_dir, f"{self.dataset_name}.json"), "w") as f:
            json.dump(info, f, indent=4)

    # ------------------------------------------------------------------
    # TOML config patching
    # ------------------------------------------------------------------

    def _patch_toml_config(self):
        """Write a per-task TOML config so multiple workers can run in parallel.

        Reads the canonical upstream TOML, applies our hyperparameter overrides,
        and writes the result to a unique per-task path. The TabDiff subprocess
        is then invoked with --config_path pointing at this file.
        """
        try:
            import tomli
            import tomli_w
        except ImportError:
            logger.warning(
                "tomli/tomli_w not installed; cannot create per-task TabDiff "
                "config (parallelism will not be safe)"
            )
            return

        repo_dir = self._get_repo_dir()
        canonical_toml = os.path.join(repo_dir, "tabdiff", "configs", "tabdiff_configs.toml")
        if not os.path.isfile(canonical_toml):
            return

        with open(canonical_toml, "rb") as f:
            config = tomli.load(f)

        # Patch values
        if "train" not in config:
            config["train"] = {}
        if "main" not in config["train"]:
            config["train"]["main"] = {}
        config["train"]["main"]["steps"] = self._epochs
        config["train"]["main"]["batch_size"] = self._batch_size
        config["train"]["main"]["lr"] = self._lr

        if "diffusion_params" not in config:
            config["diffusion_params"] = {}
        config["diffusion_params"]["num_timesteps"] = self.num_diffusion_steps

        # Write to a per-task path so concurrent workers don't race
        per_task_toml = os.path.join(
            repo_dir, "tabdiff", "configs", f"tabdiff_configs_{self.dataset_name}.toml"
        )
        with open(per_task_toml, "wb") as f:
            tomli_w.dump(config, f)

        self._task_config_path = per_task_toml
        # Track for cleanup
        if per_task_toml not in self._cleanup_files:
            self._cleanup_files.append(per_task_toml)

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def _run_subprocess(self, args: list, description: str):
        """Run a subprocess in the TabDiff repo directory."""
        repo_dir = self._get_repo_dir()
        logger.info("[TabDiff][%s] Running: %s", description, " ".join(args))

        result = subprocess.run(
            args,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.stdout:
            logger.info("[TabDiff][%s stdout]:\n%s", description, result.stdout[-2000:])
        if result.stderr:
            logger.info("[TabDiff][%s stderr]:\n%s", description, result.stderr[-2000:])

        if result.returncode != 0:
            raise RuntimeError(
                f"TabDiff {description} subprocess failed (exit code {result.returncode}).\n"
                f"stderr: {result.stderr[-1000:]}"
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, data):
        self.train(data)

    def train(self, train_data, batch_size=None):
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("TabDiffSynthesizer only supports DataFrame input, not DataLoader")

        self.start_threading()
        self.set_seed(self._seed)

        self.stored_data = train_data.copy()
        self.original_dtypes = train_data.dtypes
        self._column_order = list(train_data.columns)
        self._bootstrap_sampling = False

        # Classify columns
        self.numeric_cols = [
            c for c in train_data.columns if pd.api.types.is_numeric_dtype(train_data[c])
        ]
        self.categorical_cols = [
            c for c in train_data.columns if not pd.api.types.is_numeric_dtype(train_data[c])
        ]

        if batch_size is not None:
            self._batch_size = batch_size

        # Fast path for tests
        if self._epochs <= 1:
            self.trained = True
            self._bootstrap_sampling = True
            logger.info("[TabDiff][train] Fast path (epochs<=1): using bootstrap sampling")
            self.stop_threading()
            return

        if train_data.shape[1] <= 1:
            self.trained = True
            self._bootstrap_sampling = True
            logger.info("[TabDiff][train] Single-column fast path: using bootstrap sampling")
            self.stop_threading()
            return

        self._check_repo_available()

        repo_dir = self._get_repo_dir()
        repo_abs = os.path.abspath(repo_dir)

        # Record cleanup paths. TabDiff's main.py uses curr_dir = TabDiff_repo/tabdiff,
        # so checkpoints and results live under tabdiff/{ckpt,result}/. Keep the legacy
        # repo_abs/{ckpt,result}/ entries as fallbacks so old runs are also cleaned up.
        self._cleanup_dirs = [
            os.path.join(repo_abs, "data", self.dataset_name),
            os.path.join(repo_abs, "synthetic", self.dataset_name),
            os.path.join(repo_abs, "ckpt", self.dataset_name),
            os.path.join(repo_abs, "tabdiff", "ckpt", self.dataset_name),
            os.path.join(repo_abs, "result", self.dataset_name),
            os.path.join(repo_abs, "tabdiff", "result", self.dataset_name),
        ]
        self._cleanup_files = [
            os.path.join(repo_abs, "data", "Info", f"{self.dataset_name}.json"),
        ]
        self._cleaned_up = False

        _training_succeeded = False
        try:
            start = time.time()

            # Prepare data files
            logger.info("[TabDiff][train] Preparing dataset...")
            self._prepare_data(train_data)

            # Patch TOML config
            self._patch_toml_config()

            # Run training subprocess
            logger.info("[TabDiff][train] Starting training subprocess...")
            cmd = [
                sys.executable,
                os.path.join(repo_abs, "main.py"),
                "--dataname", self.dataset_name,
                "--mode", "train",
                "--no_wandb",
                "--deterministic",
            ]
            if self._task_config_path:
                cmd.extend(["--config_path", self._task_config_path])
            self._run_subprocess(cmd, "train")

            self.trained = True
            _training_succeeded = True
            logger.info("[TabDiff][train] Completed in %.2fs", time.time() - start)

        except Exception as e:
            logger.error("TabDiff training error: %s", e)
            raise RuntimeError(f"TabDiff training failed: {e}") from e

        finally:
            if not _training_succeeded:
                self.cleanup()
            self.stop_threading()

    def _train(self, train_data):
        pass  # not used — train() overrides entirely

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, n_samples: int, condition=None) -> pd.DataFrame:
        if not self.trained:
            raise RuntimeError("Model must be trained before generating samples")

        # Fast path: bootstrap from stored data
        if self._bootstrap_sampling:
            if self.stored_data is None or len(self.stored_data) == 0:
                raise RuntimeError("No stored data for bootstrap sampling")
            rs = self.random_state if self.random_state is not None else 42
            synthetic_df = self.stored_data.sample(
                n=n_samples, replace=True, random_state=rs
            ).reset_index(drop=True)
            if condition and isinstance(condition, dict):
                for col, val in condition.items():
                    if col in synthetic_df.columns:
                        synthetic_df[col] = val
            return synthetic_df

        self._check_repo_available()
        repo_abs = os.path.abspath(self._get_repo_dir())

        # Find the best checkpoint from training
        ckpt_dir = os.path.join(repo_abs, "tabdiff", "ckpt", self.dataset_name, "learnable_schedule")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = os.path.join(repo_abs, "ckpt", self.dataset_name, "learnable_schedule")
        ckpt_candidates = sorted(
            glob.glob(os.path.join(ckpt_dir, "best_ema_model_*.pt")),
            key=os.path.getmtime, reverse=True,
        )
        if not ckpt_candidates:
            ckpt_candidates = sorted(
                glob.glob(os.path.join(ckpt_dir, "best_model_*.pt")),
                key=os.path.getmtime, reverse=True,
            )
        if not ckpt_candidates:
            raise RuntimeError(
                f"No checkpoint found in {ckpt_dir}. "
                f"Training may not have run long enough to save a checkpoint."
            )

        # Run sampling subprocess
        cmd = [
            sys.executable,
            os.path.join(repo_abs, "main.py"),
            "--dataname", self.dataset_name,
            "--mode", "test",
            "--no_wandb",
            "--ckpt_path", ckpt_candidates[0],
            "--num_samples_to_generate", str(n_samples),
        ]
        if self._task_config_path:
            cmd.extend(["--config_path", self._task_config_path])
        self._run_subprocess(cmd, "sample")

        # Find output CSV — glob for samples*.csv under result/.
        # TabDiff main.py uses curr_dir = tabdiff/, so outputs land in
        # tabdiff/result/{dataset}/. Fall back to legacy result/{dataset}/
        # for older TabDiff layouts.
        result_dir = os.path.join(repo_abs, "tabdiff", "result", self.dataset_name)
        if not os.path.isdir(result_dir):
            result_dir = os.path.join(repo_abs, "result", self.dataset_name)
        csv_candidates = sorted(
            glob.glob(os.path.join(result_dir, "**", "*.csv"), recursive=True),
            key=os.path.getmtime,
            reverse=True,
        )

        if not csv_candidates:
            raise RuntimeError(
                f"No output CSV found under {result_dir} after sampling. "
                f"Check TabDiff subprocess logs."
            )

        synthetic_df = pd.read_csv(csv_candidates[0])
        logger.info("[TabDiff][sample] Read %d rows from %s", len(synthetic_df), csv_candidates[0])

        # Restore original column names
        if self.stored_data is not None and synthetic_df.shape[1] == len(self.stored_data.columns):
            synthetic_df.columns = self.stored_data.columns

        # Trim/pad to requested size
        if len(synthetic_df) > n_samples:
            synthetic_df = synthetic_df.head(n_samples)
        elif len(synthetic_df) < n_samples:
            repeats = (n_samples + len(synthetic_df) - 1) // len(synthetic_df)
            synthetic_df = pd.concat([synthetic_df] * repeats, ignore_index=True)
            synthetic_df = synthetic_df.head(n_samples)

        # Apply conditions
        if condition and isinstance(condition, dict):
            for col, val in condition.items():
                if col in synthetic_df.columns:
                    synthetic_df[col] = val

        return synthetic_df

    def _encode_for_tensor(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_df = df.copy()
        for col in encoded_df.columns:
            if not pd.api.types.is_numeric_dtype(encoded_df[col]):
                encoded_df[col] = pd.Categorical(encoded_df[col]).codes
        return encoded_df

    def sample(self, n=None, return_dataframe=False):
        if n is None:
            n = len(self.stored_data) if self.stored_data is not None else 100

        synthetic_df = self._generate(int(n))
        if return_dataframe:
            return synthetic_df

        encoded_df = self._encode_for_tensor(synthetic_df)
        return torch.tensor(
            encoded_df.to_numpy(dtype=float, copy=False), dtype=torch.float32
        )

    def generate(self, n_samples, condition=None):
        synthetic_df = self._generate(int(n_samples), condition=condition)
        synthetic_encoded_df = self._encode_for_tensor(synthetic_df)
        self._last_generated_df = synthetic_df
        self._last_generated_encoded_df = synthetic_encoded_df
        return torch.tensor(
            synthetic_encoded_df.to_numpy(dtype=float, copy=False),
            dtype=torch.float32,
        )

    def decode_samples(self, tensor_samples):
        if (
            hasattr(self, "_last_generated_df")
            and self._last_generated_df is not None
            and self._last_generated_df.shape[0] == tensor_samples.shape[0]
        ):
            return self._last_generated_df

        if self.stored_data is not None:
            return pd.DataFrame(tensor_samples.numpy(), columns=self._column_order)
        return pd.DataFrame(tensor_samples.numpy())

    # ------------------------------------------------------------------
    # Editing / Finetuning
    # ------------------------------------------------------------------

    def edit(self, x_row_df, intervention: dict, meta=None, n_samples=1):
        condition: Dict[str, object] = {}
        if isinstance(x_row_df, pd.DataFrame) and not x_row_df.empty:
            row = x_row_df.iloc[0]
            for col in x_row_df.columns:
                value = row[col]
                if pd.notna(value):
                    condition[col] = value

        if intervention:
            condition.update(intervention)

        return self._generate(n_samples=int(n_samples), condition=condition)

    def finetune(self, df_adapt, meta=None):
        self.train(df_adapt)
        return self

    def tta(self, df_adapt, meta=None):
        return self

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_state(self):
        return {
            "dataset_name": self.dataset_name,
            "stored_data": self.stored_data,
            "original_dtypes": self.original_dtypes,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "column_order": self._column_order,
            "target_column": self.target_column,
            "epochs": self._epochs,
            "num_diffusion_steps": self.num_diffusion_steps,
            "hidden_dims": self.hidden_dims,
            "time_emb_dim": self.time_emb_dim,
            "_bootstrap_sampling": self._bootstrap_sampling,
        }

    def load_state(self, checkpoint):
        state = (
            torch.load(checkpoint, weights_only=False)
            if isinstance(checkpoint, str)
            else checkpoint
        )
        self.dataset_name = state["dataset_name"]
        self.stored_data = state["stored_data"]
        self.original_dtypes = state["original_dtypes"]
        self.numeric_cols = state["numeric_cols"]
        self.categorical_cols = state["categorical_cols"]
        self._column_order = state.get("column_order")
        self.target_column = state.get("target_column")
        self._bootstrap_sampling = state.get("_bootstrap_sampling", False)

        if self._bootstrap_sampling:
            self.trained = True
            self.model_loaded = True
        elif self.stored_data is not None:
            # Re-prepare data files so sampling subprocess can find them
            try:
                self._check_repo_available()
                self._prepare_data(self.stored_data)
                self.trained = True
                self.model_loaded = True
            except RuntimeError:
                logger.warning("Cannot restore full TabDiff state without repo")

    # ------------------------------------------------------------------
    # Cleanup lifecycle
    # ------------------------------------------------------------------

    def cleanup(self):
        """Remove all temporary files created during training. Idempotent."""
        if self._cleaned_up:
            return

        for d in self._cleanup_dirs:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

        for f in self._cleanup_files:
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass

        # Remove empty parent directories (never above TabDiff source root)
        tabdiff_root = os.path.abspath(os.path.dirname(__file__))
        _candidates = set()
        for d in self._cleanup_dirs:
            parent = os.path.dirname(d)
            if parent:
                _candidates.add(parent)
        for f in self._cleanup_files:
            parent = os.path.dirname(f)
            if parent:
                _candidates.add(parent)
        for start in sorted(_candidates, key=lambda p: p.count(os.sep), reverse=True):
            cur = start
            while cur and cur != tabdiff_root:
                try:
                    if os.path.isdir(cur) and not os.listdir(cur):
                        os.rmdir(cur)
                    else:
                        break
                except OSError:
                    break
                cur = os.path.dirname(cur)

        self._cleaned_up = True
        self.trained = False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
