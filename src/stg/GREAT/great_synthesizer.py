import numpy as np
import pandas as pd
import torch
from typing import Optional

from ..base import BaseSynthesizer

try:
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    # Verify be_great and its datasets dependency are importable.
    # be_great requires datasets>=3.1.0 for pyarrow 14+ compatibility.
    # If datasets is too old (uses pa.PyExtensionType removed in pyarrow 14),
    # be_great import will fail with AttributeError at module level.
    import be_great as _be_great_check  # noqa: F401
    SYNTHCITY_AVAILABLE = True
except (ImportError, AttributeError):
    SYNTHCITY_AVAILABLE = False


def _patch_great_trainer():
    """Patch be_great's GReaT.fit to work with transformers 5.x.

    transformers 5.0 renamed the Trainer `tokenizer` parameter to
    `processing_class`. be_great 0.0.9 still passes `tokenizer=`, so we
    monkey-patch GReaT.fit to use the new parameter name.
    """
    try:
        import be_great.great as _great_module
        from transformers import Trainer
        import inspect

        # Check if Trainer still accepts `tokenizer`
        sig = inspect.signature(Trainer.__init__)
        if 'tokenizer' in sig.parameters:
            return  # No patch needed

        _original_fit = _great_module.GReaT.fit

        def _patched_fit(self, data, column_names=None, conditional_col=None,
                         resume_from_checkpoint=False):
            import logging
            from be_great.great import (
                GReaTDataCollator, GReaTDataset, GReaTTrainer,
                TrainingArguments, _array_to_dataframe,
            )

            df = _array_to_dataframe(data, columns=column_names)
            self._update_column_information(df)
            self._update_conditional_information(df, conditional_col)

            logging.info("Convert data into HuggingFace dataset object...")
            great_ds = GReaTDataset.from_pandas(df)
            great_ds.set_tokenizer(self.tokenizer)

            logging.info("Create GReaT Trainer...")
            training_args = TrainingArguments(
                self.experiment_dir,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                **self.train_hyperparameters,
            )
            great_trainer = GReaTTrainer(
                self.model,
                training_args,
                train_dataset=great_ds,
                processing_class=self.tokenizer,
                data_collator=GReaTDataCollator(self.tokenizer),
            )

            logging.info("Start training...")
            great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            return great_trainer

        _great_module.GReaT.fit = _patched_fit
    except Exception:
        pass  # If patching fails, let the original error surface


class GREATSynthesizer(BaseSynthesizer):
    """
    GREAT (GeneRative fEAture Transformer) synthesizer for tabular data generation.

    This synthesizer uses synthcity's GREAT implementation which is a transformer-based
    generative model for mixed-type tabular data.
    Only supports DataFrame input (not DataLoader).

    Synthcity plugin parameters (passed via config dict):
        n_iter (int): Number of training iterations/epochs. Default: 100.
        llm (str): Language model to use. Default: "distilgpt2".
        batch_size (int): Training batch size. Default: 8.
        experiment_dir (str): Directory for trainer output. Default: "trainer_great".
        device (str): Device for training ("cpu" or "cuda"). Default: "cpu".
        random_state (int): Random seed. Default: 0.
        sampling_patience (int): Max retries for schema-valid sampling. Default: 500.
        logging_epoch (int): Log every N epochs. Default: 100.
    """

    # Parameters that synthcity's GREAT plugin accepts
    _SYNTHCITY_PARAMS = {
        'n_iter', 'llm', 'batch_size', 'experiment_dir',
        'device', 'random_state', 'sampling_patience', 'logging_epoch',
    }

    def __init__(self, data_info=None, **kwargs):
        if not SYNTHCITY_AVAILABLE:
            raise ImportError("synthcity package is required for GREATSynthesizer. "
                            "Install it with: pip install synthcity")

        # Extract synthcity-specific params before passing to base class
        self._synthcity_kwargs = {}
        for key in list(kwargs.keys()):
            if key in self._SYNTHCITY_PARAMS:
                self._synthcity_kwargs[key] = kwargs.pop(key)

        super().__init__(data_info=data_info, **kwargs)
        self.model = None
        self.stored_data = None
        _patch_great_trainer()

    def fit(self, data):
        """Sklearn-style fit method."""
        self.train(data)

    def train(self, train_data, batch_size=32):
        """Override base train method to handle DataFrame input directly."""
        import logging
        logger = logging.getLogger(__name__)

        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("GREATSynthesizer only supports DataFrame input, not DataLoader")

        # Skip base class conversion and handle DataFrame directly
        self.start_threading()

        self.stored_data = train_data.copy()

        # Build synthcity plugin kwargs
        plugin_kwargs = dict(self._synthcity_kwargs)

        # Map 'epochs' to 'n_iter' for consistency with other synthesizers
        if hasattr(self, '_epochs') and self._epochs is not None and 'n_iter' not in plugin_kwargs:
            plugin_kwargs['n_iter'] = self._epochs

        # Auto-detect device if not explicitly set by user (CUDA → CPU fallback)
        # Synthcity GREAT accepts "cuda" or "cpu" only (not "mps")
        if 'device' not in plugin_kwargs:
            from ..gpu_utils import detect_best_device
            detected = detect_best_device()
            plugin_kwargs['device'] = "cuda" if detected.type == "cuda" else "cpu"
            logger.info("GREAT: auto-detected device: %s", plugin_kwargs['device'])

        if plugin_kwargs:
            logger.info("GREAT: using plugin params: %s", plugin_kwargs)

        # Create synthcity loader and train model
        loader = GenericDataLoader(train_data)
        self.model = Plugins().get("great", **plugin_kwargs)
        self.model.fit(loader)

        logger.info("GREAT: trained on %d samples", len(self.stored_data))

        self.stop_threading()

    def _train(self, train_data):
        """Not used - we override train() directly."""
        pass

    def _generate(self, n_samples):
        """Generate synthetic samples using GREAT."""
        if self.model is None:
            raise RuntimeError("Model must be trained before generating samples")

        # Generate samples using synthcity
        synthetic_loader = self.model.generate(count=n_samples)
        synthetic_df = synthetic_loader.dataframe()

        return synthetic_df

    def sample(self, n=None, return_dataframe=False):
        """Generate synthetic samples."""
        if n is None:
            n = len(self.stored_data) if self.stored_data is not None else 100

        synthetic_df = self._generate(n)

        if return_dataframe:
            return synthetic_df
        else:
            # Convert to tensor format for compatibility
            # First encode categorical columns if any
            encoded_df = self._encode_for_tensor(synthetic_df)
            return torch.tensor(encoded_df.values, dtype=torch.float32)

    def _encode_for_tensor(self, df):
        """Encode DataFrame for tensor conversion.

        Normalises every column to float64 so that df.values returns a
        homogeneous numpy array that torch can consume.  In pandas 2.0+ a
        DataFrame with mixed numeric dtypes (bool + int64 + float64) can
        return an object-dtype array from .values, which torch rejects.

        Rules applied per column:
          - pd.Categorical → cast to object first (avoids code-vs-label mismatch)
          - non-numeric (object/string/bool-object) → label-encode → float64
          - numeric (bool, int*, float*) → cast to float64 directly
        """
        encoded_df = df.copy()

        for col in df.columns:
            series = encoded_df[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                encoded_df[col] = series.astype(object)
                series = encoded_df[col]
            if not pd.api.types.is_numeric_dtype(series):
                encoded_df[col] = pd.Categorical(series).codes.astype(np.float64)
            else:
                encoded_df[col] = series.astype(np.float64)

        return encoded_df

    def generate(self, n_samples, condition=None):
        """Generate synthetic samples - called by TableSynthesizer.sample()."""
        # Generate decoded synthetic data (with original types)
        synthetic_decoded_df = self._generate(n_samples)

        # Create encoded version for tensor compatibility
        synthetic_encoded_df = self._encode_for_tensor(synthetic_decoded_df)

        # Store both versions
        self._last_generated_encoded_df = synthetic_encoded_df
        self._last_generated_df = synthetic_decoded_df

        # Convert encoded version to tensor for TableSynthesizer compatibility
        return torch.tensor(synthetic_encoded_df.values, dtype=torch.float32)

    def decode_samples(self, tensor_samples):
        """Convert tensor samples back to DataFrame - used for return_dataframe=True."""
        # Return the stored DataFrame if available and matches size
        if hasattr(self, '_last_generated_df') and self._last_generated_df.shape[0] == tensor_samples.shape[0]:
            return self._last_generated_df
        else:
            # Fallback: reconstruct DataFrame from tensor (loses original dtypes)
            if self.stored_data is not None:
                columns = self.stored_data.columns
                return pd.DataFrame(tensor_samples.numpy(), columns=columns)
            else:
                return pd.DataFrame(tensor_samples.numpy())
