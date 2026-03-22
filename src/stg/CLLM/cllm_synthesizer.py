from __future__ import annotations

import io
import json
import logging
import math
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..base import BaseSynthesizer

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class CLLMSynthesizer(BaseSynthesizer):
    """
    LLM-based tabular data synthesizer using the OpenAI API.

    Uses few-shot prompting to generate synthetic tabular rows. The LLM is
    given column metadata and example rows from the training data, then asked
    to produce new rows in CSV format.

    Based on the CLLM framework (https://github.com/seedatnabeel/CLLM).
    """

    def __init__(
        self,
        data_info=None,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano-2025-08-07",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        n_examples: int = 5,
        batch_size_per_call: int = 10,
        max_retries: int = 3,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "CLLMSynthesizer requires the openai package. "
                "Install with: pip install openai"
            )

        super().__init__(data_info=data_info, **kwargs)
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_examples = n_examples
        self.batch_size_per_call = batch_size_per_call
        self.max_retries = max_retries
        self.random_state = random_state

        self.stored_data: Optional[pd.DataFrame] = None
        self._column_names: List[str] = []
        self._dtypes: Dict[str, str] = {}
        self._column_stats: Dict[str, dict] = {}
        self._example_rows: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Training (stores data + metadata, no model fitting)
    # ------------------------------------------------------------------
    def fit(self, data):
        self.train(data)

    def train(self, train_data, batch_size=32):
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError(
                "CLLMSynthesizer only supports DataFrame input, not DataLoader"
            )

        self.start_threading()

        self.stored_data = train_data.copy()
        self._column_names = list(train_data.columns)
        self._dtypes = {col: str(train_data[col].dtype) for col in train_data.columns}

        # Compute summary statistics per column
        self._column_stats = {}
        for col in train_data.columns:
            stats: dict = {}
            if pd.api.types.is_numeric_dtype(train_data[col]):
                stats["type"] = "numerical"
                stats["min"] = float(train_data[col].min())
                stats["max"] = float(train_data[col].max())
                stats["mean"] = float(train_data[col].mean())
                stats["std"] = float(train_data[col].std())
            else:
                stats["type"] = "categorical"
                vc = train_data[col].value_counts()
                stats["unique_values"] = vc.index.tolist()[:20]  # cap at 20
                stats["n_unique"] = int(train_data[col].nunique())
            self._column_stats[col] = stats

        # Select few-shot example rows
        rng = np.random.RandomState(self.random_state)
        n_ex = min(self.n_examples, len(train_data))
        idx = rng.choice(len(train_data), size=n_ex, replace=False)
        self._example_rows = train_data.iloc[idx].reset_index(drop=True)

        self.stop_threading()

    def _train(self, train_data):
        pass

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        return (
            "You are a synthetic tabular data generator. Your task is to generate "
            "realistic rows of tabular data that mirror the statistical properties, "
            "correlations, and patterns of the training data provided. Output ONLY "
            "valid CSV rows (no header, no explanation, no markdown fences). Each "
            "row must have exactly the same number of fields as columns specified."
        )

    def _build_user_prompt(self, n_rows: int) -> str:
        lines = []
        lines.append(f"Generate exactly {n_rows} new CSV rows for a table with these columns:\n")

        # Column schema
        for col in self._column_names:
            stats = self._column_stats[col]
            if stats["type"] == "numerical":
                lines.append(
                    f"- {col} (numerical): min={stats['min']:.4g}, "
                    f"max={stats['max']:.4g}, mean={stats['mean']:.4g}, "
                    f"std={stats['std']:.4g}"
                )
            else:
                vals = stats["unique_values"]
                lines.append(
                    f"- {col} (categorical): possible values = {vals}"
                )

        # Few-shot examples
        lines.append(f"\nHere are {len(self._example_rows)} example rows (CSV with header):")
        csv_buf = io.StringIO()
        self._example_rows.to_csv(csv_buf, index=False)
        lines.append(csv_buf.getvalue().strip())

        lines.append(
            f"\nGenerate exactly {n_rows} new rows in CSV format "
            f"(no header, no numbering, no extra text). "
            f"Each row must have exactly {len(self._column_names)} comma-separated fields."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        if self.api_key is None:
            raise ValueError(
                "API key is required for generation. Pass api_key in constructor "
                "or set it before calling generate/sample."
            )

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, response_text: str, n_expected: int) -> pd.DataFrame:
        text = response_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try to detect and skip a header row
        first_line = text.split("\n")[0] if text else ""
        has_header = all(
            col_name in first_line for col_name in self._column_names[:3]
        ) if len(self._column_names) >= 3 else False

        try:
            df = pd.read_csv(
                io.StringIO(text),
                header=0 if has_header else None,
                names=None if has_header else self._column_names,
            )
            if has_header:
                # Ensure column order matches
                df.columns = self._column_names[:len(df.columns)]
        except Exception:
            # Fallback: try line-by-line parsing
            rows = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == len(self._column_names):
                    rows.append(parts)
            if not rows:
                logger.warning("Failed to parse any rows from LLM response")
                return pd.DataFrame(columns=self._column_names)
            df = pd.DataFrame(rows, columns=self._column_names)

        # Ensure correct column count
        if len(df.columns) > len(self._column_names):
            df = df.iloc[:, : len(self._column_names)]
        if len(df.columns) < len(self._column_names):
            for col in self._column_names[len(df.columns):]:
                df[col] = np.nan
        df.columns = self._column_names

        # Cast dtypes
        for col in self._column_names:
            dtype_str = self._dtypes.get(col, "object")
            try:
                if "int" in dtype_str:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                elif "float" in dtype_str:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(str)
            except (ValueError, TypeError):
                pass

        return df.head(n_expected)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(self, n_samples: int, condition=None) -> pd.DataFrame:
        if self.stored_data is None:
            raise RuntimeError("Model must be trained before generating samples")

        system_prompt = self._build_system_prompt()
        all_dfs = []
        remaining = n_samples

        while remaining > 0:
            batch_n = min(remaining, self.batch_size_per_call)
            user_prompt = self._build_user_prompt(batch_n)

            parsed_df = pd.DataFrame(columns=self._column_names)
            for attempt in range(self.max_retries):
                try:
                    response_text = self._call_api(system_prompt, user_prompt)
                    parsed_df = self._parse_response(response_text, batch_n)
                    if len(parsed_df) > 0:
                        break
                except Exception as e:
                    logger.warning(
                        "API call attempt %d/%d failed: %s",
                        attempt + 1,
                        self.max_retries,
                        str(e),
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)

            if len(parsed_df) > 0:
                all_dfs.append(parsed_df)
                remaining -= len(parsed_df)
            else:
                logger.warning("Failed to generate batch after %d retries", self.max_retries)
                break

        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True).head(n_samples)
        else:
            result = pd.DataFrame(columns=self._column_names)

        # Apply condition if provided
        if condition and isinstance(condition, dict):
            for col, val in condition.items():
                if col in result.columns:
                    result[col] = val

        return result

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
            return pd.DataFrame(tensor_samples.numpy(), columns=self._column_names)
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
    # Checkpointing (never serialize api_key)
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            "stored_data": self.stored_data,
            "column_names": self._column_names,
            "dtypes": self._dtypes,
            "column_stats": self._column_stats,
            "example_rows": self._example_rows,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n_examples": self.n_examples,
            "batch_size_per_call": self.batch_size_per_call,
            "max_retries": self.max_retries,
        }

    def load_state(self, checkpoint):
        self.stored_data = checkpoint["stored_data"]
        self._column_names = checkpoint["column_names"]
        self._dtypes = checkpoint["dtypes"]
        self._column_stats = checkpoint["column_stats"]
        self._example_rows = checkpoint["example_rows"]
        self.model_name = checkpoint.get("model_name", self.model_name)
        self.temperature = checkpoint.get("temperature", self.temperature)
        self.max_tokens = checkpoint.get("max_tokens", self.max_tokens)
        self.n_examples = checkpoint.get("n_examples", self.n_examples)
        self.batch_size_per_call = checkpoint.get("batch_size_per_call", self.batch_size_per_call)
        self.max_retries = checkpoint.get("max_retries", self.max_retries)
