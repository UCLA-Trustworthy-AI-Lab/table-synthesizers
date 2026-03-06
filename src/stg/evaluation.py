from pathlib import Path
from typing import Any, Dict, Mapping, Union

import pandas as pd

DEFAULT_FIDELITY_METRICS = ["SumStats", "ColumnShape", "PairwiseSimilarity"]
DEFAULT_UTILITY_METRICS = ["TabularUtility"]
DEFAULT_PRIVACY_METRICS = ["DCR"]


def generate_column_name_to_datatype(df: pd.DataFrame) -> Dict[str, str]:
    """Generate evaluator metadata mapping from DataFrame dtypes."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    mapping: Dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            mapping[col] = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            mapping[col] = "binary"
        elif pd.api.types.is_numeric_dtype(series):
            mapping[col] = "numerical"
        else:
            mapping[col] = "categorical"
    return mapping


def _normalize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)

    if "target_column" not in normalized:
        if "target" in normalized:
            normalized["target_column"] = normalized["target"]
        else:
            raise ValueError(
                "config must include 'target_column' for utility evaluation. "
                "You may also pass 'target', which will be used as target_column."
            )

    # Keep all three evaluation modules active by default.
    normalized.setdefault("fidelity_metrics", list(DEFAULT_FIDELITY_METRICS))
    normalized.setdefault("utility_metrics", list(DEFAULT_UTILITY_METRICS))
    normalized.setdefault("privacy_metrics", list(DEFAULT_PRIVACY_METRICS))

    # Evaluator requires holdout split instructions for utility/privacy metrics.
    has_holdout_index = "holdout_index" in normalized
    has_holdout_seed_and_size = "holdout_seed" in normalized and "holdout_size" in normalized
    if not has_holdout_index and not has_holdout_seed_and_size:
        normalized["holdout_size"] = 0.2
        normalized["holdout_seed"] = 0

    return normalized


def evaluate(
    real_data_path: Union[str, Path],
    synth_data_path: Union[str, Path],
    column_name_to_datatype: Mapping[str, str],
    config: Mapping[str, Any],
    save_path: Union[str, Path],
) -> Dict[str, Any]:
    """Run evaluation from CSV paths, save artifacts, and return the report."""
    from .evaluator import EvaluationPipeline

    if not column_name_to_datatype:
        raise ValueError("column_name_to_datatype must be a non-empty mapping.")

    real_path = Path(real_data_path)
    synth_path = Path(synth_data_path)

    if not real_path.exists():
        raise FileNotFoundError(f"real_data_path does not exist: {real_path}")
    if not synth_path.exists():
        raise FileNotFoundError(f"synth_data_path does not exist: {synth_path}")

    real_data = pd.read_csv(real_path)
    synth_data = pd.read_csv(synth_path)

    normalized_config = _normalize_config(config)

    evaluation_pipeline = EvaluationPipeline(
        real_data=real_data,
        synth_data=synth_data,
        column_name_to_datatype=dict(column_name_to_datatype),
        config=normalized_config,
        save_path=str(Path(save_path)),
    )

    evaluation_pipeline.run_pipeline()
    report = evaluation_pipeline.get_report()

    if report is None:
        raise RuntimeError("Evaluation completed without producing a report.")

    return report
