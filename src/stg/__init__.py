from .tableSynthesizer import TableSynthesizer


def evaluate(*args, **kwargs):
    """Lazy evaluation entrypoint to keep evaluation dependencies optional."""
    from .evaluation import evaluate as _evaluate

    return _evaluate(*args, **kwargs)


def generate_column_name_to_datatype(df):
    """Lazy metadata helper for evaluator column type mapping."""
    from .evaluation import generate_column_name_to_datatype as _generate

    return _generate(df)


__all__ = ["TableSynthesizer", "evaluate", "generate_column_name_to_datatype"]
