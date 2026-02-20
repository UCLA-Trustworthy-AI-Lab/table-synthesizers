try:
    from sdv.single_table import GaussianCopulaSynthesizer as _SDVGaussianCopulaSynthesizer

    from .gaussian_copula_synthesizer import GaussianCopulaSynthesizer

    __all__ = ["GaussianCopulaSynthesizer"]
except Exception as e:
    raise ImportError(f"GaussianCopula requires sdv to be installed: {e}")
