try:
    from .tabpfn_unsupervised_synthesizer import TabPFNUnsupervisedSynthesizer, TABPFN_EXTENSIONS_AVAILABLE
    __all__ = ["TabPFNUnsupervisedSynthesizer", "TABPFN_EXTENSIONS_AVAILABLE"]
except ImportError:
    TABPFN_EXTENSIONS_AVAILABLE = False
    __all__ = ["TABPFN_EXTENSIONS_AVAILABLE"]
