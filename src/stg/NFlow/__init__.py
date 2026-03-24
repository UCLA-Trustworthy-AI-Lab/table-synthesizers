try:
    from synthcity.plugins import Plugins
    from .nflow_synthesizer import NFlowSynthesizer
    __all__ = ['NFlowSynthesizer']
except (ImportError, AttributeError) as e:
    raise ImportError(f"NFlow requires synthcity to be installed: {e}")