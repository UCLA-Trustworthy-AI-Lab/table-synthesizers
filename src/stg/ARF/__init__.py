try:
    from synthcity.plugins import Plugins
    from .arf_synthesizer import ARFSynthesizer
    __all__ = ['ARFSynthesizer']
except (ImportError, AttributeError) as e:
    raise ImportError(f"ARF requires synthcity to be installed: {e}")