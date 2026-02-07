try:
    from synthcity.plugins import Plugins
    from .great_synthesizer import GREATSynthesizer
    __all__ = ['GREATSynthesizer']
except (ImportError, AttributeError) as e:
    raise ImportError(f"GREAT requires synthcity to be installed: {e}")