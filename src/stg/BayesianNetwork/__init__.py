try:
    from synthcity.plugins import Plugins
    from .bayesian_network_synthesizer import BayesianNetworkSynthesizer
    __all__ = ['BayesianNetworkSynthesizer']
except (ImportError, AttributeError) as e:
    raise ImportError(f"BayesianNetwork requires synthcity to be installed: {e}")