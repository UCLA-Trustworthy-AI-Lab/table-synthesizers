try:
    from .cllm_synthesizer import CLLMSynthesizer, OPENAI_AVAILABLE
    __all__ = ["CLLMSynthesizer", "OPENAI_AVAILABLE"]
except ImportError:
    OPENAI_AVAILABLE = False
    __all__ = ["OPENAI_AVAILABLE"]
