"""BellowsKV: cross-layer shared-latent KV-cache compression (training-free).

v1 ships BellowsKV-A, the paper's deployment variant: n=4 layers share one INT4 key latent and one INT4 value
latent; the key read-head commutes with RoPE, so decode scores and values are taken directly in the latent by
two Triton kernels, with nothing rebuilt.

    import bellowskv
    model = bellowskv.enable(model, tokenizer)     # fits the codec on first use (cached afterwards)
    out = model.generate(**inputs, max_new_tokens=256)
"""
from .config import BellowsConfig
from .api import enable, disable, fit
from .runtime import BellowsCache

__version__ = "0.1.0.dev0"
__all__ = ["BellowsConfig", "enable", "disable", "fit", "BellowsCache", "__version__"]
