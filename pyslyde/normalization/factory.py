# stain_normalization/factory.py

from typing import Dict, Type
from .base import StainNormalizer
from .macenko import MacenkoStainNormalizer
from .vahadane import VahadaneStainNormalizer
from .reinhard import ReinhardStainNormalizer

# Registry of available normalizers
_REGISTRY: Dict[str, Type[StainNormalizer]] = {
    "macenko": MacenkoStainNormalizer,
    "vahadane": VahadaneStainNormalizer,
    "reinhard": ReinhardStainNormalizer,
}


def make_normalizer(method: str, **kwargs) -> StainNormalizer:
    """
    Factory for creating a stain normalizer.

    Parameters
    ----------
    method : str
        Name of the normalization method ('macenko', 'vahadane', 'reinhard').
    **kwargs :
        Arguments forwarded to the class constructor.

    Returns
    -------
    StainNormalizer
        Instance of the requested normalizer.
    """
    try:
        cls = _REGISTRY[method.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(_REGISTRY.keys())}"
        )
    return cls(**kwargs)
