"""
Provider factory – returns the correct FaceGenerator implementation based on
the application settings.

Usage
-----
    from generation.provider_factory import get_provider

    provider = get_provider()   # uses settings.effective_provider
    result = provider.generate(request)

Adding a new provider
---------------------
1.  Create a new class in generation/ that inherits from FaceGenerator.
2.  Import it here.
3.  Add an entry to the _REGISTRY dict.
4.  Set GENERATION_PROVIDER=<your_key> in .env.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from generation.interface import FaceGenerator

logger = logging.getLogger(__name__)

# Registry maps provider name strings to lazy import callables.
# Using callables avoids importing heavy dependencies (torch, diffusers, etc.)
# at module load time – only the selected provider is instantiated.
def _make_mock() -> FaceGenerator:
    from generation.mock_generator import MockFaceGenerator
    return MockFaceGenerator()


def _make_kaggle_batch() -> FaceGenerator:
    from generation.kaggle_provider import KaggleBatchProvider
    return KaggleBatchProvider()


_REGISTRY: dict[str, callable] = {
    "MOCK": _make_mock,
    "KAGGLE_BATCH": _make_kaggle_batch,
}


def get_provider(provider_name: str | None = None) -> FaceGenerator:
    """
    Return a FaceGenerator instance for the given provider name.

    Parameters
    ----------
    provider_name : str | None
        Provider key to instantiate.  If None, reads from
        ``settings.effective_provider``.

    Raises
    ------
    ValueError
        If the provider name is not registered.
    """
    if provider_name is None:
        from backend.app.core.config import settings
        provider_name = settings.effective_provider

    provider_name = provider_name.upper()
    factory = _REGISTRY.get(provider_name)

    if factory is None:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown generation provider '{provider_name}'. "
            f"Available providers: {available}"
        )

    logger.info("Instantiating generation provider: %s", provider_name)
    return factory()


@lru_cache(maxsize=1)
def get_cached_provider() -> FaceGenerator:
    """
    Return a cached (singleton) provider instance.

    Use this in services that are instantiated once per process (e.g. the
    synchronous generation service).  Do NOT use this when the provider
    selection can change at runtime.
    """
    return get_provider()
