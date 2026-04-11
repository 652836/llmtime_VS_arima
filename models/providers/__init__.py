from functools import lru_cache

from models.providers.legacy_openai import LegacyOpenAIProvider
from models.providers.local_hf import LocalHFProvider
from models.providers.qwen import QwenProvider


_PROVIDER_CLASSES = {
    "qwen": QwenProvider,
    "legacy_openai": LegacyOpenAIProvider,
    "local_hf": LocalHFProvider,
}


@lru_cache(maxsize=None)
def get_provider(provider_name):
    if provider_name not in _PROVIDER_CLASSES:
        raise KeyError("Unknown provider '%s'." % provider_name)
    return _PROVIDER_CLASSES[provider_name]()
