import copy
import json
import os
from pathlib import Path


DEFAULT_RUNTIME_CONFIG = {
    "default_model": "qwen-plus",
    "max_concurrency": 4,
    "request_timeout_seconds": 120,
    "providers": {
        "qwen": {
            "api_key": None,
            "api_key_env": "DASHSCOPE_API_KEY",
            "base_url": None,
            "base_url_env": "DASHSCOPE_BASE_URL",
            "enable_thinking": False,
            "thinking_budget": None,
            "result_format": "message",
        },
        "legacy_openai": {
            "api_key": None,
            "api_key_env": "OPENAI_API_KEY",
            "base_url": None,
            "base_url_env": "OPENAI_BASE_URL",
        },
        "local_hf": {},
    },
    "models": {},
}


_RUNTIME_CONFIG_CACHE = None


def _deep_merge(base, overlay):
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_bool(value):
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Expected a boolean-like value, got %r" % value)


def _parse_int(value):
    if value in {None, ""}:
        return None
    return int(value)


def _load_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}


def _candidate_config_paths(explicit_path=None):
    repo_root = Path(__file__).resolve().parents[1]
    paths = []
    if explicit_path:
        paths.append(Path(explicit_path))
    env_path = os.getenv("LLMTIME_CONFIG")
    if env_path:
        paths.append(Path(env_path))
    paths.extend([
        repo_root / "configs" / "runtime.local.json",
        repo_root / "configs" / "runtime.json",
    ])
    return paths


def _env_overrides():
    providers = {}
    models = {}

    if os.getenv("DASHSCOPE_API_KEY"):
        providers.setdefault("qwen", {})["api_key"] = os.getenv("DASHSCOPE_API_KEY")
    if os.getenv("DASHSCOPE_BASE_URL"):
        providers.setdefault("qwen", {})["base_url"] = os.getenv("DASHSCOPE_BASE_URL")
    if os.getenv("LLMTIME_QWEN_ENABLE_THINKING") is not None:
        providers.setdefault("qwen", {})["enable_thinking"] = _parse_bool(
            os.getenv("LLMTIME_QWEN_ENABLE_THINKING")
        )
    if os.getenv("LLMTIME_QWEN_THINKING_BUDGET") is not None:
        providers.setdefault("qwen", {})["thinking_budget"] = _parse_int(
            os.getenv("LLMTIME_QWEN_THINKING_BUDGET")
        )

    if os.getenv("OPENAI_API_KEY"):
        providers.setdefault("legacy_openai", {})["api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_BASE_URL"):
        providers.setdefault("legacy_openai", {})["base_url"] = os.getenv("OPENAI_BASE_URL")

    if os.getenv("LLMTIME_QWEN_MODEL"):
        models.setdefault("qwen-plus", {})["api_model_name"] = os.getenv("LLMTIME_QWEN_MODEL")
    if os.getenv("LLMTIME_QWEN_TOKENIZER_ALIAS"):
        models.setdefault("qwen-plus", {})["tokenizer_name_or_alias"] = os.getenv(
            "LLMTIME_QWEN_TOKENIZER_ALIAS"
        )
    if os.getenv("LLMTIME_QWEN_CONTEXT_LENGTH"):
        models.setdefault("qwen-plus", {})["context_length"] = _parse_int(
            os.getenv("LLMTIME_QWEN_CONTEXT_LENGTH")
        )
    if os.getenv("LLMTIME_QWEN_SNAPSHOT_MODEL"):
        models.setdefault("qwen-plus-snapshot-logprobs", {})["api_model_name"] = os.getenv(
            "LLMTIME_QWEN_SNAPSHOT_MODEL"
        )

    overrides = {}
    if os.getenv("LLMTIME_DEFAULT_MODEL"):
        overrides["default_model"] = os.getenv("LLMTIME_DEFAULT_MODEL")
    if os.getenv("LLMTIME_MAX_CONCURRENCY"):
        overrides["max_concurrency"] = _parse_int(os.getenv("LLMTIME_MAX_CONCURRENCY"))
    if os.getenv("LLMTIME_REQUEST_TIMEOUT_SECONDS"):
        overrides["request_timeout_seconds"] = _parse_int(
            os.getenv("LLMTIME_REQUEST_TIMEOUT_SECONDS")
        )
    if providers:
        overrides["providers"] = providers
    if models:
        overrides["models"] = models
    return overrides


def get_runtime_config(path=None, refresh=False):
    global _RUNTIME_CONFIG_CACHE
    if refresh:
        _RUNTIME_CONFIG_CACHE = None
    if _RUNTIME_CONFIG_CACHE is not None:
        return copy.deepcopy(_RUNTIME_CONFIG_CACHE)

    config = copy.deepcopy(DEFAULT_RUNTIME_CONFIG)
    for candidate in _candidate_config_paths(path):
        if candidate.exists():
            config = _deep_merge(config, _load_json(candidate))
    config = _deep_merge(config, _env_overrides())
    _RUNTIME_CONFIG_CACHE = config
    return copy.deepcopy(_RUNTIME_CONFIG_CACHE)


def reset_runtime_config_cache():
    global _RUNTIME_CONFIG_CACHE
    _RUNTIME_CONFIG_CACHE = None


def get_provider_config(provider_name):
    config = get_runtime_config()
    provider_config = copy.deepcopy(config.get("providers", {}).get(provider_name, {}))

    api_key = provider_config.get("api_key")
    api_key_env = provider_config.get("api_key_env")
    if api_key is None and api_key_env:
        api_key = os.getenv(api_key_env)
    provider_config["api_key"] = api_key

    base_url = provider_config.get("base_url")
    base_url_env = provider_config.get("base_url_env")
    if base_url is None and base_url_env:
        base_url = os.getenv(base_url_env)
    provider_config["base_url"] = base_url
    return provider_config


def get_model_override(logical_model_name):
    config = get_runtime_config()
    return copy.deepcopy(config.get("models", {}).get(logical_model_name, {}))


def get_default_model():
    return get_runtime_config().get("default_model", "qwen-plus")


def _redact(value):
    if not value:
        return None
    if len(value) <= 8:
        return "***"
    return "%s...%s" % (value[:4], value[-4:])


def describe_runtime():
    config = get_runtime_config()
    summary = {
        "default_model": config.get("default_model"),
        "max_concurrency": config.get("max_concurrency"),
        "request_timeout_seconds": config.get("request_timeout_seconds"),
        "providers": {},
    }
    try:
        from models.model_registry import get_model_spec

        default_spec = get_model_spec(summary["default_model"])
        summary["default_model_api_name"] = default_spec.api_model_name
        summary["default_model_provider"] = default_spec.provider
    except Exception:
        pass
    for provider_name in sorted(config.get("providers", {})):
        provider_config = get_provider_config(provider_name)
        summary["providers"][provider_name] = {
            "api_key": _redact(provider_config.get("api_key")),
            "base_url": provider_config.get("base_url"),
            "enable_thinking": provider_config.get("enable_thinking"),
            "thinking_budget": provider_config.get("thinking_budget"),
        }
    return summary
