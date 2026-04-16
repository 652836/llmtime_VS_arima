import os
from collections.abc import Iterable

from data.serialize import SerializerSettings
from configs.runtime import get_default_model
from models.model_registry import (
    get_default_autotune_mode,
    get_model_api_name,
    get_model_capabilities,
    get_model_spec,
    get_supported_autotune_modes,
)


def _env_text(name, default=None):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name, default=None):
    value = _env_text(name)
    if value is None:
        return default
    return int(value)


def _env_optional_bool(name, default=None):
    value = _env_text(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("Environment variable %s must be a boolean string, got %r" % (name, value))


def _is_grid_iterable(value):
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict, tuple))


def _candidate_values(values, enabled):
    values = list(values)
    if enabled:
        return values
    return values[0]


def _resolve_llmtime_autotune_mode(model):
    override = _env_text("LLMTIME_AUTOTUNE_MODE")
    supported_modes = tuple(get_supported_autotune_modes(model))
    mode = (override or get_default_autotune_mode(model)).lower()
    if mode not in supported_modes:
        raise ValueError(
            "Model %r does not support autotune_mode=%r. Supported modes: %s"
            % (model, mode, ", ".join(supported_modes))
        )
    return mode


def _resolve_promptcast_autotune_mode(model):
    override = _env_text("LLMTIME_PROMPTCAST_AUTOTUNE_MODE") or _env_text("LLMTIME_AUTOTUNE_MODE")
    supported_modes = tuple(get_supported_autotune_modes(model))
    fallback = "validation_metric" if "validation_metric" in supported_modes else "disabled"
    mode = (override or fallback).lower()
    if mode not in supported_modes and mode != "nll":
        raise ValueError(
            "Model %r does not support PromptCast autotune_mode=%r. Supported provider modes: %s"
            % (model, mode, ", ".join(supported_modes))
        )
    return mode


def materialize_single_hyper(hyper_grid):
    materialized = {}
    for key, value in hyper_grid.items():
        if _is_grid_iterable(value):
            values = list(value)
            if not values:
                raise ValueError("Hyperparameter %r cannot use an empty candidate list." % key)
            materialized[key] = values[0]
        else:
            materialized[key] = value
    return materialized


def describe_model_capabilities(model):
    capabilities = get_model_capabilities(model)
    spec = get_model_spec(model)
    return {
        "model": model,
        "api_model_name": spec.api_model_name,
        "supports_sampling": capabilities.supports_sampling,
        "supports_nll_scoring": capabilities.supports_nll_scoring,
        "supported_autotune_modes": list(capabilities.supported_autotune_modes),
        "default_autotune_mode": capabilities.default_autotune_mode,
    }


def get_llmtime_model_name(model=None):
    resolved_model = model or get_default_model()
    return "LLMTime %s" % get_model_api_name(resolved_model)


def get_promptcast_model_name(model=None):
    resolved_model = model or get_default_model()
    return "PromptCast %s" % get_model_api_name(resolved_model)


DEFAULT_REMOTE_MODEL = get_default_model()
DEFAULT_REMOTE_MODEL_SPEC = get_model_spec(DEFAULT_REMOTE_MODEL)
DEFAULT_REMOTE_API_MODEL = DEFAULT_REMOTE_MODEL_SPEC.api_model_name
DEFAULT_REMOTE_CAPABILITIES = get_model_capabilities(DEFAULT_REMOTE_MODEL)
DEFAULT_LLMTIME_MODEL_NAME = get_llmtime_model_name(DEFAULT_REMOTE_MODEL)
DEFAULT_PROMPTCAST_MODEL_NAME = get_promptcast_model_name(DEFAULT_REMOTE_MODEL)
DEFAULT_SUPPORTED_AUTOTUNE_MODES = tuple(get_supported_autotune_modes(DEFAULT_REMOTE_MODEL))
DEFAULT_AUTOTUNE_MODE = _resolve_llmtime_autotune_mode(DEFAULT_REMOTE_MODEL)
DEFAULT_PROMPTCAST_AUTOTUNE_MODE = _resolve_promptcast_autotune_mode(DEFAULT_REMOTE_MODEL)
DEFAULT_AUTOTUNE_METRIC = (_env_text("LLMTIME_AUTOTUNE_METRIC", "mae") or "mae").lower()
DEFAULT_AUTOTUNE_AGGREGATION = (_env_text("LLMTIME_AUTOTUNE_AGGREGATION", "median") or "median").lower()
DEFAULT_VALIDATION_WINDOW_STRATEGY = (
    _env_text("LLMTIME_VALIDATION_WINDOW_STRATEGY", "forecast_horizon") or "forecast_horizon"
).lower()
DEFAULT_AUTOTUNE_NUM_SAMPLES = _env_int(
    "LLMTIME_AUTOTUNE_NUM_SAMPLES",
    2 if DEFAULT_AUTOTUNE_MODE == "validation_metric" else None,
)
DEFAULT_AUTOTUNE_MAX_CANDIDATES = _env_int(
    "LLMTIME_AUTOTUNE_MAX_CANDIDATES",
    4 if DEFAULT_AUTOTUNE_MODE == "validation_metric" else None,
)
DEFAULT_AUTOTUNE_PARALLEL = _env_optional_bool(
    "LLMTIME_AUTOTUNE_PARALLEL",
    False if DEFAULT_AUTOTUNE_MODE == "validation_metric" else None,
)
DEFAULT_PROMPTCAST_AUTOTUNE_NUM_SAMPLES = _env_int(
    "LLMTIME_PROMPTCAST_AUTOTUNE_NUM_SAMPLES",
    2 if DEFAULT_PROMPTCAST_AUTOTUNE_MODE == "validation_metric" else None,
)
DEFAULT_PROMPTCAST_AUTOTUNE_MAX_CANDIDATES = _env_int(
    "LLMTIME_PROMPTCAST_AUTOTUNE_MAX_CANDIDATES",
    2 if DEFAULT_PROMPTCAST_AUTOTUNE_MODE == "validation_metric" else None,
)
DEFAULT_PROMPTCAST_AUTOTUNE_PARALLEL = _env_optional_bool(
    "LLMTIME_PROMPTCAST_AUTOTUNE_PARALLEL",
    False if DEFAULT_PROMPTCAST_AUTOTUNE_MODE == "validation_metric" else None,
)

LLMTIME_SERIALIZER_SETTINGS = SerializerSettings(
    base=10,
    prec=3,
    signed=True,
    half_bin_correction=True,
)
PROMPTCAST_SERIALIZER_SETTINGS = SerializerSettings(
    base=10,
    prec=0,
    signed=True,
    time_sep=", ",
    bit_sep="",
    plus_sign="",
    minus_sign="-",
    half_bin_correction=False,
    decimal_point="",
)

DEFAULT_LLMTIME_HYPERS = {
    "model": DEFAULT_REMOTE_MODEL,
    "temp": _candidate_values([0.2, 0.7], DEFAULT_AUTOTUNE_MODE != "disabled"),
    "alpha": _candidate_values([0.9, 0.95], DEFAULT_AUTOTUNE_MODE != "disabled"),
    "beta": 0.3,
    "basic": False,
    "compute_nll": DEFAULT_AUTOTUNE_MODE == "nll",
    "autotune_mode": DEFAULT_AUTOTUNE_MODE,
    "autotune_metric": DEFAULT_AUTOTUNE_METRIC,
    "autotune_num_samples": DEFAULT_AUTOTUNE_NUM_SAMPLES,
    "autotune_max_candidates": DEFAULT_AUTOTUNE_MAX_CANDIDATES,
    "autotune_parallel": DEFAULT_AUTOTUNE_PARALLEL,
    "autotune_aggregation": DEFAULT_AUTOTUNE_AGGREGATION,
    "validation_window_strategy": DEFAULT_VALIDATION_WINDOW_STRATEGY,
    "settings": LLMTIME_SERIALIZER_SETTINGS,
}

DEFAULT_PROMPTCAST_HYPERS = {
    "model": DEFAULT_REMOTE_MODEL,
    "temp": _candidate_values([0.2, 0.7], DEFAULT_PROMPTCAST_AUTOTUNE_MODE != "disabled"),
    "compute_nll": False,
    "autotune_mode": DEFAULT_PROMPTCAST_AUTOTUNE_MODE,
    "autotune_metric": DEFAULT_AUTOTUNE_METRIC,
    "autotune_num_samples": DEFAULT_PROMPTCAST_AUTOTUNE_NUM_SAMPLES,
    "autotune_max_candidates": DEFAULT_PROMPTCAST_AUTOTUNE_MAX_CANDIDATES,
    "autotune_parallel": DEFAULT_PROMPTCAST_AUTOTUNE_PARALLEL,
    "autotune_aggregation": DEFAULT_AUTOTUNE_AGGREGATION,
    "validation_window_strategy": DEFAULT_VALIDATION_WINDOW_STRATEGY,
    "settings": PROMPTCAST_SERIALIZER_SETTINGS,
}

DEFAULT_LLMTIME_FIXED_HYPER = materialize_single_hyper(DEFAULT_LLMTIME_HYPERS)
DEFAULT_PROMPTCAST_FIXED_HYPER = materialize_single_hyper(DEFAULT_PROMPTCAST_HYPERS)

QWEN_LLMTIME_HYPERS = DEFAULT_LLMTIME_HYPERS
QWEN_PROMPTCAST_HYPERS = DEFAULT_PROMPTCAST_HYPERS
QWEN_LLMTIME_FIXED_HYPER = DEFAULT_LLMTIME_FIXED_HYPER
QWEN_PROMPTCAST_FIXED_HYPER = DEFAULT_PROMPTCAST_FIXED_HYPER


def is_sampling_only_model(model):
    return not get_model_capabilities(model).supports_nll_scoring


def is_remote_model(model):
    return model not in {"gp", "arima", "TCN", "N-BEATS", "N-HiTS"}
