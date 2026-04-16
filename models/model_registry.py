from dataclasses import replace

from configs.runtime import get_default_model, get_model_override
from models.providers.base import ModelCapabilities, ModelSpec


DEFAULT_TIME_SERIES_SYSTEM_PROMPT = (
    "You are a careful time-series forecasting assistant. "
    "Continue the numeric sequence exactly, without commentary, Markdown, "
    "code fences, or explanations."
)


_DASHSCOPE_CAPABILITIES = ModelCapabilities(
    supports_sampling=True,
    supports_logprobs=False,
    supports_nll_scoring=False,
    supports_autotune_nll=False,
    supports_autotune_validation_metric=True,
    default_autotune_mode="validation_metric",
    supported_autotune_modes=("validation_metric", "disabled"),
    supports_reasoning=True,
    supports_chat=True,
    supports_text_completion_like_mode=True,
    supports_logit_bias=False,
)


def _dashscope_generation_spec(
    logical_model_name,
    api_model_name=None,
    *,
    context_length=None,
    tokenizer_name_or_alias="cl100k_base",
    supports_logprobs=False,
    supports_reasoning=True,
    description="",
):
    capabilities = replace(
        _DASHSCOPE_CAPABILITIES,
        supports_logprobs=supports_logprobs,
        supports_reasoning=supports_reasoning,
    )
    extra_generation_defaults = {
        "result_format": "message",
        "enable_thinking": False,
    }
    if supports_logprobs:
        extra_generation_defaults.update({"logprobs": True, "top_logprobs": 5})
    return ModelSpec(
        logical_model_name=logical_model_name,
        provider="qwen",
        api_model_name=api_model_name or logical_model_name,
        tokenizer_name_or_alias=tokenizer_name_or_alias,
        context_length=context_length,
        mode="chat",
        description=description,
        capabilities=capabilities,
        extra_generation_defaults=extra_generation_defaults,
        provider_options={"system_prompt": DEFAULT_TIME_SERIES_SYSTEM_PROMPT},
    )


_MODEL_SPECS = {
    "qwen-plus": _dashscope_generation_spec(
        "qwen-plus",
        context_length=997952,
        description="Default Qwen-native forecasting model via DashScope.",
    ),
    "qwen-plus-latest": _dashscope_generation_spec(
        "qwen-plus-latest",
        context_length=997952,
        description="Latest Qwen Plus alias via DashScope. Defaults to non-thinking mode.",
    ),
    "qwen-plus-2025-07-28": _dashscope_generation_spec(
        "qwen-plus-2025-07-28",
        context_length=997952,
        description="Pinned Qwen Plus snapshot for reproducible notebook and experiment runs.",
    ),
    "qwen-plus-2025-12-01": _dashscope_generation_spec(
        "qwen-plus-2025-12-01",
        context_length=997952,
        description="Pinned Qwen Plus snapshot exposed as a direct model code.",
    ),
    "qwen-plus-snapshot-logprobs": _dashscope_generation_spec(
        "qwen-plus-snapshot-logprobs",
        api_model_name="qwen-plus-2025-12-01",
        context_length=997952,
        supports_logprobs=True,
        description=(
            "Snapshot Qwen Plus route for optional output-token logprobs. "
            "Teacher-forced NLL/autotune still remain disabled."
        ),
    ),
    "qwen3.6-plus": _dashscope_generation_spec(
        "qwen3.6-plus",
        context_length=991808,
        description="Explicit Qwen3.6 Plus route with thinking disabled for numeric forecasting.",
    ),
    "qwen-max": _dashscope_generation_spec(
        "qwen-max",
        description="Qwen Max via DashScope generation API.",
    ),
    "qvq-max-2025-03-25": _dashscope_generation_spec(
        "qvq-max-2025-03-25",
        description="QVQ Max reasoning model exposed through the DashScope generation interface.",
    ),
    "qwen-math-turbo": _dashscope_generation_spec(
        "qwen-math-turbo",
        description="Qwen Math Turbo model code from DashScope/Bailian.",
    ),
    "qwen-coder-turbo-0919": _dashscope_generation_spec(
        "qwen-coder-turbo-0919",
        description="Qwen Coder Turbo model code from DashScope/Bailian.",
    ),
    "qwen2.5-math-7b-instruct": _dashscope_generation_spec(
        "qwen2.5-math-7b-instruct",
        description="Qwen2.5 Math 7B Instruct model code from DashScope/Bailian.",
    ),
    "qwen2.5-14b-instruct": _dashscope_generation_spec(
        "qwen2.5-14b-instruct",
        description="Qwen2.5 14B Instruct model code from DashScope/Bailian.",
    ),
    "qwen2.5-7b-instruct": _dashscope_generation_spec(
        "qwen2.5-7b-instruct",
        description="Qwen2.5 7B Instruct model code from DashScope/Bailian.",
    ),
    "qwen3-32b": _dashscope_generation_spec(
        "qwen3-32b",
        description="Qwen3 32B model code from DashScope/Bailian.",
    ),
    "qwen-mt-flash": _dashscope_generation_spec(
        "qwen-mt-flash",
        description="Qwen MT Flash model code from DashScope/Bailian.",
    ),
    "deepseek-r1-distill-qwen-7b": _dashscope_generation_spec(
        "deepseek-r1-distill-qwen-7b",
        description="DeepSeek R1 Distill Qwen 7B model code routed through DashScope.",
    ),
    "glm-5": _dashscope_generation_spec(
        "glm-5",
        description="GLM-5 model code routed through DashScope generation API.",
    ),
    "qwen-vl-plus-2025-05-07": _dashscope_generation_spec(
        "qwen-vl-plus-2025-05-07",
        description=(
            "Qwen VL Plus snapshot from DashScope/Bailian. The current forecasting pipeline sends text-only "
            "messages, so multimodal abilities are not exercised."
        ),
    ),
    "qwen-vl-plus-latest": _dashscope_generation_spec(
        "qwen-vl-plus-latest",
        description=(
            "Latest Qwen VL Plus alias. The current forecasting pipeline sends text-only messages."
        ),
    ),
    "qwen-vl-ocr-latest": _dashscope_generation_spec(
        "qwen-vl-ocr-latest",
        description=(
            "Latest Qwen VL OCR alias. The current forecasting pipeline sends text-only messages."
        ),
    ),
    "qwen2.5-vl-72b-instruct": _dashscope_generation_spec(
        "qwen2.5-vl-72b-instruct",
        description=(
            "Qwen2.5 VL 72B Instruct model code. Registered for direct selection; current forecasting calls remain text-only."
        ),
    ),
    "qwen2.5-vl-3b-instruct": _dashscope_generation_spec(
        "qwen2.5-vl-3b-instruct",
        description=(
            "Qwen2.5 VL 3B Instruct model code. Registered for direct selection; current forecasting calls remain text-only."
        ),
    ),
    "qwen3-vl-30b-a3b-thinking": _dashscope_generation_spec(
        "qwen3-vl-30b-a3b-thinking",
        description=(
            "Qwen3 VL 30B A3B Thinking model code. Registered for direct selection; current forecasting calls remain text-only."
        ),
    ),
    "qwen3-vl-32b-thinking": _dashscope_generation_spec(
        "qwen3-vl-32b-thinking",
        description=(
            "Qwen3 VL 32B Thinking model code. Registered for direct selection; current forecasting calls remain text-only."
        ),
    ),
    "qwen3-vl-235b-a22b-thinking": _dashscope_generation_spec(
        "qwen3-vl-235b-a22b-thinking",
        description=(
            "Qwen3 VL 235B A22B Thinking model code. Registered for direct selection; current forecasting calls remain text-only."
        ),
    ),
    "text-davinci-003": ModelSpec(
        logical_model_name="text-davinci-003",
        provider="legacy_openai",
        api_model_name="text-davinci-003",
        tokenizer_name_or_alias="p50k_base",
        context_length=4097,
        mode="text_completion",
        description="Legacy paper-era OpenAI completion model. Optional compatibility only.",
        capabilities=ModelCapabilities(
            supports_sampling=True,
            supports_logprobs=True,
            supports_nll_scoring=True,
            supports_autotune_nll=True,
            supports_autotune_validation_metric=True,
            default_autotune_mode="nll",
            supported_autotune_modes=("nll", "validation_metric", "disabled"),
            supports_reasoning=False,
            supports_chat=False,
            supports_text_completion_like_mode=True,
            supports_logit_bias=True,
        ),
    ),
    "gpt-3.5-turbo-instruct": ModelSpec(
        logical_model_name="gpt-3.5-turbo-instruct",
        provider="legacy_openai",
        api_model_name="gpt-3.5-turbo-instruct",
        tokenizer_name_or_alias="cl100k_base",
        context_length=4097,
        mode="text_completion",
        description="Legacy OpenAI instruct model kept only for back-compat comparisons.",
        capabilities=ModelCapabilities(
            supports_sampling=True,
            supports_logprobs=True,
            supports_nll_scoring=True,
            supports_autotune_nll=True,
            supports_autotune_validation_metric=True,
            default_autotune_mode="nll",
            supported_autotune_modes=("nll", "validation_metric", "disabled"),
            supports_reasoning=False,
            supports_chat=False,
            supports_text_completion_like_mode=True,
            supports_logit_bias=True,
        ),
    ),
    "gpt-4": ModelSpec(
        logical_model_name="gpt-4",
        provider="legacy_openai",
        api_model_name="gpt-4",
        tokenizer_name_or_alias="cl100k_base",
        context_length=8192,
        mode="chat",
        description="Legacy OpenAI chat model for historical comparisons only.",
        capabilities=ModelCapabilities(
            supports_sampling=True,
            supports_logprobs=False,
            supports_nll_scoring=False,
            supports_autotune_nll=False,
            supports_autotune_validation_metric=True,
            default_autotune_mode="validation_metric",
            supported_autotune_modes=("validation_metric", "disabled"),
            supports_reasoning=False,
            supports_chat=True,
            supports_text_completion_like_mode=True,
            supports_logit_bias=False,
        ),
        provider_options={"system_prompt": DEFAULT_TIME_SERIES_SYSTEM_PROMPT},
    ),
    "local-hf-placeholder": ModelSpec(
        logical_model_name="local-hf-placeholder",
        provider="local_hf",
        api_model_name="local-hf-placeholder",
        tokenizer_name_or_alias=None,
        context_length=None,
        mode="text_completion",
        description="Placeholder entry for a future local Hugging Face provider.",
        capabilities=ModelCapabilities(
            supports_sampling=False,
            supports_logprobs=False,
            supports_nll_scoring=False,
            supports_autotune_nll=False,
            supports_autotune_validation_metric=False,
            default_autotune_mode="disabled",
            supported_autotune_modes=("disabled",),
            supports_reasoning=False,
            supports_chat=False,
            supports_text_completion_like_mode=True,
            supports_logit_bias=False,
        ),
    ),
}


def _normalize_capability_overrides(override):
    normalized = dict(override)
    if "supports_nll" in normalized and "supports_nll_scoring" not in normalized:
        normalized["supports_nll_scoring"] = normalized.pop("supports_nll")
    if "supports_autotune" in normalized:
        value = normalized.pop("supports_autotune")
        normalized.setdefault("supports_autotune_nll", value)
        normalized.setdefault("supports_autotune_validation_metric", value)
        if value and "supported_autotune_modes" not in normalized:
            normalized["supported_autotune_modes"] = ("nll", "validation_metric", "disabled")
            normalized.setdefault("default_autotune_mode", "nll")
        elif not value and "supported_autotune_modes" not in normalized:
            normalized["supported_autotune_modes"] = ("disabled",)
            normalized.setdefault("default_autotune_mode", "disabled")
    return normalized



def _apply_overrides(spec, override):
    if not override:
        return spec

    capabilities = spec.capabilities
    capability_keys = {
        "supports_sampling",
        "supports_logprobs",
        "supports_nll_scoring",
        "supports_autotune_nll",
        "supports_autotune_validation_metric",
        "default_autotune_mode",
        "supported_autotune_modes",
        "supports_reasoning",
        "supports_chat",
        "supports_text_completion_like_mode",
        "supports_logit_bias",
    }
    capability_update = {}
    override = _normalize_capability_overrides(override)
    for key in list(override.keys()):
        if key in capability_keys:
            capability_update[key] = override.pop(key)
    if "capabilities" in override:
        nested_capabilities = _normalize_capability_overrides(override.pop("capabilities"))
        capability_update.update(nested_capabilities)
    if capability_update:
        if "supported_autotune_modes" in capability_update:
            capability_update["supported_autotune_modes"] = tuple(capability_update["supported_autotune_modes"])
        capabilities = replace(capabilities, **capability_update)

    if "tokenizer_alias" in override and "tokenizer_name_or_alias" not in override:
        override["tokenizer_name_or_alias"] = override.pop("tokenizer_alias")
    if "completion_mode" in override and "mode" not in override:
        override["mode"] = override.pop("completion_mode")
    if "provider_options" in override and spec.provider_options:
        override["provider_options"] = dict(spec.provider_options, **override["provider_options"])
    if "extra_generation_defaults" in override and spec.extra_generation_defaults:
        override["extra_generation_defaults"] = dict(
            spec.extra_generation_defaults, **override["extra_generation_defaults"]
        )
    override["capabilities"] = capabilities
    return replace(spec, **override)



def register_model_spec(spec):
    _MODEL_SPECS[spec.logical_model_name] = spec
    return spec



def configure_model(logical_model_name, **kwargs):
    base_spec = _MODEL_SPECS.get(logical_model_name)
    if base_spec is None:
        provider = kwargs.get("provider")
        api_model_name = kwargs.get("api_model_name", logical_model_name)
        if provider is None:
            raise KeyError(
                "Unknown model '%s'. Provide at least `provider=` when configuring a new model."
                % logical_model_name
            )
        base_spec = ModelSpec(
            logical_model_name=logical_model_name,
            provider=provider,
            api_model_name=api_model_name,
        )
    spec = _apply_overrides(base_spec, dict(kwargs))
    _MODEL_SPECS[logical_model_name] = spec
    return spec



def get_model_spec(logical_model_name):
    if logical_model_name not in _MODEL_SPECS:
        raise KeyError(
            "Unknown model '%s'. Registered models: %s"
            % (logical_model_name, ", ".join(sorted(_MODEL_SPECS)))
        )
    override = get_model_override(logical_model_name)
    return _apply_overrides(_MODEL_SPECS[logical_model_name], override)



def get_model_api_name(logical_model_name):
    return get_model_spec(logical_model_name).api_model_name



def get_model_capabilities(logical_model_name):
    return get_model_spec(logical_model_name).capabilities



def get_supported_autotune_modes(logical_model_name):
    return get_model_capabilities(logical_model_name).supported_autotune_modes



def get_default_autotune_mode(logical_model_name):
    return get_model_capabilities(logical_model_name).default_autotune_mode



def list_model_specs():
    return [get_model_spec(name) for name in sorted(_MODEL_SPECS)]



def list_model_names():
    return sorted(_MODEL_SPECS)



def get_resolved_default_model():
    return get_default_model()
