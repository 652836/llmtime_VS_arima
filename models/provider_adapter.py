"""Backward-compatible adapter over the new provider/model registry layer.

This file remains to avoid breaking notebooks that imported `configure_model`
during earlier migration attempts. The mainline refactor now lives in:

- configs/runtime.py
- models/model_registry.py
- models/providers/
"""

from models.gpt import score_text as _score_text
from models.model_registry import configure_model, get_model_capabilities, get_model_spec
from models.providers import get_provider
from models.providers.base import GenerationRequest, ModelSpec
from models.tokenization import get_encoding as _get_tokenizer_encoding


ModelConfig = ModelSpec



def get_context_length(model):
    return get_model_spec(model).context_length



def get_encoding(model):
    spec = get_model_spec(model)
    return _get_tokenizer_encoding(spec.tokenizer_name_or_alias)



def sample_text(
    model,
    input_str,
    settings,
    num_samples,
    temp,
    max_tokens,
    logit_bias=None,
    top_p=None,
    stop=None,
):
    spec = get_model_spec(model)
    provider = get_provider(spec.provider)
    extra_body = {}
    if logit_bias and spec.capabilities.supports_logit_bias:
        extra_body["logit_bias"] = logit_bias
    request = GenerationRequest(
        prompt=input_str,
        max_new_tokens=max_tokens,
        temperature=temp,
        num_samples=num_samples,
        top_p=top_p,
        stop=stop,
        system_prompt=spec.provider_options.get("system_prompt"),
        extra_body=extra_body,
    )
    return [sample.text for sample in provider.generate(spec, request)]



def score_text(model, prompt, temp, logprobs=5):
    return _score_text(model=model, prompt=prompt, temp=temp, logprobs=logprobs)


__all__ = [
    "ModelConfig",
    "configure_model",
    "get_context_length",
    "get_encoding",
    "get_model_capabilities",
    "get_model_spec",
    "sample_text",
    "score_text",
]
