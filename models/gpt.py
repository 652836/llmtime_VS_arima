"""Legacy module name, new generic remote-model helpers.

The paper-era code concentrated provider-specific sampling and teacher-forced
scoring in this file. In the Qwen refactor, these helpers stay import-compatible
but route through the provider abstraction so upper forecasting logic no longer
depends on OpenAI-only SDK semantics.
"""

import numpy as np
from jax import grad, vmap

from data.serialize import SerializerSettings, serialize_arr
from models.model_registry import get_model_spec
from models.providers import get_provider
from models.providers.base import GenerationRequest, ScoreRequest
from models.tokenization import tokenize_text


_CHAT_COMPLETION_INSTRUCTION = (
    "Continue the following numeric time series. "
    "Return only the continuation as serialized numbers and separators, "
    "with no commentary or surrounding text.\n\n"
)


def tokenize_fn(text, model):
    spec = get_model_spec(model)
    return tokenize_text(text, tokenizer_name_or_alias=spec.tokenizer_name_or_alias)



def get_allowed_ids(strs, model):
    spec = get_model_spec(model)
    ids = []
    for token in strs:
        ids.extend(
            tokenize_text(
                token,
                tokenizer_name_or_alias=spec.tokenizer_name_or_alias,
                require_exact=True,
            )
        )
    return ids



def _prepare_prompt(spec, input_str, steps):
    if spec.mode == "chat":
        return (
            _CHAT_COMPLETION_INSTRUCTION
            + "Rules:\n"
            + f"- Output exactly {steps} additional values.\n"
            + "- Do not repeat the original history.\n"
            + "- Keep the same numeric serialization style and separators.\n"
            + "- Stop immediately after the last requested value.\n\n"
            + "History:\n"
            + input_str
        )
    return input_str



def _system_prompt(spec):
    return spec.provider_options.get("system_prompt")



def _count_serialized_steps(serialized, settings):
    trimmed = serialized
    if settings.time_sep:
        while trimmed.endswith(settings.time_sep):
            trimmed = trimmed[: -len(settings.time_sep)]
    if not trimmed:
        return 1
    return max(1, len([chunk for chunk in trimmed.split(settings.time_sep) if chunk]))



def _estimate_generation_budget(spec, input_str, steps, settings):
    num_steps = _count_serialized_steps(input_str, settings)
    try:
        token_count = len(tokenize_text(input_str, tokenizer_name_or_alias=spec.tokenizer_name_or_alias))
    except Exception:
        token_count = len(input_str)
    avg_tokens_per_step = max(1.0, token_count / max(1, num_steps))
    margin = 1.8 if spec.mode == "chat" else 1.4
    return max(8, int(np.ceil(avg_tokens_per_step * steps * margin)))



def gpt_completion_fn(model, input_str, steps, settings, num_samples, temp, top_p=None, stop=None, **kwargs):
    spec = get_model_spec(model)
    provider = get_provider(spec.provider)
    explicit_max_new_tokens = kwargs.pop("max_new_tokens", None)
    max_new_tokens = int(explicit_max_new_tokens) if explicit_max_new_tokens is not None else _estimate_generation_budget(spec, input_str, steps, settings)

    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [token for token in allowed_tokens if len(token) > 0]

    extra_body = dict(kwargs)
    if spec.capabilities.supports_logit_bias:
        logit_bias = {token_id: 30 for token_id in get_allowed_ids(allowed_tokens, model)}
        extra_body["logit_bias"] = logit_bias

    prompt = _prepare_prompt(spec, input_str, steps)
    request = GenerationRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        num_samples=num_samples,
        top_p=top_p,
        stop=stop,
        system_prompt=_system_prompt(spec),
        extra_body=extra_body,
    )
    return [sample.text for sample in provider.generate(spec, request)]



def score_text(model, prompt, temp, logprobs=5):
    spec = get_model_spec(model)
    provider = get_provider(spec.provider)
    return provider.score(
        spec,
        ScoreRequest(prompt=prompt, temperature=temp, top_logprobs=logprobs),
    )



def gpt_nll_fn(model, input_arr, target_arr, settings: SerializerSettings, transform, count_seps=True, temp=1):
    spec = get_model_spec(model)
    if not spec.capabilities.supports_nll_scoring:
        raise NotImplementedError(
            "Model '%s' (provider=%s) does not support teacher-forced NLL scoring in the current setup. Use a provider/model with supports_nll_scoring=True for exact_nll_autotune."
            % (model, spec.provider)
        )

    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    assert input_str.endswith(settings.time_sep), (
        "Input string must end with %r, got %r" % (settings.time_sep, input_str)
    )

    scored = score_text(model=model, prompt=input_str + target_str, temp=temp, logprobs=5)
    logprobs = np.array(scored.token_logprobs, dtype=np.float32)
    tokens = np.array(scored.tokens)
    top5logprobs = scored.top_logprobs

    seps = tokens == settings.time_sep
    target_start = np.argmax(np.cumsum(seps) == len(input_arr)) + 1
    logprobs = logprobs[target_start:]
    tokens = tokens[target_start:]
    top5logprobs = top5logprobs[target_start:]
    seps = tokens == settings.time_sep

    assert len(logprobs[seps]) == len(target_arr), (
        "There should be one separator per target. Got %d separators and %d targets."
        % (len(logprobs[seps]), len(target_arr))
    )

    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [
        settings.time_sep,
        settings.plus_sign,
        settings.minus_sign,
        settings.bit_sep + settings.decimal_point,
    ]
    allowed_tokens = {token for token in allowed_tokens if len(token) > 0}

    p_extra = np.array(
        [
            sum(np.exp(value) for key, value in top5logprobs[i].items() if key not in allowed_tokens)
            for i in range(len(top5logprobs))
        ]
    )
    if settings.bit_sep == "":
        p_extra = 0

    adjusted_logprobs = logprobs - np.log(1 - p_extra)
    digits_bits = -adjusted_logprobs[~seps].sum()
    seps_bits = -adjusted_logprobs[seps].sum()
    bpd = digits_bits / len(target_arr)
    if count_seps:
        bpd += seps_bits / len(target_arr)

    transformed_nll = bpd - settings.prec * np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll - avg_logdet_dydx
