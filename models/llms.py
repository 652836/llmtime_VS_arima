from functools import partial

from models.gpt import gpt_completion_fn, gpt_nll_fn, tokenize_fn as remote_tokenize_fn
from models.llama import llama_completion_fn, llama_nll_fn
from models.llama import tokenize_fn as llama_tokenize_fn
from models.model_registry import get_model_capabilities, get_model_spec


_LEGACY_LLAMA_COMPLETIONS = {
    "llama-7b": partial(llama_completion_fn, model="7b"),
    "llama-13b": partial(llama_completion_fn, model="13b"),
    "llama-70b": partial(llama_completion_fn, model="70b"),
    "llama-7b-chat": partial(llama_completion_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_completion_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_completion_fn, model="70b-chat"),
}


_LEGACY_LLAMA_NLL = {
    "llama-7b": partial(llama_nll_fn, model="7b"),
    "llama-13b": partial(llama_nll_fn, model="13b"),
    "llama-70b": partial(llama_nll_fn, model="70b"),
    "llama-7b-chat": partial(llama_nll_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_nll_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_nll_fn, model="70b-chat"),
}


_LEGACY_LLAMA_TOKENIZATION = {
    "llama-7b": partial(llama_tokenize_fn, model="7b"),
    "llama-13b": partial(llama_tokenize_fn, model="13b"),
    "llama-70b": partial(llama_tokenize_fn, model="70b"),
    "llama-7b-chat": partial(llama_tokenize_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_tokenize_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_tokenize_fn, model="70b-chat"),
}


_LEGACY_LLAMA_CONTEXT_LENGTHS = {
    "llama-7b": 4096,
    "llama-13b": 4096,
    "llama-70b": 4096,
    "llama-7b-chat": 4096,
    "llama-13b-chat": 4096,
    "llama-70b-chat": 4096,
}



def get_completion_fn(model):
    if model in _LEGACY_LLAMA_COMPLETIONS:
        return _LEGACY_LLAMA_COMPLETIONS[model]

    capabilities = get_model_capabilities(model)
    if capabilities.supports_sampling:
        return partial(gpt_completion_fn, model=model)

    raise KeyError("No completion function registered for model %s" % model)



def get_nll_fn(model):
    if model in _LEGACY_LLAMA_NLL:
        return _LEGACY_LLAMA_NLL[model]

    capabilities = get_model_capabilities(model)
    if capabilities.supports_nll_scoring:
        return partial(gpt_nll_fn, model=model)
    return None



def get_tokenization_fn(model):
    if model in _LEGACY_LLAMA_TOKENIZATION:
        return _LEGACY_LLAMA_TOKENIZATION[model]

    spec = get_model_spec(model)
    if spec.tokenizer_name_or_alias is None and spec.context_length is None:
        return None
    return partial(remote_tokenize_fn, model=model)



def get_context_length(model):
    if model in _LEGACY_LLAMA_CONTEXT_LENGTHS:
        return _LEGACY_LLAMA_CONTEXT_LENGTHS[model]
    return get_model_spec(model).context_length
