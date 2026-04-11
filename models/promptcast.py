from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.serialize import SerializerSettings, deserialize_str, serialize_arr
from models.model_registry import get_model_spec, get_resolved_default_model
from models.providers import get_provider
from models.providers.base import GenerationRequest
from models.tokenization import tokenize_text


@dataclass
class Scaler:
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x



def get_scaler(history, alpha=0.9, beta=0.3, basic=False):
    history = history[~np.isnan(history)]
    min_ = np.min(history) - beta * (np.max(history) - np.min(history))
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha), 0.01)

        def transform(x):
            return x / q

        def inv_transform(x):
            return x * q

        return Scaler(transform=transform, inv_transform=inv_transform)
    if alpha == -1:
        q = 1
    else:
        q = np.quantile(history - min_, alpha)
        if q == 0:
            q = 1

    def transform(x):
        return (x - min_) / q

    def inv_transform(x):
        return x * q + min_

    return Scaler(transform=transform, inv_transform=inv_transform)



def get_token_ids(tokens, model):
    spec = get_model_spec(model)
    ids = []
    for token in tokens:
        ids.extend(
            tokenize_text(
                token,
                tokenizer_name_or_alias=spec.tokenizer_name_or_alias,
                require_exact=True,
            )
        )
    return ids



def get_avg_tokens_per_step(input_str, settings):
    input_tokens = sum([1 + len(chunk) / 2 for chunk in input_str.split(settings.time_sep)])
    input_steps = max(1, len(input_str.split(settings.time_sep)))
    return input_tokens / input_steps



def truncate(train, test, scaler, model, settings, prompt="", post_prompt=""):
    spec = get_model_spec(model)
    if spec.context_length is None:
        return train

    def format_prompt(train_series):
        full_series = serialize_arr(scaler.transform(pd.concat([train_series, test]).values), settings)
        return prompt + full_series.rstrip(settings.time_sep) + post_prompt

    token_budget = spec.context_length
    if len(tokenize_text(format_prompt(train), spec.tokenizer_name_or_alias)) <= token_budget:
        return train

    full_train_len = len(train)
    for start in range(len(train)):
        sub_train = train.iloc[start:]
        if len(tokenize_text(format_prompt(sub_train), spec.tokenizer_name_or_alias)) <= token_budget:
            print("Truncated train to %d --> %d timesteps" % (full_train_len, len(sub_train)))
            return sub_train
    raise ValueError("PromptCast input is still too large for model '%s' after truncation." % model)



def sample_completions(model, prompt_text, steps, settings, num_samples, temp, logit_bias=None, top_p=None, stop=None, **kwargs):
    spec = get_model_spec(model)
    provider = get_provider(spec.provider)
    tokens_per_step = get_avg_tokens_per_step(prompt_text, settings)
    extra_body = dict(kwargs)
    if logit_bias and spec.capabilities.supports_logit_bias:
        extra_body["logit_bias"] = logit_bias
    request = GenerationRequest(
        prompt=prompt_text,
        max_new_tokens=max(1, int(tokens_per_step * int(steps * 1.3))),
        temperature=temp,
        num_samples=num_samples,
        top_p=top_p,
        stop=stop,
        system_prompt=spec.provider_options.get("system_prompt"),
        extra_body=extra_body,
    )
    return [sample.text for sample in provider.generate(spec, request)]



def handle_prediction(input_arr, pred, expected_length, strict=False):
    if strict:
        if pred is None or len(pred) < expected_length:
            print("Found invalid prediction")
            return None
        return pred[:expected_length]
    if pred is None:
        print("Warning: prediction failed to be deserialized, replaced with last value")
        return np.full(expected_length, input_arr[-1])
    if len(pred) < expected_length:
        print(
            "Warning: Prediction too short %d < %d, padded with last value"
            % (len(pred), expected_length)
        )
        return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
    if len(pred) > expected_length:
        return pred[:expected_length]
    return pred



def generate_predictions(
    model,
    inputs,
    steps,
    settings: SerializerSettings,
    scalers=None,
    num_samples=1,
    temp=0.3,
    prompts=None,
    post_prompts=None,
    parallel=True,
    return_input_strs=False,
    constrain_tokens=True,
    strict_handling=False,
    **kwargs,
):
    if prompts is None:
        prompts = [""] * len(inputs)
    if post_prompts is None:
        post_prompts = [""] * len(inputs)
    assert len(prompts) == len(inputs)
    assert len(post_prompts) == len(inputs)

    if scalers is None:
        scalers = [Scaler() for _ in inputs]
    else:
        assert len(scalers) == len(inputs)

    transformed_inputs = np.array(
        [scaler.transform(input_array) for input_array, scaler in zip(inputs, scalers)]
    )
    input_strs = [serialize_arr(scaled_input_array, settings) for scaled_input_array in transformed_inputs]
    if post_prompts[0] != "":
        input_strs = [
            prompt + input_str.rstrip(settings.time_sep) + post_prompt
            for input_str, prompt, post_prompt in zip(input_strs, prompts, post_prompts)
        ]
    else:
        input_strs = [prompt + input_str for input_str, prompt in zip(input_strs, prompts)]

    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [token for token in allowed_tokens if len(token) > 0]

    spec = get_model_spec(model)
    logit_bias = {}
    if spec.capabilities.supports_logit_bias:
        strength = 30 if constrain_tokens else 5
        logit_bias = {token_id: strength for token_id in get_token_ids(allowed_tokens, model)}

    def complete(prompt_text):
        return sample_completions(
            model,
            prompt_text,
            steps,
            settings,
            num_samples,
            temp,
            logit_bias,
            **kwargs,
        )

    if parallel and len(inputs) > 1:
        with ThreadPoolExecutor(len(inputs)) as pool:
            completions_list = list(tqdm(pool.map(complete, input_strs), total=len(inputs)))
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]

    def completion_to_pred(completion, transformed_input, inv_transform):
        pred = handle_prediction(
            transformed_input,
            deserialize_str(completion, settings, ignore_last=False, steps=steps),
            expected_length=steps,
            strict=strict_handling,
        )
        if pred is not None:
            return inv_transform(pred)
        return None

    preds = [
        [completion_to_pred(completion, transformed_input, scaler.inv_transform) for completion in completions]
        for completions, transformed_input, scaler in zip(completions_list, transformed_inputs, scalers)
    ]
    if return_input_strs:
        return preds, completions_list, input_strs
    return preds, completions_list



def get_promptcast_predictions_data(
    train,
    test,
    model=None,
    settings=None,
    num_samples=10,
    temp=0.8,
    dataset_name="dataset",
    **kwargs,
):
    model = model or get_resolved_default_model()
    spec = get_model_spec(model)
    compute_nll = kwargs.pop("compute_nll", False)
    if compute_nll:
        raise NotImplementedError(
            "PromptCast scoring/NLL has not been ported to the Qwen-native refactor. "
            "The original implementation depended on completion-logprob semantics; the current PromptCast path "
            "is sampling-only and should be run with compute_nll=False."
        )

    if settings is None:
        settings = SerializerSettings(
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
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    if not isinstance(train, list):
        train = [train]
        test = [test]

    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index=pd.RangeIndex(len(train[i]), len(test[i]) + len(train[i])))

    test_len = len(test[0])
    assert all(len(series) == test_len for series in test)

    scalers = [Scaler() for _ in range(len(train))]
    prompt = "The values in the %s for the past %d time steps are " % (dataset_name, len(train[0]))
    post_prompt = (
        ". What will the values for the next %d time steps will be? "
        "The values for the next %d time steps will be "
    ) % (len(test[0]), len(test[0]))

    for i in range(len(train)):
        train[i] = truncate(train[i], test[i], scalers[i], model, settings, prompt=prompt, post_prompt=post_prompt)

    prompts = [prompt] * len(train)
    post_prompts = [post_prompt] * len(train)
    inputs = [train[i].values for i in range(len(train))]
    steps = test_len

    samples = None
    medians = None
    completions_list = None
    input_strs = None
    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(
            model,
            inputs,
            steps,
            settings,
            scalers,
            num_samples=num_samples,
            temp=temp,
            prompts=prompts,
            post_prompts=post_prompts,
            parallel=True,
            return_input_strs=True,
            constrain_tokens=False,
            strict_handling=True,
            **kwargs,
        )
        samples = [pd.DataFrame(np.array([pred for pred in preds[i] if pred is not None]), columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        print("Got %d properly formatted samples" % len(samples))
        medians = medians if len(medians) > 1 else medians[0]

    return {
        "samples": samples,
        "median": medians,
        "info": {
            "Method": model,
            "Provider": spec.provider,
            "APImodel": spec.api_model_name,
            "PromptCastScoring": "sampling_only",
        },
        "completions_list": completions_list,
        "input_strs": input_strs,
        "NLL/D": None,
    }
