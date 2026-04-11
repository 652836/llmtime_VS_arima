from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.serialize import SerializerSettings, deserialize_str, serialize_arr
from models.llms import get_completion_fn, get_context_length, get_nll_fn, get_tokenization_fn
from models.model_registry import get_resolved_default_model, get_model_spec


@dataclass
class Scaler:
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x



def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha), 0.01)

        def transform(x):
            return x / q

        def inv_transform(x):
            return x * q
    else:
        min_ = np.min(history) - beta * (np.max(history) - np.min(history))
        q = np.quantile(history - min_, alpha)

        def transform(x):
            return (x - min_) / q

        def inv_transform(x):
            return x * q + min_

    return Scaler(transform=transform, inv_transform=inv_transform)



def truncate_input(input_arr, input_str, settings, model):
    tokenization_fn = get_tokenization_fn(model)
    context_length = get_context_length(model)
    if tokenization_fn is not None and context_length is not None:
        input_str_chunks = input_str.split(settings.time_sep)
        for i in range(len(input_str_chunks)):
            truncated_input_str = settings.time_sep.join(input_str_chunks[i:])
            if not truncated_input_str.endswith(settings.time_sep):
                truncated_input_str += settings.time_sep
            tokens = tokenization_fn(truncated_input_str)
            if len(tokens) <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if i > 0:
            print("Warning: Truncated input from %d to %d" % (len(input_arr), len(truncated_input_arr)))
        return truncated_input_arr, truncated_input_str
    return input_arr, input_str



def handle_prediction(pred, expected_length, strict=False):
    if pred is None:
        return None
    if len(pred) < expected_length:
        if strict:
            print(
                "Warning: Prediction too short %d < %d, returning None"
                % (len(pred), expected_length)
            )
            return None
        print(
            "Warning: Prediction too short %d < %d, padded with last value"
            % (len(pred), expected_length)
        )
        return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
    return pred[:expected_length]



def generate_predictions(
    completion_fn,
    input_strs,
    steps,
    settings: SerializerSettings,
    scalers,
    num_samples=1,
    temp=0.7,
    parallel=True,
    strict_handling=False,
    **kwargs,
):
    completions_list = []

    def complete(serialized_input):
        return completion_fn(
            input_str=serialized_input,
            steps=steps,
            settings=settings,
            num_samples=num_samples,
            temp=temp,
            **kwargs,
        )

    if parallel and len(input_strs) > 1:
        print("Running completions in parallel for each input")
        with ThreadPoolExecutor(len(input_strs)) as pool:
            completions_list = list(tqdm(pool.map(complete, input_strs), total=len(input_strs)))
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]

    def completion_to_pred(completion, inv_transform):
        pred = handle_prediction(
            deserialize_str(completion, settings, ignore_last=False, steps=steps),
            expected_length=steps,
            strict=strict_handling,
        )
        if pred is not None:
            return inv_transform(pred)
        return None

    preds = [
        [completion_to_pred(completion, scaler.inv_transform) for completion in completions]
        for completions, scaler in zip(completions_list, scalers)
    ]
    return preds, completions_list, input_strs



def get_llmtime_predictions_data(
    train,
    test,
    model=None,
    settings=None,
    num_samples=10,
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    parallel=True,
    compute_nll=True,
    **kwargs,
):
    model = model or get_resolved_default_model()
    spec = get_model_spec(model)
    completion_fn = get_completion_fn(model)
    nll_fn = get_nll_fn(model)

    if settings is None:
        settings = SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
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
    assert all(len(series) == test_len for series in test), (
        "All test series must have same length, got %r" % [len(series) for series in test]
    )

    scalers = [get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic) for i in range(len(train))]
    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = np.array(
        [scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)]
    )
    input_strs = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in transformed_input_arrs]
    input_arrs, input_strs = zip(
        *[
            truncate_input(input_array, input_str, settings, model)
            for input_array, input_str in zip(input_arrs, input_strs)
        ]
    )

    steps = test_len
    samples = None
    medians = None
    completions_list = None
    if num_samples <= 0 and not compute_nll:
        raise ValueError(
            "Sampling-only mode requires num_samples > 0. "
            "If you need autotune/NLL, use a provider/model with teacher-forced scoring support."
        )

    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(
            completion_fn,
            input_strs,
            steps,
            settings,
            scalers,
            num_samples=num_samples,
            temp=temp,
            parallel=parallel,
            **kwargs,
        )
        samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]

    out_dict = {
        "samples": samples,
        "median": medians,
        "info": {
            "Method": model,
            "Provider": spec.provider,
            "APImodel": spec.api_model_name,
            "SupportsNLL": spec.capabilities.supports_nll,
            "SupportsAutotune": spec.capabilities.supports_autotune,
        },
        "completions_list": completions_list,
        "input_strs": input_strs,
    }

    if compute_nll:
        if nll_fn is None:
            raise NotImplementedError(
                "Model '%s' (provider=%s, api_model=%s) does not support NLL/logprob scoring in the current setup. "
                "For the default Qwen-native path this is expected: DashScope sampling works, but teacher-forced "
                "prompt+target scoring is not exposed, so set compute_nll=False and use fixed hyperparameters instead "
                "of validation autotune."
                % (model, spec.provider, spec.api_model_name)
            )
        bpds = [
            nll_fn(
                input_arr=input_arrs[i],
                target_arr=test[i].values,
                settings=settings,
                transform=scalers[i].transform,
                count_seps=True,
                temp=temp,
            )
            for i in range(len(train))
        ]
        out_dict["NLL/D"] = np.mean(bpds)
    return out_dict
