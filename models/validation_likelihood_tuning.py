from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import is_dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.metrics import compute_point_forecast_metric
from models.model_registry import get_default_autotune_mode, get_model_capabilities
from models.utils import grid_iter


AUTOTUNE_OPTION_KEYS = {
    "autotune_mode",
    "autotune_metric",
    "autotune_num_samples",
    "autotune_max_candidates",
    "autotune_parallel",
    "autotune_aggregation",
    "validation_window_strategy",
}

VALID_AUTOTUNE_MODES = {"nll", "validation_metric", "disabled"}
VALID_AUTOTUNE_METRICS = {"mae", "mse", "smape"}
VALID_AUTOTUNE_AGGREGATIONS = {"median", "mean"}



def strip_autotune_kwargs(kwargs):
    cleaned = dict(kwargs)
    for key in AUTOTUNE_OPTION_KEYS:
        cleaned.pop(key, None)
    return cleaned



def make_validation_dataset(train, n_val, val_length, strategy="forecast_horizon"):
    assert isinstance(train, list), "Train should be a list of series"
    if strategy != "forecast_horizon":
        raise ValueError(
            "Unsupported validation_window_strategy %r. Only 'forecast_horizon' is currently implemented."
            % strategy
        )

    train_minus_val_list, val_list = [], []
    if n_val is None:
        n_val = len(train)
    for train_series in train[:n_val]:
        train_len = max(len(train_series) - val_length, 1)
        train_minus_val, val = train_series[:train_len], train_series[train_len:]
        train_minus_val_list.append(train_minus_val)
        val_list.append(val)

    return train_minus_val_list, val_list, n_val



def _to_1d_array(value):
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)



def _to_2d_array(value):
    if isinstance(value, list):
        if not value:
            return np.empty((0, 0), dtype=float)
        return np.stack([_to_1d_array(item) for item in value], axis=0)
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float).reshape(1, -1)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr



def _aggregate_samples(samples, aggregation):
    if isinstance(samples, list):
        return [_aggregate_samples(sample, aggregation) for sample in samples]
    if isinstance(samples, pd.DataFrame):
        if aggregation == "median":
            return samples.median(axis=0)
        return samples.mean(axis=0)
    arr = np.asarray(samples, dtype=float)
    axis = 0 if arr.ndim > 1 else None
    if aggregation == "median":
        return np.median(arr, axis=axis)
    return np.mean(arr, axis=axis)



def _extract_point_forecast(prediction_dict, aggregation):
    if aggregation == "median" and prediction_dict.get("median") is not None:
        return prediction_dict["median"]
    samples = prediction_dict.get("samples")
    if samples is None:
        return prediction_dict.get("median")
    return _aggregate_samples(samples, aggregation)



def _split_hyper_config(hyper):
    forecast_hyper = dict(hyper)
    autotune_options = {}
    for key in AUTOTUNE_OPTION_KEYS:
        if key in forecast_hyper:
            autotune_options[key] = forecast_hyper.pop(key)
    return forecast_hyper, autotune_options



def _infer_supported_modes(hyper):
    model = hyper.get("model")
    if model:
        return tuple(get_model_capabilities(model).supported_autotune_modes)
    return ("nll", "validation_metric", "disabled")



def _infer_default_mode(hyper):
    model = hyper.get("model")
    if model:
        return get_default_autotune_mode(model)
    return "nll"



def _resolve_shared_autotune_options(hypers, overrides):
    forecast_hypers = []
    option_snapshots = []
    for hyper in hypers:
        forecast_hyper, options = _split_hyper_config(hyper)
        forecast_hypers.append(forecast_hyper)
        option_snapshots.append(options)

    shared_options = dict(option_snapshots[0]) if option_snapshots else {}
    for options in option_snapshots[1:]:
        if options != shared_options:
            raise ValueError(
                "All hyperparameter candidates must share the same autotune meta-options. "
                "Vary forecast hyperparameters only; keep autotune_mode/metric/sample controls fixed."
            )

    explicit_overrides = {
        key: value
        for key, value in overrides.items()
        if key in AUTOTUNE_OPTION_KEYS and value is not None
    }
    shared_options.update(explicit_overrides)
    return forecast_hypers, shared_options



def _resolve_autotune_mode(forecast_hypers, shared_options):
    mode = shared_options.get("autotune_mode")
    if mode is None:
        mode = _infer_default_mode(forecast_hypers[0])
    mode = str(mode).lower()
    if mode not in VALID_AUTOTUNE_MODES:
        raise ValueError(
            "Unsupported autotune_mode %r. Expected one of: disabled, nll, validation_metric." % mode
        )

    supported_modes = _infer_supported_modes(forecast_hypers[0])
    if mode not in supported_modes:
        model = forecast_hypers[0].get("model")
        capabilities = get_model_capabilities(model) if model else None
        suggestion = ""
        if mode == "nll" and capabilities is not None and capabilities.supports_autotune_validation_metric:
            suggestion = " Set autotune_mode='validation_metric' to use forecast-quality tuning instead."
        raise NotImplementedError(
            "Model '%s' does not support autotune_mode='%s'. Supported modes: %s.%s"
            % (model or "<non-registry model>", mode, ", ".join(supported_modes), suggestion)
        )
    return mode, supported_modes



def _sanitize_candidates(forecast_hypers, max_candidates):
    if max_candidates in {None, 0}:
        return forecast_hypers
    return forecast_hypers[: int(max_candidates)]



def _resolve_metric_options(mode, shared_options, parallel):
    metric = str(shared_options.get("autotune_metric", "mae")).lower()
    if metric not in VALID_AUTOTUNE_METRICS:
        raise ValueError(
            "Unsupported autotune_metric %r. Expected one of: mae, mse, smape." % metric
        )
    aggregation = str(shared_options.get("autotune_aggregation", "median")).lower()
    if aggregation not in VALID_AUTOTUNE_AGGREGATIONS:
        raise ValueError(
            "Unsupported autotune_aggregation %r. Expected one of: mean, median." % aggregation
        )
    strategy = str(shared_options.get("validation_window_strategy", "forecast_horizon")).lower()
    autotune_parallel = shared_options.get("autotune_parallel")
    if autotune_parallel is None:
        autotune_parallel = parallel
    autotune_num_samples = shared_options.get("autotune_num_samples")
    if autotune_num_samples is not None:
        autotune_num_samples = int(autotune_num_samples)
    return metric, aggregation, strategy, bool(autotune_parallel), autotune_num_samples



def evaluate_hyper_nll(hyper, train_minus_val, val, get_predictions_fn):
    eval_hyper = strip_autotune_kwargs(hyper)
    eval_hyper["compute_nll"] = True
    try:
        return get_predictions_fn(train_minus_val, val, **eval_hyper, num_samples=0)["NLL/D"]
    except NotImplementedError as exc:
        raise NotImplementedError(
            "exact_nll_autotune requires teacher-forced token-level scoring on the validation window. "
            "The selected provider/model path does not expose that capability. Use autotune_mode='validation_metric' instead."
        ) from exc



def evaluate_hyper_validation_metric(
    hyper,
    train_minus_val,
    val,
    get_predictions_fn,
    metric,
    autotune_num_samples,
    aggregation,
):
    if autotune_num_samples is None or autotune_num_samples <= 0:
        raise ValueError(
            "validation_metric autotune requires autotune_num_samples > 0 because it scores numeric forecasts, not token logprobs."
        )

    eval_hyper = strip_autotune_kwargs(hyper)
    eval_hyper["compute_nll"] = False
    prediction_dict = get_predictions_fn(
        train_minus_val,
        val,
        **eval_hyper,
        num_samples=autotune_num_samples,
    )
    if prediction_dict.get('info', {}).get('HasShortPrediction'):
        return float('inf')
    point_forecast = _extract_point_forecast(prediction_dict, aggregation)
    if point_forecast is None:
        return float("inf")
    pred_arr = _to_2d_array(point_forecast)
    target_arr = _to_2d_array(val)
    if pred_arr.shape != target_arr.shape or np.isnan(pred_arr).any():
        return float("inf")
    return compute_point_forecast_metric(target_arr, pred_arr, metric=metric)



def _search_hypers(hypers, score_fn, verbose=False, parallel=True):
    best_score = float("inf")
    best_hyper = None
    scores = []

    if not parallel:
        for hyper in tqdm(hypers, desc="Hyperparameter search"):
            _, score = score_fn(hyper)
            scores.append(score)
            if score < best_score:
                best_score = score
                best_hyper = hyper
            if verbose:
                print("Hyper: %r\n\t Score: %3f" % (hyper, score))
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(score_fn, hyper) for hyper in hypers]
            for future in tqdm(as_completed(futures), total=len(hypers), desc="Hyperparameter search"):
                hyper, score = future.result()
                scores.append(score)
                if score < best_score:
                    best_score = score
                    best_hyper = hyper
                if verbose:
                    print("Hyper: %r\n\t Score: %3f" % (hyper, score))

    return best_hyper, best_score, scores



def get_autotuned_predictions_data(
    train,
    test,
    hypers,
    num_samples,
    get_predictions_fn,
    verbose=False,
    parallel=True,
    n_train=None,
    n_val=None,
    **autotune_overrides,
):
    if isinstance(hypers, dict):
        hypers = list(grid_iter(hypers))
    else:
        assert isinstance(hypers, list), "hypers must be a list or dict"
    if not hypers:
        raise ValueError("At least one hyperparameter candidate is required.")
    if not isinstance(train, list):
        train = [train]
        test = [test]
    if n_val is None:
        n_val = len(train)

    forecast_hypers, shared_options = _resolve_shared_autotune_options(hypers, autotune_overrides)
    mode, supported_modes = _resolve_autotune_mode(forecast_hypers, shared_options)
    metric, aggregation, strategy, autotune_parallel, autotune_num_samples = _resolve_metric_options(mode, shared_options, parallel)
    forecast_hypers = _sanitize_candidates(forecast_hypers, shared_options.get("autotune_max_candidates"))
    if not forecast_hypers:
        raise ValueError("No hyperparameter candidates remain after applying autotune_max_candidates.")

    selected_hyper = forecast_hypers[0]
    best_score = None
    val_length = None
    if mode == "disabled":
        if len(forecast_hypers) > 1:
            raise ValueError(
                "autotune_mode='disabled' cannot choose among multiple hyperparameter candidates. "
                "Provide exactly one candidate or switch to autotune_mode='validation_metric' or 'nll'."
            )
    elif len(forecast_hypers) > 1:
        val_length = len(test[0])
        train_minus_val, val, n_val = make_validation_dataset(train, n_val=n_val, val_length=val_length, strategy=strategy)
        valid_pairs = [
            (train_series, val_series)
            for train_series, val_series in zip(train_minus_val, val)
            if len(val_series) == val_length
        ]
        if not valid_pairs:
            raise ValueError(
                "No validation series have the required horizon length %d for autotune_mode=%r."
                % (val_length, mode)
            )
        train_minus_val, val = zip(*valid_pairs)
        train_minus_val = list(train_minus_val)
        val = list(val)
        if len(train_minus_val) <= int(0.9 * n_val):
            raise ValueError(
                "Removed too many validation series. Only %d out of %d series have length >= %d."
                % (len(train_minus_val), n_val, val_length)
            )

        if mode == "nll":
            def score_fn(hyper):
                try:
                    return hyper, evaluate_hyper_nll(hyper, train_minus_val, val, get_predictions_fn)
                except ValueError:
                    return hyper, float("inf")
        elif mode == "validation_metric":
            def score_fn(hyper):
                try:
                    return hyper, evaluate_hyper_validation_metric(
                        hyper,
                        train_minus_val,
                        val,
                        get_predictions_fn,
                        metric,
                        autotune_num_samples,
                        aggregation,
                    )
                except ValueError:
                    return hyper, float("inf")
        else:
            raise ValueError("Unexpected autotune mode %r" % mode)

        selected_hyper, best_score, _ = _search_hypers(
            forecast_hypers,
            score_fn,
            verbose=verbose,
            parallel=autotune_parallel,
        )
        if selected_hyper is None:
            raise ValueError(
                "Autotune could not select a valid hyperparameter candidate because every candidate failed "
                "during validation. This often means the model returned only short or invalid validation forecasts."
            )
    elif mode == "validation_metric" and autotune_num_samples is not None and autotune_num_samples <= 0:
        raise ValueError(
            "validation_metric autotune requires autotune_num_samples > 0 because it evaluates forecast quality directly."
        )

    best_hyper = dict(selected_hyper)
    final_hyper = strip_autotune_kwargs(best_hyper)
    out = get_predictions_fn(train, test, **final_hyper, num_samples=num_samples, n_train=n_train)
    out["best_hyper"] = convert_to_dict(best_hyper)
    out["autotune"] = {
        "mode": mode,
        "supported_modes": list(supported_modes),
        "metric": None if mode == "nll" else metric,
        "aggregation": None if mode == "nll" else aggregation,
        "validation_window_strategy": strategy,
        "validation_window_length": val_length,
        "autotune_num_samples": autotune_num_samples,
        "autotune_parallel": autotune_parallel,
        "num_candidates_evaluated": len(forecast_hypers),
        "best_score": None if best_score is None or not np.isfinite(best_score) else float(best_score),
    }
    if mode == "nll" and best_score is not None and np.isfinite(best_score):
        out["validation_NLL/D"] = float(best_score)
    if mode == "validation_metric" and best_score is not None and np.isfinite(best_score):
        out["validation_metric"] = metric
        out["validation_metric_score"] = float(best_score)
    return out



def convert_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_dict(elem) for elem in obj]
    if is_dataclass(obj):
        return convert_to_dict(obj.__dict__)
    return obj
