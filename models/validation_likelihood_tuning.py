from dataclasses import is_dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm.auto import tqdm

from models.utils import grid_iter



def make_validation_dataset(train, n_val, val_length):
    assert isinstance(train, list), "Train should be a list of series"

    train_minus_val_list, val_list = [], []
    if n_val is None:
        n_val = len(train)
    for train_series in train[:n_val]:
        train_len = max(len(train_series) - val_length, 1)
        train_minus_val, val = train_series[:train_len], train_series[train_len:]
        print("Train length: %d, Val length: %d" % (len(train_minus_val), len(val)))
        train_minus_val_list.append(train_minus_val)
        val_list.append(val)

    return train_minus_val_list, val_list, n_val



def evaluate_hyper(hyper, train_minus_val, val, get_predictions_fn):
    assert isinstance(train_minus_val, list) and isinstance(val, list)
    try:
        return get_predictions_fn(train_minus_val, val, **hyper, num_samples=0)["NLL/D"]
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Autotune-by-NLL is disabled for the current model/provider path. "
            "Grid search ranks hyperparameters by validation NLL/D, so sampling-only providers such as the default "
            "Qwen-native route must use a single fixed hyperparameter dictionary instead of a search grid."
        ) from exc



def _requires_validation_nll(hyper):
    return hyper.get("compute_nll", True)



def get_autotuned_predictions_data(train, test, hypers, num_samples, get_predictions_fn, verbose=False, parallel=True, n_train=None, n_val=None):
    if isinstance(hypers, dict):
        hypers = list(grid_iter(hypers))
    else:
        assert isinstance(hypers, list), "hypers must be a list or dict"
    if not isinstance(train, list):
        train = [train]
        test = [test]
    if n_val is None:
        n_val = len(train)

    if len(hypers) > 1 and any(not _requires_validation_nll(hyper) for hyper in hypers):
        raise NotImplementedError(
            "Hyperparameter search requires compute_nll=True because the objective is validation NLL/D. "
            "For sampling-only models such as default Qwen specs, pass a single fixed hyperparameter dictionary."
        )

    if len(hypers) > 1:
        val_length = min(len(test[0]), int(np.mean([len(series) for series in train]) / 2))
        train_minus_val, val, n_val = make_validation_dataset(train, n_val=n_val, val_length=val_length)
        train_minus_val, val = zip(
            *[
                (train_series, val_series)
                for train_series, val_series in zip(train_minus_val, val)
                if len(val_series) == val_length
            ]
        )
        train_minus_val = list(train_minus_val)
        val = list(val)
        if len(train_minus_val) <= int(0.9 * n_val):
            raise ValueError(
                "Removed too many validation series. Only %d out of %d series have length >= %d."
                % (len(train_minus_val), n_val, val_length)
            )
        val_nlls = []

        def eval_hyper(hyper):
            try:
                return hyper, evaluate_hyper(hyper, train_minus_val, val, get_predictions_fn)
            except ValueError:
                return hyper, float("inf")

        best_val_nll = float("inf")
        best_hyper = None
        if not parallel:
            for hyper in tqdm(hypers, desc="Hyperparameter search"):
                _, val_nll = eval_hyper(hyper)
                val_nlls.append(val_nll)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_hyper = hyper
                if verbose:
                    print("Hyper: %r\n\t Val NLL: %3f" % (hyper, val_nll))
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(eval_hyper, hyper) for hyper in hypers]
                for future in tqdm(as_completed(futures), total=len(hypers), desc="Hyperparameter search"):
                    hyper, val_nll = future.result()
                    val_nlls.append(val_nll)
                    if val_nll < best_val_nll:
                        best_val_nll = val_nll
                        best_hyper = hyper
                    if verbose:
                        print("Hyper: %r\n\t Val NLL: %3f" % (hyper, val_nll))
    else:
        best_hyper = hypers[0]
        best_val_nll = float("inf")

    out = get_predictions_fn(train, test, **best_hyper, num_samples=num_samples, n_train=n_train)
    out["best_hyper"] = convert_to_dict(best_hyper)
    if best_val_nll < float("inf"):
        out["validation_NLL/D"] = best_val_nll
    return out



def convert_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_dict(elem) for elem in obj]
    if is_dataclass(obj):
        return convert_to_dict(obj.__dict__)
    return obj
