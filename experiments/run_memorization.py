import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import os
import pickle

import numpy as np

from data.small_context import get_memorization_datasets
from models.darts import get_NBEATS_predictions_data, get_NHITS_predictions_data, get_TCN_predictions_data
from models.darts import get_arima_predictions_data
from models.gaussian_process import get_gp_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.utils import grid_iter
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from experiments.model_defaults import DEFAULT_LLMTIME_HYPERS, DEFAULT_REMOTE_MODEL, is_remote_model, is_sampling_only_model
from experiments.runtime_filters import (
    get_local_num_samples,
    get_output_dir,
    get_remote_num_samples,
    select_named_items,
    select_names,
)


gp_hypers = dict(lr=[5e-3, 1e-2, 5e-2, 1e-1])
arima_hypers = dict(p=[12, 20, 30], d=[1, 2], q=[0, 1, 2])
TCN_hypers = dict(in_len=[10, 100, 400], out_len=[1], kernel_size=[3, 5], num_filters=[1, 3], likelihood=["laplace", "gaussian"])
NHITS_hypers = dict(in_len=[10, 100, 400], out_len=[1], layer_widths=[64, 16], num_layers=[1, 2], likelihood=["laplace", "gaussian"])
NBEATS_hypers = dict(in_len=[10, 100, 400], out_len=[1], layer_widths=[64, 16], num_layers=[1, 2], likelihood=["laplace", "gaussian"])

model_hypers = {
    'gp': gp_hypers,
    'arima': arima_hypers,
    'TCN': TCN_hypers,
    'N-BEATS': NBEATS_hypers,
    'N-HiTS': NHITS_hypers,
    DEFAULT_REMOTE_MODEL: dict(DEFAULT_LLMTIME_HYPERS),
}

model_predict_fns = {
    'gp': get_gp_predictions_data,
    'arima': get_arima_predictions_data,
    'TCN': get_TCN_predictions_data,
    'N-BEATS': get_NBEATS_predictions_data,
    'N-HiTS': get_NHITS_predictions_data,
    DEFAULT_REMOTE_MODEL: get_llmtime_predictions_data,
}

output_dir = get_output_dir('outputs/memorization')
os.makedirs(output_dir, exist_ok=True)

datasets = select_named_items(get_memorization_datasets(predict_steps=30))
models_to_run = select_names([DEFAULT_REMOTE_MODEL, 'gp', 'arima', 'N-HiTS'], filter_env='LLMTIME_MODEL_FILTER')
for dsname, data in datasets:
    train, test = data
    output_path = f'{output_dir}/{dsname}.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        with open(output_path, 'rb') as handle:
            out_dict = pickle.load(handle)
    else:
        out_dict = {}

    for model in models_to_run:
        if model in out_dict and not is_remote_model(model):
            if out_dict[model]['samples'] is not None:
                print(f'Skipping {dsname} {model}')
                continue
            print('Using best hyper...')
            hypers = [out_dict[model]['best_hyper']]
        elif model in out_dict and is_remote_model(model):
            print(f'Skipping {dsname} {model}')
            continue
        else:
            print(f'Starting {dsname} {model}')
            hypers = list(grid_iter(model_hypers[model]))
        parallel = is_remote_model(model)
        num_samples = get_remote_num_samples(20) if is_remote_model(model) else get_local_num_samples(100)
        try:
            preds = get_autotuned_predictions_data(
                train,
                test,
                hypers,
                num_samples,
                model_predict_fns[model],
                verbose=False,
                parallel=parallel,
            )
            autotune_mode = preds.get('autotune', {}).get('mode')
            if is_sampling_only_model(model) or autotune_mode == 'validation_metric' or preds.get('NLL/D', np.inf) < np.inf:
                out_dict[model] = preds
            else:
                print(f'Failed {dsname} {model}')
        except Exception as exc:
            print(f'Failed {dsname} {model}')
            print(exc)
            continue
        with open(output_path, 'wb') as handle:
            pickle.dump(out_dict, handle)

    print(f'Finished {dsname}')
