import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import os
import pickle

import numpy as np

from data.monash import get_datasets
from models.llmtime import get_llmtime_predictions_data
from models.utils import grid_iter
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from experiments.model_defaults import DEFAULT_LLMTIME_HYPERS, DEFAULT_REMOTE_MODEL, is_remote_model
from experiments.runtime_filters import get_output_dir, get_remote_num_samples, select_names


model_hypers = {
    DEFAULT_REMOTE_MODEL: dict(DEFAULT_LLMTIME_HYPERS),
}

model_predict_fns = {
    DEFAULT_REMOTE_MODEL: get_llmtime_predictions_data,
}

output_dir = get_output_dir('outputs/monash')
os.makedirs(output_dir, exist_ok=True)

datasets_to_run = [
    'covid_deaths', 'solar_weekly', 'tourism_monthly', 'australian_electricity_demand', 'pedestrian_counts',
    'traffic_hourly', 'hospital', 'fred_md', 'tourism_yearly', 'tourism_quarterly', 'us_births',
    'nn5_weekly', 'solar_10_minutes', 'traffic_weekly', 'saugeenday', 'cif_2016',
]

datasets_to_run = select_names(datasets_to_run, filter_env='LLMTIME_DATASET_FILTER', max_env='LLMTIME_MAX_DATASETS')
models_to_run = select_names([DEFAULT_REMOTE_MODEL], filter_env='LLMTIME_MODEL_FILTER')
max_history_len = 500
datasets = get_datasets()
for dsname in datasets_to_run:
    print(f'Starting {dsname}')
    train, test = datasets[dsname]
    train = [series[-max_history_len:] for series in train]
    output_path = f'{output_dir}/{dsname}.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        with open(output_path, 'rb') as handle:
            out_dict = pickle.load(handle)
    else:
        out_dict = {}

    for model in models_to_run:
        if model in out_dict:
            print(f'Skipping {dsname} {model}')
            continue
        print(f'Starting {dsname} {model}')
        hypers = list(grid_iter(model_hypers[model]))
        parallel = is_remote_model(model)
        num_samples = get_remote_num_samples(5)
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
            medians = preds['median']
            targets = np.array(test)
            maes = np.mean(np.abs(medians - targets), axis=1)
            preds['maes'] = maes
            preds['mae'] = np.mean(maes)
            out_dict[model] = preds
        except Exception as exc:
            print(f'Failed {dsname} {model}')
            print(exc)
            continue
        with open(output_path, 'wb') as handle:
            pickle.dump(out_dict, handle)
    print(f'Finished {dsname}')
