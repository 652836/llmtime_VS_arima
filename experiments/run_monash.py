import os
import pickle

import numpy as np

from data.monash import get_datasets
from models.llmtime import get_llmtime_predictions_data
from models.utils import grid_iter
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from experiments.model_defaults import DEFAULT_REMOTE_MODEL, QWEN_LLMTIME_HYPERS, is_remote_model


model_hypers = {
    DEFAULT_REMOTE_MODEL: dict(QWEN_LLMTIME_HYPERS),
}

model_predict_fns = {
    DEFAULT_REMOTE_MODEL: get_llmtime_predictions_data,
}

output_dir = 'outputs/monash'
os.makedirs(output_dir, exist_ok=True)

datasets_to_run = [
    'covid_deaths', 'solar_weekly', 'tourism_monthly', 'australian_electricity_demand', 'pedestrian_counts',
    'traffic_hourly', 'hospital', 'fred_md', 'tourism_yearly', 'tourism_quarterly', 'us_births',
    'nn5_weekly', 'solar_10_minutes', 'traffic_weekly', 'saugeenday', 'cif_2016',
]

max_history_len = 500
datasets = get_datasets()
for dsname in datasets_to_run:
    print(f'Starting {dsname}')
    train, test = datasets[dsname]
    train = [series[-max_history_len:] for series in train]
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl', 'rb') as handle:
            out_dict = pickle.load(handle)
    else:
        out_dict = {}

    for model in [DEFAULT_REMOTE_MODEL]:
        if model in out_dict:
            print(f'Skipping {dsname} {model}')
            continue
        print(f'Starting {dsname} {model}')
        hypers = list(grid_iter(model_hypers[model]))
        parallel = True if is_remote_model(model) else False
        num_samples = 5
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
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
        with open(f'{output_dir}/{dsname}.pkl', 'wb') as handle:
            pickle.dump(out_dict, handle)
    print(f'Finished {dsname}')
