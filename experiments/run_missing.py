import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from data.small_context import get_datasets
from data.serialize import SerializerSettings
from experiments.model_defaults import DEFAULT_REMOTE_MODEL, is_sampling_only_model
from models.darts import get_NBEATS_predictions_data, get_NHITS_predictions_data, get_TCN_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data



def nan_corruption(x: pd.Series, p=0.0):
    x = x.copy()
    x.iloc[np.random.choice(len(x), int(p * len(x)), replace=False)] = np.nan
    return x



def interp_nans(x: pd.Series):
    x = x.copy()
    nans = np.isnan(x)
    f = lambda z: z.values.nonzero()[0]
    x[nans] = np.interp(f(nans), f(~nans), x[~nans])
    return x


input_dir = 'outputs/darts'
output_dir = 'outputs/missing'
os.makedirs(output_dir, exist_ok=True)

datasets = get_datasets()
all_output = {}
for dsname, data in datasets.items():
    train, test = data
    if not os.path.exists(f'{input_dir}/{dsname}.pkl'):
        continue
    with open(f'{input_dir}/{dsname}.pkl', 'rb') as handle:
        in_dict = pickle.load(handle)

    prediction_fns = [get_arima_predictions_data, get_TCN_predictions_data, get_NHITS_predictions_data, get_llmtime_predictions_data]
    model_names = ['arima', 'TCN', 'N-HiTS', DEFAULT_REMOTE_MODEL]

    output_dict = defaultdict(list)
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        output_dict['p'].append(p)
        corrupted_train = nan_corruption(train, p=p)
        interpolated = interp_nans(corrupted_train)
        for predict, model in zip(prediction_fns, model_names):
            best_hyper = in_dict[model]['best_hyper']
            if 'settings' in best_hyper and isinstance(best_hyper['settings'], dict):
                best_hyper['settings'] = SerializerSettings(**best_hyper['settings'])
            if is_sampling_only_model(model):
                output_dict[model].append(np.nan)
                output_dict[model + '-Nan'].append(np.nan)
                output_dict[model + '_capability'] = 'nll_unsupported'
                continue
            if model == DEFAULT_REMOTE_MODEL:
                output_dict[model + '-Nan'].append(predict(corrupted_train.copy(), test.copy(), **best_hyper, num_samples=0)['NLL/D'])
            output_dict[model].append(predict(interpolated.copy(), test.copy(), **best_hyper, num_samples=0)['NLL/D'])
    all_output[dsname] = output_dict

with open(f'{output_dir}/missing.pkl', 'wb') as handle:
    pickle.dump(all_output, handle)
