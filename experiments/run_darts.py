import os
import pickle

from data.small_context import get_datasets
from models.darts import get_NBEATS_predictions_data, get_NHITS_predictions_data, get_TCN_predictions_data
from models.darts import get_arima_predictions_data
from models.gaussian_process import get_gp_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.utils import grid_iter
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from experiments.model_defaults import DEFAULT_REMOTE_MODEL, QWEN_LLMTIME_HYPERS, is_remote_model


gp_hypers = dict(lr=[5e-3, 1e-2, 5e-2, 1e-1])
arima_hypers = dict(p=[12, 20, 30], d=[1, 2], q=[0, 1, 2])
TCN_hypers = dict(in_len=[10, 100, 400], out_len=[1], kernel_size=[3, 5], num_filters=[1, 3], likelihood=["laplace", "gaussian"])
NHITS_hypers = dict(in_len=[10, 100, 400], out_len=[1], layer_widths=[64, 16], num_layers=[1, 2], likelihood=["laplace", "gaussian"])
NBEATS_hypers = dict(in_len=[10, 100, 400], out_len=[1], layer_widths=[64, 16], num_layers=[1, 2], likelihood=["laplace", "gaussian"])

model_hypers = {
    "gp": gp_hypers,
    "arima": arima_hypers,
    "TCN": TCN_hypers,
    "N-BEATS": NBEATS_hypers,
    "N-HiTS": NHITS_hypers,
    DEFAULT_REMOTE_MODEL: dict(QWEN_LLMTIME_HYPERS),
}

model_predict_fns = {
    "gp": get_gp_predictions_data,
    "arima": get_arima_predictions_data,
    "TCN": get_TCN_predictions_data,
    "N-BEATS": get_NBEATS_predictions_data,
    "N-HiTS": get_NHITS_predictions_data,
    DEFAULT_REMOTE_MODEL: get_llmtime_predictions_data,
}

output_dir = 'outputs/darts'
os.makedirs(output_dir, exist_ok=True)

datasets = get_datasets()
for dsname, data in datasets.items():
    train, test = data
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl', 'rb') as handle:
            out_dict = pickle.load(handle)
    else:
        out_dict = {}

    for model in [DEFAULT_REMOTE_MODEL, 'gp', 'arima', 'N-HiTS', 'TCN', 'N-BEATS']:
        if model in out_dict:
            print(f'Skipping {dsname} {model}')
            continue
        print(f'Starting {dsname} {model}')
        hypers = list(grid_iter(model_hypers[model]))
        parallel = True if is_remote_model(model) else False
        num_samples = 20 if is_remote_model(model) else 100
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            out_dict[model] = preds
        except Exception as exc:
            print(f'Failed {dsname} {model}')
            print(exc)
            continue
        with open(f'{output_dir}/{dsname}.pkl', 'wb') as handle:
            pickle.dump(out_dict, handle)

    print(f'Finished {dsname}')
