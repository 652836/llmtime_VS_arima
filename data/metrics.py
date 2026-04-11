import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap

from .serialize import SerializerSettings
from models.llms import get_nll_fn



def quantile_loss(target, pred, q):
    q_pred = jnp.quantile(pred, q, axis=0)
    return 2 * jnp.sum(jnp.abs((q_pred - target) * ((target <= q_pred) * 1.0 - q)))



def calculate_crps(target, pred, num_quantiles=20):
    quantiles = jnp.linspace(0, 1.0, num_quantiles + 1)[1:]
    vec_quantile_loss = vmap(lambda q: quantile_loss(target, pred, q))
    crps = jnp.sum(vec_quantile_loss(quantiles))
    crps = crps / (jnp.sum(np.abs(target)) * len(quantiles))
    return crps



def nll(input_arr, target_arr, model, settings: SerializerSettings, transform, count_seps=True, prompt=None, temp=1):
    nll_fn = get_nll_fn(model)
    if nll_fn is None:
        raise NotImplementedError(
            "Model '%s' does not expose generic NLL scoring in the current provider setup."
            % model
        )
    if prompt:
        raise NotImplementedError(
            "The generic data.metrics.nll wrapper no longer supports an extra free-form prompt. "
            "Use the model-specific forecasting pipeline if you need prompt engineering."
        )
    return nll_fn(
        input_arr=input_arr,
        target_arr=target_arr,
        settings=settings,
        transform=transform,
        count_seps=count_seps,
        temp=temp,
    )


class Evaluator:
    def __init__(self):
        self.non_numerical_cols = [
            "serialized_history",
            "serialized_target",
            "serialized_prediction",
            "history_len",
            "num_channels",
            "example_num",
            "sample_num",
        ]

    def evaluate_df(self, gt_df, pred_df):
        cols = [column for column in gt_df.columns if column not in self.non_numerical_cols]
        num_channels = gt_df["num_channels"].iloc[0]
        history_len = gt_df["history_len"].iloc[0]
        gt_vals = gt_df[cols].to_numpy().reshape(len(gt_df), -1, num_channels)
        gt_vals = gt_vals[:, history_len:, :]

        cols = [column for column in pred_df.columns if column not in self.non_numerical_cols]
        num_channels = pred_df["num_channels"].iloc[0]
        pred_df = pred_df[cols + ["example_num"]]

        all_pred_vals = []
        for example_num in sorted(pred_df["example_num"].unique()):
            pred_vals = pred_df[pred_df["example_num"] == example_num][cols].to_numpy()
            pred_vals = pred_vals.reshape(pred_vals.shape[0], -1, num_channels)
            all_pred_vals.append(pred_vals)

        pred_vals = np.stack(all_pred_vals, axis=1)
        assert gt_vals.shape == pred_vals.shape[1:]

        diff = gt_vals[None] - pred_vals
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        crps = calculate_crps(gt_vals, pred_vals)

        return {"mse": mse, "mae": mae, "crps": crps}

    def evaluate(self, gt, pred):
        assert gt.shape == (pred.shape[0], pred.shape[2]), (
            "wrong shapes: gt.shape: %r pred.shape: %r" % (gt.shape, pred.shape)
        )
        diff = gt[:, None, :] - pred
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        std = np.std(gt, axis=1) + 1e-8
        normalized_diff = diff / std[:, None, None]
        nmse = np.mean(normalized_diff**2)
        nmae = np.mean(np.abs(normalized_diff))

        return {"nmse": nmse, "nmae": nmae, "mse": mse, "mae": mae}
