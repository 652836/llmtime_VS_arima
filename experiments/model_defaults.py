from data.serialize import SerializerSettings
from configs.runtime import get_default_model
from models.model_registry import get_model_capabilities


DEFAULT_REMOTE_MODEL = get_default_model()

QWEN_LLMTIME_HYPERS = {
    "model": DEFAULT_REMOTE_MODEL,
    "temp": 0.7,
    "alpha": 0.95,
    "beta": 0.3,
    "basic": False,
    "compute_nll": False,
    "settings": SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
}

QWEN_PROMPTCAST_HYPERS = {
    "model": DEFAULT_REMOTE_MODEL,
    "temp": 0.7,
    "compute_nll": False,
    "settings": SerializerSettings(
        base=10,
        prec=0,
        signed=True,
        time_sep=", ",
        bit_sep="",
        plus_sign="",
        minus_sign="-",
        half_bin_correction=False,
        decimal_point="",
    ),
}



def is_sampling_only_model(model):
    return not get_model_capabilities(model).supports_nll



def is_remote_model(model):
    return model not in {"gp", "arima", "TCN", "N-BEATS", "N-HiTS"}
