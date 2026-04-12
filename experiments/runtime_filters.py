import os


def _env_text(name, default=None):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name, default=None):
    value = _env_text(name)
    if value is None:
        return default
    return int(value)


def _csv_env(name):
    value = _env_text(name)
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_names(names, filter_env, max_env=None):
    selected = list(names)
    allowed = set(_csv_env(filter_env))
    if allowed:
        selected = [name for name in selected if name in allowed]
    if max_env:
        limit = _env_int(max_env)
        if limit is not None:
            selected = selected[:limit]
    return selected


def select_named_items(mapping, filter_env="LLMTIME_DATASET_FILTER", max_env="LLMTIME_MAX_DATASETS"):
    items = list(mapping.items())
    allowed = set(_csv_env(filter_env))
    if allowed:
        items = [(name, value) for name, value in items if name in allowed]
    limit = _env_int(max_env)
    if limit is not None:
        items = items[:limit]
    return items


def get_output_dir(default_dir):
    return _env_text("LLMTIME_OUTPUT_DIR", default_dir)


def get_remote_num_samples(default):
    return _env_int("LLMTIME_REMOTE_NUM_SAMPLES", default)


def get_local_num_samples(default):
    return _env_int("LLMTIME_LOCAL_NUM_SAMPLES", default)
