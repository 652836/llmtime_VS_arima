# Qwen-Native Refactor Notes

## Goal

This refactor keeps the original LLMTime / PromptCast / baseline experiment structure, but replaces the remote-model layer with a provider-oriented architecture that treats Qwen / DashScope as the primary path instead of patching old OpenAI code.

## New Layering

### 1. Config layer

- `configs/runtime.py`
- `configs/runtime.example.json`

Responsibilities:
- load provider credentials from environment variables such as `DASHSCOPE_API_KEY`
- optionally load local overrides from `configs/runtime.local.json` or `LLMTIME_CONFIG`
- expose one shared runtime view for notebooks and scripts

### 2. Model spec / registry layer

- `models/model_registry.py`

Responsibilities:
- separate logical model names from provider-facing API model names
- store tokenizer aliases independently from model names
- store context lengths independently from tokenizer lookup
- declare capabilities such as `supports_nll` and `supports_autotune`

### 3. Provider layer

- `models/providers/base.py`
- `models/providers/qwen.py`
- `models/providers/legacy_openai.py`
- `models/providers/local_hf.py`

Responsibilities:
- implement sampling and optional scoring behind one interface
- keep Qwen-native message wrapping, thinking control, and response parsing out of forecasting code
- reserve legacy OpenAI and LocalHF as optional / future providers instead of default dependencies

### 4. Forecasting core

- `models/llmtime.py`
- `models/promptcast.py`
- `models/llms.py`
- `models/gpt.py`

Responsibilities:
- scaling
- serialization / deserialization
- truncation against model context windows
- sampling orchestration
- teacher-forced NLL only when the provider actually supports it

### 5. Experiment layer

- `experiments/model_defaults.py`
- `experiments/run_*.py`
- notebooks

Responsibilities:
- choose datasets
- choose logical models
- choose baselines
- choose hyperparameters
- store outputs

## Why split logical model / API model / tokenizer / context

The old repository tightly coupled:
- model naming
- API backend
- tokenizer lookup
- context window
- completion vs chat behavior

That fails as soon as a provider uses different naming rules or exposes only chat-style APIs. Qwen-native integration especially breaks if code tries to feed Qwen names into `tiktoken.encoding_for_model()` or assumes OpenAI `Completion.create(..., echo=True, logprobs=...)` exists.

The new registry stores these concerns separately so higher-level experiment logic can stay stable even when providers differ.

## NLL / autotune status

### Original logic

Original LLMTime autotune ranks hyperparameters by validation `NLL/D`, which depends on teacher-forced token-level scoring over the concatenated serialized prompt + target sequence.

### Current Qwen-native status

For the default Qwen path:
- `supports_sampling = True`
- `supports_logprobs = False` on the default stable route
- `supports_nll = False`
- `supports_autotune = False`

Reason:
- DashScope native sampling works and can optionally expose output-token logprobs on some snapshot routes.
- LLMTime `NLL/D` needs teacher-forced scoring analogous to legacy completion `echo=True` behavior.
- That capability is not exposed on the main Qwen-native path used here, so the refactor leaves NLL/autotune explicitly disabled instead of silently changing the objective.

### Consequence

- Qwen default experiments should use a single fixed hyperparameter dictionary with `compute_nll=False`.
- Validation grid search remains available only for providers/models that really implement teacher-forced scoring.
- `PromptCast` is currently sampling-only in the refactor and raises a clear error if `compute_nll=True` is requested.

## Default model choices

Default mainline model:
- logical name: `qwen-plus`
- provider: `qwen`
- API model: `qwen-plus`
- tokenizer alias: `cl100k_base`
- context length: `997952`
- thinking: disabled by default for numeric forecasting stability

Optional registry entries also exist for:
- `qwen-plus-latest`
- `qwen-plus-snapshot-logprobs`
- `qwen3.6-plus`
- legacy OpenAI comparison models
- `local-hf-placeholder`

## Minimal runtime setup

### Environment-only setup

Set:
- `DASHSCOPE_API_KEY`

Optional:
- `LLMTIME_DEFAULT_MODEL`
- `LLMTIME_QWEN_MODEL`
- `LLMTIME_QWEN_ENABLE_THINKING=false`
- `LLMTIME_QWEN_CONTEXT_LENGTH`

### Local config setup

Create `configs/runtime.local.json` based on `configs/runtime.example.json` if you want notebook/script runs to share the same model and provider overrides without hardcoding values in notebook cells.

## Recommended first runs

Recommended notebook:
- `qwen_quickstart.ipynb`

Recommended minimal script smoke test:
- a tiny inline call to `models.llmtime.get_llmtime_predictions_data(..., compute_nll=False)` with a short toy series

Recommended full experiment entry points after that:
- `experiments/run_darts.py`
- `experiments/run_synthetic.py`
- `experiments/run_monash.py`

## Legacy comparison support

If you want to add OpenAI back later:
- configure `OPENAI_API_KEY`
- choose a legacy logical model already registered in `models/model_registry.py`
- do not change `models/llmtime.py`, `models/promptcast.py`, or experiment orchestration code

## Original vs Qwen refactor

| Area | Original repo | Qwen-native refactor |
| --- | --- | --- |
| Provider | old OpenAI SDK assumptions | provider abstraction with Qwen default |
| API key | `OPENAI_API_KEY` in scripts/notebooks | `DASHSCOPE_API_KEY` via shared runtime config |
| Model naming | logic name coupled to API model name | logical name decoupled from API model name |
| Tokenizer/context | model-name-driven `tiktoken` lookup | registry alias + explicit context length |
| Sampling path | `Completion.create` / `ChatCompletion.create` branches | provider-specific `generate()` behind one interface |
| Scoring path | old completion logprob path | capability-gated teacher-forced scoring |
| Autotune | assumes validation NLL exists | disabled for Qwen default, explicit error |
| Notebook entry | provider details in cells | shared runtime + logical model defaults |
| Script entry | hardcoded OpenAI setup | shared experiment defaults with Qwen mainline |

## Verified smoke checks in the llmtime interpreter

Using `D:\anaconda_py_package\Anaconda_envs\envs\llmtime\python.exe`:
- compiled `configs`, `models`, `data`, and `experiments`
- installed `dashscope`
- ran a tiny live `LLMTime` Qwen call with `compute_nll=False`
- ran a tiny live `PromptCast` Qwen sampling call with `compute_nll=False`

These checks validate that the refactored environment can import, compile, and execute the main Qwen-native sampling paths without falling back to old OpenAI SDK calls.
