from http import HTTPStatus

from configs.runtime import get_provider_config
from models.providers.base import BaseProvider, GenerationSample, ProviderCapabilityError

try:  # pragma: no cover - optional dependency
    import dashscope
    from dashscope import Generation
except ImportError:  # pragma: no cover - optional dependency
    dashscope = None
    Generation = None


class QwenProvider(BaseProvider):
    name = "qwen"

    def _ensure_sdk(self):
        if Generation is None:
            raise ImportError(
                "DashScope SDK is not installed. Install it with `pip install dashscope` "
                "to use the Qwen-native provider."
            )

    def _provider_config(self):
        config = get_provider_config("qwen")
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError(
                "Missing DASHSCOPE_API_KEY for the Qwen provider. "
                "Set it in the environment or configs/runtime.local.json."
            )
        if dashscope is not None and config.get("base_url"):
            if hasattr(dashscope, "base_http_api_url"):
                dashscope.base_http_api_url = config.get("base_url")
        return config

    @staticmethod
    def _safe_get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return getattr(obj, key)
        except (AttributeError, KeyError):
            pass
        try:
            return obj[key]
        except Exception:
            return default

    def _messages(self, request):
        system_prompt = request.system_prompt
        if system_prompt:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt},
            ]
        return [{"role": "user", "content": request.prompt}]

    def _extract_choice(self, response):
        output = self._safe_get(response, "output")
        choices = self._safe_get(output, "choices", []) or []
        text = self._safe_get(output, "text", "")
        if choices:
            choice = choices[0]
            message = self._safe_get(choice, "message", {})
            content = self._safe_get(message, "content", "")
            reasoning_content = self._safe_get(message, "reasoning_content")
            logprobs = self._safe_get(choice, "logprobs")
            return {
                "text": content or "",
                "reasoning_content": reasoning_content,
                "logprobs": logprobs,
            }
        return {"text": text or "", "reasoning_content": None, "logprobs": None}

    def generate(self, spec, request):
        self._ensure_sdk()
        provider_config = self._provider_config()
        messages = self._messages(request)
        generation_defaults = dict(spec.extra_generation_defaults)
        generation_defaults.update(spec.provider_options)
        generation_defaults.update(request.extra_body)

        result_format = generation_defaults.pop(
            "result_format", provider_config.get("result_format", "message")
        )
        enable_thinking = generation_defaults.pop(
            "enable_thinking", provider_config.get("enable_thinking", False)
        )
        thinking_budget = generation_defaults.pop(
            "thinking_budget", provider_config.get("thinking_budget")
        )
        generation_defaults.pop("system_prompt", None)

        samples = []
        for _ in range(request.num_samples):
            call_kwargs = {
                "api_key": provider_config.get("api_key"),
                "model": spec.api_model_name,
                "messages": messages,
                "result_format": result_format,
                "temperature": request.temperature,
                "max_tokens": request.max_new_tokens,
                "stream": False,
                "enable_thinking": enable_thinking,
            }
            if request.top_p is not None:
                call_kwargs["top_p"] = request.top_p
            if request.stop is not None:
                call_kwargs["stop"] = request.stop
            if thinking_budget is not None:
                call_kwargs["thinking_budget"] = thinking_budget
            call_kwargs.update(generation_defaults)

            response = Generation.call(**call_kwargs)
            status_code = getattr(response, "status_code", None)
            if status_code != HTTPStatus.OK:
                raise RuntimeError(
                    "DashScope request failed for model '%s' with status=%s code=%s message=%s"
                    % (
                        spec.api_model_name,
                        status_code,
                        getattr(response, "code", None),
                        getattr(response, "message", None),
                    )
                )
            choice = self._extract_choice(response)
            usage = getattr(response, "usage", None) or {}
            samples.append(
                GenerationSample(
                    text=choice["text"],
                    reasoning_content=choice["reasoning_content"],
                    logprobs=choice["logprobs"],
                    usage=usage,
                    raw=response,
                )
            )
        return samples

    def score(self, spec, request):
        raise ProviderCapabilityError(
            "Model '%s' uses the Qwen-native DashScope provider, but LLMTime NLL/autotune "
            "requires teacher-forced token scores over the supplied prompt+target sequence. "
            "DashScope's native logprobs only cover generated output tokens, so `supports_nll` "
            "and `supports_autotune` stay disabled for the main Qwen path."
            % spec.logical_model_name
        )
