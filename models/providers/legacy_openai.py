from configs.runtime import get_provider_config
from models.providers.base import BaseProvider, GenerationSample, ScoreResult

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


class LegacyOpenAIProvider(BaseProvider):
    name = "legacy_openai"

    def _client(self):
        if OpenAI is None:
            raise ImportError(
                "The optional `openai` package is not installed. "
                "Install it if you need the legacy OpenAI provider."
            )
        config = get_provider_config("legacy_openai")
        if not config.get("api_key"):
            raise ValueError(
                "Missing OPENAI_API_KEY for the legacy OpenAI provider. "
                "Configure it only if you explicitly want the legacy route."
            )
        kwargs = {"api_key": config.get("api_key")}
        if config.get("base_url"):
            kwargs["base_url"] = config.get("base_url")
        return OpenAI(**kwargs)

    def generate(self, spec, request):
        client = self._client()
        generation_defaults = dict(spec.extra_generation_defaults)
        generation_defaults.update(spec.provider_options)
        generation_defaults.update(request.extra_body)
        generation_defaults.pop("system_prompt", None)

        if spec.mode == "chat":
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            response = client.chat.completions.create(
                model=spec.api_model_name,
                messages=messages,
                n=request.num_samples,
                temperature=request.temperature,
                max_tokens=request.max_new_tokens,
                top_p=request.top_p,
                stop=request.stop,
                **generation_defaults,
            )
            samples = []
            for choice in response.choices:
                content = choice.message.content if choice.message is not None else ""
                samples.append(
                    GenerationSample(
                        text=content or "",
                        reasoning_content=getattr(choice.message, "reasoning_content", None),
                        logprobs=getattr(choice, "logprobs", None),
                        raw=choice,
                    )
                )
            return samples

        response = client.completions.create(
            model=spec.api_model_name,
            prompt=request.prompt,
            n=request.num_samples,
            temperature=request.temperature,
            max_tokens=request.max_new_tokens,
            top_p=request.top_p,
            stop=request.stop,
            **generation_defaults,
        )
        return [GenerationSample(text=choice.text or "", raw=choice) for choice in response.choices]

    def score(self, spec, request):
        if spec.mode != "text_completion":
            raise NotImplementedError(
                "Model '%s' does not expose completion-style teacher-forced scoring in the "
                "legacy provider because its mode is '%s'." % (spec.logical_model_name, spec.mode)
            )
        client = self._client()
        response = client.completions.create(
            model=spec.api_model_name,
            prompt=request.prompt,
            logprobs=request.top_logprobs,
            max_tokens=0,
            echo=True,
            temperature=request.temperature,
        )
        choice = response.choices[0]
        logprobs = choice.logprobs
        return ScoreResult(
            tokens=list(logprobs.tokens),
            token_logprobs=list(logprobs.token_logprobs),
            top_logprobs=list(logprobs.top_logprobs),
            raw=response,
        )
