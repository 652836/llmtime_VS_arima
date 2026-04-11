from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelCapabilities:
    supports_sampling: bool = True
    supports_logprobs: bool = False
    supports_nll: bool = False
    supports_autotune: bool = False
    supports_reasoning: bool = False
    supports_chat: bool = True
    supports_text_completion_like_mode: bool = False
    supports_logit_bias: bool = False


@dataclass(frozen=True)
class ModelSpec:
    logical_model_name: str
    provider: str
    api_model_name: str
    tokenizer_name_or_alias: Optional[str] = None
    context_length: Optional[int] = None
    mode: str = "chat"
    description: str = ""
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    extra_generation_defaults: Dict[str, Any] = field(default_factory=dict)
    provider_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float
    num_samples: int = 1
    top_p: Optional[float] = None
    stop: Any = None
    system_prompt: Optional[str] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationSample:
    text: str
    reasoning_content: Optional[str] = None
    logprobs: Any = None
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class ScoreRequest:
    prompt: str
    temperature: float = 1.0
    top_logprobs: int = 5


@dataclass
class ScoreResult:
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[dict]
    raw: Any = None


class ProviderCapabilityError(NotImplementedError):
    pass


class BaseProvider(ABC):
    name = "base"

    @abstractmethod
    def generate(self, spec, request):
        raise NotImplementedError

    def score(self, spec, request):
        raise ProviderCapabilityError(
            "Provider '%s' does not implement teacher-forced scoring for model '%s'."
            % (self.name, spec.logical_model_name)
        )

    def estimate_tokens(self, spec, text):
        from models.tokenization import estimate_token_count

        return estimate_token_count(text, tokenizer_name_or_alias=spec.tokenizer_name_or_alias)


def default_time_series_system_prompt():
    return (
        "You are a careful time-series forecasting assistant. "
        "Continue the numeric sequence exactly, without commentary, Markdown, "
        "or explanatory text."
    )
