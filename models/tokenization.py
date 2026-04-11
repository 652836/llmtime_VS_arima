import math

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


def _resolve_tiktoken_encoding(tokenizer_name_or_alias):
    if tokenizer_name_or_alias is None or tiktoken is None:
        return None

    alias = tokenizer_name_or_alias
    if alias.startswith("tiktoken:"):
        alias = alias.split(":", 1)[1]

    try:
        return tiktoken.get_encoding(alias)
    except KeyError:
        try:
            return tiktoken.encoding_for_model(alias)
        except KeyError:
            return None


def get_encoding(tokenizer_name_or_alias):
    encoding = _resolve_tiktoken_encoding(tokenizer_name_or_alias)
    if encoding is None:
        raise KeyError(
            "Could not resolve tokenizer alias %r. Install tiktoken or configure a supported alias."
            % tokenizer_name_or_alias
        )
    return encoding


def tokenize_text(text, tokenizer_name_or_alias=None, require_exact=False):
    encoding = _resolve_tiktoken_encoding(tokenizer_name_or_alias)
    if encoding is not None:
        return encoding.encode(text)
    if require_exact:
        raise RuntimeError(
            "Exact tokenization requires tiktoken plus a supported tokenizer alias. "
            "Install tiktoken or choose a model/tokenizer alias that can be resolved."
        )
    return list(range(estimate_token_count(text, tokenizer_name_or_alias=tokenizer_name_or_alias)))


def estimate_token_count(text, tokenizer_name_or_alias=None):
    encoding = _resolve_tiktoken_encoding(tokenizer_name_or_alias)
    if encoding is not None:
        return len(encoding.encode(text))
    byte_length = len(text.encode("utf-8"))
    return max(1, int(math.ceil(byte_length / 3.0)))
