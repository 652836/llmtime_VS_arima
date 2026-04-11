from models.providers.base import BaseProvider, ProviderCapabilityError


class LocalHFProvider(BaseProvider):
    name = "local_hf"

    def generate(self, spec, request):
        raise ProviderCapabilityError(
            "Model '%s' is registered for the LocalHF provider placeholder, "
            "but this provider is not implemented in the Qwen refactor round."
            % spec.logical_model_name
        )
