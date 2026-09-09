from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

from reporecall.config import get_settings
from reporecall.generation.config import RAGConfig
from reporecall.generation.exceptions import LLMBackendError, RAGResponseError


class _OpenAIResponse(Protocol):
    output_text: str


class _OpenAIResponses(Protocol):
    def create(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        temperature: float,
    ) -> _OpenAIResponse: ...


class _OpenAIClient(Protocol):
    responses: _OpenAIResponses


class _OpenAIModule(Protocol):
    OpenAI: Callable[..., _OpenAIClient]


class OpenAILLMBackend:
    """Lazy production LLM backend using the official OpenAI Responses API."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        api_key: str | None = None,
        client: _OpenAIClient | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self._api_key = api_key
        self._client = client

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate answer text from separated instructions and evidence."""

        if not system_prompt.strip() or not user_prompt.strip():
            raise RAGResponseError("OpenAI prompts must not be blank.")

        client = self._get_client()
        try:
            response = client.responses.create(
                model=self.config.model_name,
                instructions=system_prompt,
                input=user_prompt,
                temperature=self.config.temperature,
            )
        except Exception as exc:
            raise LLMBackendError(
                f"OpenAI generation failed for model {self.model_name!r}."
            ) from exc

        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise RAGResponseError("OpenAI returned an empty text response.")
        return output_text

    def _get_client(self) -> _OpenAIClient:
        if self._client is None:
            api_key = self._api_key or get_settings().openai_api_key
            if not api_key:
                raise LLMBackendError(
                    "OPENAI_API_KEY is required for the OpenAI LLM backend."
                )
            self._client = _load_openai_client(api_key)
        return self._client


def _load_openai_client(api_key: str) -> _OpenAIClient:
    try:
        module = cast(_OpenAIModule, import_module("openai"))
        return module.OpenAI(api_key=api_key)
    except Exception as exc:
        raise LLMBackendError("Unable to initialize the OpenAI client.") from exc
