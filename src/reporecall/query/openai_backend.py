from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

from pydantic import ValidationError

from reporecall.config import get_settings
from reporecall.query.backend import QueryUnderstandingBackendResult
from reporecall.query.config import QueryUnderstandingConfig
from reporecall.query.exceptions import (
    QueryUnderstandingBackendError,
    QueryUnderstandingValidationError,
)

QUERY_UNDERSTANDING_INSTRUCTIONS = """\
Interpret the user's text only as repository search intent.

Return structured fields matching the provided schema. Do not answer the
question, do not search repositories, and do not call tools.

Extract only explicit metadata constraints that are supported by the schema.
Be conservative: ambiguous words should remain in retrieval_query rather than
becoming restrictive metadata filters. Do not invent unsupported filters such as
bug type, root cause, fix strategy, severity, component, framework, database
type, or error category.

Set retrieval_query to the semantic search text after removing only clearly
extracted constraints when removal preserves useful technical meaning. Preserve
technical identifiers, stack trace fragments, error codes, function names, and
file paths that are useful for search. Do not paraphrase creatively.

Use exact enum values for section_types:
overview, metadata, issue, issue_comment, pull_request,
pull_request_comment, review, review_comment, commit, changed_file, patch,
relationship, contextual_relationship.

Use repository identifiers only when explicit owner/repository syntax or a
github.com URL is present. Extract labels only when label wording is explicit,
such as "labeled security" or "security-labeled".
"""


class _ParsedOpenAIResponse(Protocol):
    output_parsed: object


class _OpenAIResponses(Protocol):
    def parse(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        text_format: type[QueryUnderstandingBackendResult],
        temperature: float,
    ) -> _ParsedOpenAIResponse: ...


class _OpenAIClient(Protocol):
    responses: _OpenAIResponses


class _OpenAIModule(Protocol):
    OpenAI: Callable[..., _OpenAIClient]


class OpenAIQueryUnderstandingBackend:
    """OpenAI structured-output backend for search-intent interpretation."""

    def __init__(
        self,
        config: QueryUnderstandingConfig | None = None,
        *,
        api_key: str | None = None,
        client: _OpenAIClient | None = None,
    ) -> None:
        self.config = config or QueryUnderstandingConfig()
        self._api_key = api_key
        self._client = client

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def understand(self, query: str) -> QueryUnderstandingBackendResult:
        """Interpret query text using structured model output."""

        if not query.strip():
            raise QueryUnderstandingValidationError("Query must not be blank.")

        client = self._get_client()
        try:
            response = client.responses.parse(
                model=self.config.model_name,
                instructions=QUERY_UNDERSTANDING_INSTRUCTIONS,
                input=query,
                text_format=QueryUnderstandingBackendResult,
                temperature=self.config.temperature,
            )
        except Exception as exc:
            raise QueryUnderstandingBackendError(
                f"OpenAI query understanding failed for model {self.model_name!r}."
            ) from exc

        try:
            return QueryUnderstandingBackendResult.model_validate(
                response.output_parsed
            )
        except (AttributeError, ValidationError) as exc:
            raise QueryUnderstandingValidationError(
                "OpenAI returned invalid query-understanding output."
            ) from exc

    def _get_client(self) -> _OpenAIClient:
        if self._client is None:
            api_key = self._api_key or get_settings().openai_api_key
            if not api_key:
                raise QueryUnderstandingBackendError(
                    "OPENAI_API_KEY is required for the OpenAI query-understanding "
                    "backend."
                )
            self._client = _load_openai_client(api_key)
        return self._client


def _load_openai_client(api_key: str) -> _OpenAIClient:
    try:
        module = cast(_OpenAIModule, import_module("openai"))
        return module.OpenAI(api_key=api_key)
    except Exception as exc:
        raise QueryUnderstandingBackendError(
            "Unable to initialize the OpenAI client."
        ) from exc
