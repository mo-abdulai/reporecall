"""Optional structured OpenAI semantic judge; never an answer generator."""

import json
from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from reporecall.config import get_settings
from reporecall.evaluation.rag_evaluator import validate_judge_output
from reporecall.evaluation.rag_judge_backend import (
    RAGJudgeBackendError,
    RAGJudgeOutputError,
)
from reporecall.models.evaluation import Nonblank
from reporecall.models.rag_evaluation import (
    RAGJudgeBackendResult,
    RAGJudgeRequest,
    RAGJudgeResult,
)


class OpenAIRAGJudgeConfig(BaseModel):
    """Explicit judge model, configured independently of any answer model."""

    model_name: Nonblank
    model_config = ConfigDict(frozen=True, extra="forbid")


class _ParsedResponse(Protocol):
    output_parsed: object


class _Responses(Protocol):
    def parse(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        text_format: type[RAGJudgeBackendResult],
        store: bool,
    ) -> _ParsedResponse: ...


class _Client(Protocol):
    responses: _Responses


class _OpenAIModule(Protocol):
    OpenAI: Callable[..., _Client]


def render_judge_input(request: RAGJudgeRequest) -> str:
    """JSON-delimit all untrusted inputs; preserve exact content, omit scores."""
    evidence = []
    for item in request.evidence:
        evidence.append(
            {
                "label": item.citation.label,
                "kind": item.citation.kind.value,
                "content": item.content,
                "artifact": item.artifact.model_dump(mode="json")
                if item.artifact
                else None,
                "section_type": item.section_type.value,
                "expanded_from": item.expanded_from,
                "expansion_reasons": [
                    r.model_dump(mode="json") for r in item.expansion_reasons
                ],
            }
        )
    return json.dumps(
        {
            "QUERY": request.query,
            "ANSWER": request.answer,
            "EXPECTED FACTS": [f.model_dump() for f in request.expected_facts],
            "REFERENCE ANSWER": request.reference_answer,
            "EVIDENCE": evidence,
            "VALID CITED LABELS": request.valid_cited_labels,
            "CONTEXT TRUNCATED": request.context_truncated,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class OpenAIRAGEvaluationJudgeBackend:
    """Lazy structured-output provider with no tools or answer-generation calls."""

    def __init__(
        self,
        config: OpenAIRAGJudgeConfig,
        *,
        api_key: str | None = None,
        client: _Client | None = None,
    ) -> None:
        self.config = config
        self._api_key = api_key
        self._client = client

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def judge(self, request: RAGJudgeRequest) -> RAGJudgeResult:
        """Assess only supplied data and validate exact fact/citation coverage."""
        client = self._get_client()
        try:
            response = client.responses.parse(
                model=self.model_name,
                instructions=request.rubric,
                input=render_judge_input(request),
                text_format=RAGJudgeBackendResult,
                store=False,
            )
        except Exception as exc:
            raise RAGJudgeBackendError("OpenAI semantic judge request failed.") from exc
        try:
            raw = response.output_parsed
            payload = raw.model_dump() if isinstance(raw, BaseModel) else raw
            parsed = RAGJudgeBackendResult.model_validate(payload)
        except (AttributeError, ValidationError) as exc:
            raise RAGJudgeOutputError(
                "OpenAI returned invalid semantic judge output."
            ) from exc
        result = RAGJudgeResult(
            **parsed.model_dump(), judge_model=self.model_name, rubric=request.rubric
        )
        return validate_judge_output(result, request, self.model_name)

    def _get_client(self) -> _Client:
        if self._client is None:
            key = self._api_key or get_settings().openai_api_key
            if not key:
                raise RAGJudgeBackendError(
                    "OPENAI_API_KEY is required for the OpenAI semantic judge."
                )
            self._client = _load_openai_client(key)
        return self._client


def _load_openai_client(api_key: str) -> _Client:
    try:
        module = cast(_OpenAIModule, import_module("openai"))
        return module.OpenAI(api_key=api_key)
    except Exception as exc:
        raise RAGJudgeBackendError(
            "Unable to initialize OpenAI semantic judge client."
        ) from exc
