import re

from reporecall.generation.backend import LLMBackend
from reporecall.generation.config import RAGConfig
from reporecall.generation.context_builder import RAGContextBuilder
from reporecall.generation.exceptions import RAGResponseError
from reporecall.generation.prompt_builder import RAGPromptBuilder
from reporecall.models import MetadataFilter, RAGAnswer, RAGContext
from reporecall.retrieval import VectorRetriever

INSUFFICIENT_EVIDENCE_ANSWER = (
    "RepoRecall could not find repository evidence relevant to this question."
)
_EVIDENCE_LABEL = re.compile(r"\[(E[0-9]+)\]")


class RAGPipeline:
    """Orchestrate retrieval, bounded context, prompting, and grounded generation."""

    def __init__(
        self,
        *,
        retriever: VectorRetriever,
        context_builder: RAGContextBuilder,
        prompt_builder: RAGPromptBuilder,
        llm_backend: LLMBackend,
        config: RAGConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_backend = llm_backend
        self.config = config or RAGConfig()

    def answer(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
    ) -> RAGAnswer:
        """Answer one engineering-history question from retrieved evidence only."""

        if not query.strip():
            raise RAGResponseError("RAG query must not be blank.")

        hits = (
            self.retriever.search(query, k=self.config.top_k)
            if metadata_filter is None
            else self.retriever.search(
                query,
                k=self.config.top_k,
                metadata_filter=metadata_filter,
            )
        )
        if not hits:
            return RAGAnswer(
                query=query,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                model_name=None,
                evidence=(),
                retrieved_count=0,
                included_evidence_count=0,
                context_truncated=False,
                insufficient_evidence=True,
            )

        context = self.context_builder.build(
            query,
            hits,
            max_context_chars=self.config.max_context_chars,
        )
        prompt = self.prompt_builder.build(query, context)
        answer_text = self.llm_backend.generate(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )
        if not answer_text.strip():
            raise RAGResponseError("LLM backend returned an empty answer.")
        _validate_evidence_labels(answer_text, context)

        return RAGAnswer(
            query=query,
            answer=answer_text,
            model_name=self.llm_backend.model_name,
            evidence=context.evidence,
            retrieved_count=context.retrieved_count,
            included_evidence_count=context.included_evidence_count,
            context_truncated=context.truncated,
            insufficient_evidence=False,
        )


def _validate_evidence_labels(answer: str, context: RAGContext) -> None:
    allowed = {evidence.evidence_id for evidence in context.evidence}
    invalid = sorted(set(_EVIDENCE_LABEL.findall(answer)) - allowed)
    if invalid:
        labels = ", ".join(f"[{label}]" for label in invalid)
        raise RAGResponseError(f"LLM answer referenced unknown evidence: {labels}.")
