"""Own shared application resources independently of HTTP."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import Field, SecretStr, ValidationError

from reporecall.config import Settings
from reporecall.embeddings import EmbeddingConfig, SentenceTransformerEmbeddingBackend
from reporecall.persistence import (
    DatabaseConfig,
    PersistentCorpusRepository,
    PostgresVectorRetriever,
    create_database_engine,
    create_session_factory,
)
from reporecall.persistence.catalog import CorpusCatalog
from reporecall.provenance import CitationBundleBuilder, CitationProvenanceIndex
from reporecall.query import (
    OpenAIQueryUnderstandingBackend,
    QueryUnderstandingConfig,
    QueryUnderstandingService,
)
from reporecall.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    CrossEncoderRerankerConfig,
    HybridRetrievalConfig,
    HybridRetriever,
    KeywordRetriever,
    ReciprocalRankFusion,
    ReciprocalRankFusionConfig,
    RelationshipContextExpander,
    RelationshipContextExpansionConfig,
    RelationshipContextIndex,
    SentenceTransformerCrossEncoderBackend,
)
from reporecall.services.corpus_service import CorpusService
from reporecall.services.errors import ServiceUnavailable
from reporecall.services.search_service import SearchService


class ServiceConfig(Settings):
    """Bounded search settings and explicit browser origins."""

    api_cors_origins: tuple[str, ...] = ()
    api_docs_enabled: bool = True
    api_candidate_k: int = Field(default=50, ge=50, le=200)
    api_embedding_model: str = EmbeddingConfig().model_name
    api_reranker_model: str = CrossEncoderRerankerConfig().model_name
    api_query_model: str = QueryUnderstandingConfig().model_name


@dataclass(frozen=True)
class ServiceResources:
    corpus: CorpusService
    search: SearchService
    query: QueryUnderstandingService


@contextmanager
def open_services(config: ServiceConfig) -> Iterator[ServiceResources]:
    """Build one corpus snapshot; neural/provider clients remain lazy."""
    if not config.database_url:
        raise ServiceUnavailable("Database configuration is required.")
    try:
        database_config = DatabaseConfig(url=SecretStr(config.database_url))
    except ValidationError as exc:
        raise ServiceUnavailable("Invalid database configuration.") from exc
    engine = create_database_engine(database_config)
    try:
        sessions = create_session_factory(engine)
        catalog = CorpusCatalog(sessions)
        catalog.check_ready()
        corpus = PersistentCorpusRepository(session_factory=sessions).load_corpus()
        selected = [
            item
            for item in corpus.embeddings
            if item.model_name == config.api_embedding_model
        ]
        if {item.chunk_id for item in selected} != {
            chunk.chunk_id for chunk in corpus.chunks
        }:
            raise ServiceUnavailable(
                "The configured embedding model must cover the corpus."
            )
        embedding = SentenceTransformerEmbeddingBackend(
            EmbeddingConfig(model_name=config.api_embedding_model)
        )
        dense = PostgresVectorRetriever(
            session_factory=sessions,
            embedding_backend=embedding,
            model_name=config.api_embedding_model,
        )
        index = BM25Index()
        index.build(corpus.chunks)
        keyword = KeywordRetriever(
            index=index, chunks={chunk.chunk_id: chunk for chunk in corpus.chunks}
        )
        reranker_config = CrossEncoderRerankerConfig(
            model_name=config.api_reranker_model,
            candidate_k=config.api_candidate_k,
            top_k=50,
        )
        search = SearchService(
            hybrid=HybridRetriever(
                vector_retriever=dense,
                keyword_retriever=keyword,
                config=HybridRetrievalConfig(
                    dense_k=config.api_candidate_k, keyword_k=config.api_candidate_k
                ),
            ),
            fusion=ReciprocalRankFusion(
                ReciprocalRankFusionConfig(top_k=config.api_candidate_k)
            ),
            reranker=CrossEncoderReranker(
                backend=SentenceTransformerCrossEncoderBackend(reranker_config),
                config=reranker_config,
            ),
            expander=RelationshipContextExpander(
                index=RelationshipContextIndex(
                    events=corpus.events, chunks=corpus.chunks
                ),
                config=RelationshipContextExpansionConfig(seed_k=50),
            ),
            citations=CitationBundleBuilder(
                provenance_index=CitationProvenanceIndex(
                    documents=corpus.documents, chunks=corpus.chunks
                )
            ),
        )
        yield ServiceResources(
            CorpusService(catalog),
            search,
            QueryUnderstandingService(
                OpenAIQueryUnderstandingBackend(
                    QueryUnderstandingConfig(model_name=config.api_query_model),
                    api_key=config.openai_api_key,
                )
            ),
        )
    finally:
        engine.dispose()
