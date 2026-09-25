"""Application-facing corpus discovery and detail operations."""

from reporecall.github import GitHubRepository
from reporecall.models import EngineeringEvent, RetrievalChunk, RetrievalDocument
from reporecall.persistence.catalog import CorpusCatalog, CorpusStatistics
from reporecall.services.errors import ResourceNotFound


class CorpusService:
    """Delegate database reads while assigning application-level not-found semantics."""

    def __init__(self, catalog: CorpusCatalog) -> None:
        self.catalog = catalog

    def check_ready(self) -> None:
        self.catalog.check_ready()

    def statistics(self) -> CorpusStatistics:
        return self.catalog.statistics()

    def repositories(self) -> tuple[GitHubRepository, ...]:
        return self.catalog.repositories()

    def event(self, identifier: str) -> EngineeringEvent:
        result = self.catalog.event(identifier)
        if result is None:
            raise ResourceNotFound("event")
        return result

    def document(self, identifier: str) -> RetrievalDocument:
        result = self.catalog.document(identifier)
        if result is None:
            raise ResourceNotFound("document")
        return result

    def chunk(self, identifier: str) -> RetrievalChunk:
        result = self.catalog.chunk(identifier)
        if result is None:
            raise ResourceNotFound("chunk")
        return result
