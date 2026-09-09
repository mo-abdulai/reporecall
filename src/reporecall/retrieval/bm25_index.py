import hashlib
from collections.abc import Callable, Iterable, Sequence
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from reporecall.models import KeywordIndexMatch, RetrievalChunk
from reporecall.retrieval.engineering_tokenizer import EngineeringTokenizer
from reporecall.retrieval.exceptions import BM25IndexError


class BM25Config(BaseModel):
    """Small immutable configuration for the standard BM25 baseline."""

    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)

    model_config = ConfigDict(frozen=True)


class _BM25Model(Protocol):
    def index(
        self,
        corpus: Iterable[list[str]],
        *,
        show_progress: bool,
    ) -> None: ...

    def get_scores(self, query_tokens_single: list[str]) -> object: ...


class _BM25Module(Protocol):
    BM25: Callable[..., _BM25Model]


class BM25Index:
    """In-memory BM25 index with stable row-to-chunk identity."""

    def __init__(
        self,
        *,
        tokenizer: EngineeringTokenizer | None = None,
        config: BM25Config | None = None,
    ) -> None:
        self.tokenizer = tokenizer or EngineeringTokenizer()
        self.config = config or BM25Config()
        self._bm25 = _load_bm25_module()
        self._model: _BM25Model | None = None
        self._chunk_ids: tuple[str, ...] = ()
        self._source_text_sha256: tuple[str, ...] = ()
        self._tokenized_corpus: tuple[tuple[str, ...], ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether the index contains no lexical documents."""

        return not self._chunk_ids

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return BM25 row identities in build input order."""

        return self._chunk_ids

    @property
    def tokenized_corpus(self) -> tuple[tuple[str, ...], ...]:
        """Return the immutable tokenized corpus used by the index."""

        return self._tokenized_corpus

    def build(self, chunks: Sequence[RetrievalChunk]) -> None:
        """Replace index contents using source chunk input order as row order."""

        records = list(chunks)
        if not records:
            self._model = None
            self._chunk_ids = ()
            self._source_text_sha256 = ()
            self._tokenized_corpus = ()
            return

        chunk_ids = tuple(chunk.chunk_id for chunk in records)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise BM25IndexError("Duplicate chunk IDs cannot be indexed by BM25.")

        tokenized_corpus = tuple(
            self.tokenizer.tokenize(chunk.text) for chunk in records
        )
        for chunk, tokens in zip(records, tokenized_corpus, strict=True):
            if not tokens:
                raise BM25IndexError(
                    f"Chunk {chunk.chunk_id!r} produced no BM25 tokens."
                )

        try:
            model = self._bm25.BM25(
                k1=self.config.k1,
                b=self.config.b,
                method="lucene",
            )
            model.index(
                [list(tokens) for tokens in tokenized_corpus],
                show_progress=False,
            )
        except Exception as exc:
            raise BM25IndexError("BM25 failed while indexing retrieval chunks.") from exc

        self._model = model
        self._chunk_ids = chunk_ids
        self._source_text_sha256 = tuple(
            hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            for chunk in records
        )
        self._tokenized_corpus = tokenized_corpus

    def search(
        self,
        query_tokens: Sequence[str],
        *,
        k: int,
        candidate_chunk_ids: Sequence[str] | None = None,
    ) -> list[KeywordIndexMatch]:
        """Return positive-score BM25 matches ranked within eligible candidates."""

        if k <= 0:
            raise BM25IndexError("Search result count k must be greater than zero.")
        if self._model is None or not self._chunk_ids:
            return []
        if not query_tokens:
            return []

        candidate_rows = _candidate_rows(self._chunk_ids, candidate_chunk_ids)
        if not candidate_rows:
            return []

        try:
            raw_scores = self._model.get_scores(list(query_tokens))
        except Exception as exc:
            raise BM25IndexError("BM25 failed while scoring the query.") from exc

        scores = np.asarray(raw_scores)
        if scores.shape != (len(self._chunk_ids),):
            raise BM25IndexError("BM25 returned an invalid score array shape.")
        if not bool(np.isfinite(scores).all()):
            raise BM25IndexError("BM25 returned a non-finite lexical score.")

        candidates = [
            (
                self._chunk_ids[row_id],
                self._source_text_sha256[row_id],
                float(scores[row_id]),
            )
            for row_id in candidate_rows
            if float(scores[row_id]) > 0
        ]
        candidates.sort(key=lambda candidate: (-candidate[2], candidate[0]))
        return [
            KeywordIndexMatch(
                chunk_id=chunk_id,
                source_text_sha256=source_hash,
                score=score,
                rank=rank,
            )
            for rank, (chunk_id, source_hash, score) in enumerate(
                candidates[:k],
                start=1,
            )
        ]


def _candidate_rows(
    chunk_ids: tuple[str, ...],
    candidate_chunk_ids: Sequence[str] | None,
) -> tuple[int, ...]:
    if candidate_chunk_ids is None:
        return tuple(range(len(chunk_ids)))

    candidate_ids = tuple(candidate_chunk_ids)
    if any(not chunk_id.strip() for chunk_id in candidate_ids):
        raise BM25IndexError("Candidate chunk IDs must not be blank.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise BM25IndexError("Candidate chunk IDs must be unique.")

    indexed_ids = set(chunk_ids)
    missing_ids = sorted(set(candidate_ids) - indexed_ids)
    if missing_ids:
        raise BM25IndexError(
            f"Candidate chunk ID {missing_ids[0]!r} is missing from the BM25 index."
        )

    selected = set(candidate_ids)
    return tuple(
        row_id for row_id, chunk_id in enumerate(chunk_ids) if chunk_id in selected
    )


def _load_bm25_module() -> _BM25Module:
    try:
        module = import_module("bm25s")
    except ModuleNotFoundError as exc:
        raise BM25IndexError(
            "Keyword retrieval requires the 'bm25s' dependency."
        ) from exc
    return cast(_BM25Module, module)
