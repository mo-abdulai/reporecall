"""FastAPI factory with explicitly owned, lazy application resources."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from reporecall.api.routes import router
from reporecall.api.schemas import ErrorResponse
from reporecall.embeddings import EmbeddingError
from reporecall.persistence import PersistenceError
from reporecall.provenance import CitationProvenanceError
from reporecall.query import QueryUnderstandingError
from reporecall.retrieval import RelationshipContextError, RetrievalError
from reporecall.services.errors import ResourceNotFound, ServiceUnavailable
from reporecall.services.runtime import ServiceConfig, ServiceResources, open_services


def create_app(
    *,
    config: ServiceConfig | None = None,
    resources_factory: Callable[[], AbstractContextManager[ServiceResources]]
    | None = None,
) -> FastAPI:
    """Construct HTTP wiring without database connections or model downloads."""
    settings = config or ServiceConfig()
    factory = resources_factory or (lambda: open_services(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        stack = ExitStack()
        app.state.services = None
        try:
            try:
                app.state.services = await run_in_threadpool(
                    lambda: stack.enter_context(factory())
                )
            except (
                ServiceUnavailable,
                PersistenceError,
                RelationshipContextError,
                CitationProvenanceError,
            ):
                # Keep liveness available; readiness reports the unavailable service.
                app.state.services = None
            yield
        finally:
            app.state.services = None
            await run_in_threadpool(stack.close)

    app = FastAPI(
        title="RepoRecall API",
        version="1.0.0",
        description="Repository evidence search with ranked retrieval, structural context, and source-faithful citations. No answer generation.",
        lifespan=lifespan,
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url="/redoc" if settings.api_docs_enabled else None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api_cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    async def error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            status, code, message = 422, "invalid_request", "Request validation failed."
        elif isinstance(exc, ResourceNotFound):
            status, code, message = (
                404,
                "not_found",
                "Requested resource was not found.",
            )
        elif isinstance(exc, QueryUnderstandingError):
            status, code, message = (
                502,
                "query_provider_error",
                "Query understanding did not complete.",
            )
        elif isinstance(
            exc,
            (
                ServiceUnavailable,
                PersistenceError,
                EmbeddingError,
                RetrievalError,
                RelationshipContextError,
                CitationProvenanceError,
            ),
        ):
            status, code, message = (
                503,
                "service_unavailable",
                "A required service is unavailable.",
            )
        elif isinstance(exc, HTTPException):
            status, code, message = (
                exc.status_code,
                "http_error",
                "Request could not be completed.",
            )
        else:
            status, code, message = (
                500,
                "internal_error",
                "An unexpected error occurred.",
            )
        return JSONResponse(
            status_code=status, content={"error": {"code": code, "message": message}}
        )

    for error in (
        RequestValidationError,
        ResourceNotFound,
        QueryUnderstandingError,
        ServiceUnavailable,
        PersistenceError,
        EmbeddingError,
        RetrievalError,
        RelationshipContextError,
        CitationProvenanceError,
        HTTPException,
        Exception,
    ):
        app.add_exception_handler(error, error_handler)
    app.include_router(
        router,
        responses={
            status: {"model": ErrorResponse} for status in (404, 422, 500, 502, 503)
        },
    )
    return app
