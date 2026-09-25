"""Resolve lifespan-owned services; never open resources per request."""

from typing import cast

from fastapi import Request

from reporecall.services.errors import ServiceUnavailable
from reporecall.services.runtime import ServiceResources


def get_services(request: Request) -> ServiceResources:
    services = cast(
        ServiceResources | None, getattr(request.app.state, "services", None)
    )
    if services is None:
        raise ServiceUnavailable("Application services are unavailable.")
    return services
