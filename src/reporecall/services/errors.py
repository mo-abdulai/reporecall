"""Application errors independent of transport protocols."""


class ServiceUnavailable(Exception):
    """A required application resource cannot currently serve requests."""


class ResourceNotFound(Exception):
    """A canonical corpus resource does not exist."""

    def __init__(self, resource: str) -> None:
        self.resource = resource
        super().__init__(f"{resource} was not found")
