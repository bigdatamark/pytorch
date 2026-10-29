# [MODULE DOC] Docker helpers for lumen CLI tasks: a lazily created shared client and a
# check for the existence of a local image. Requires the docker package at import.
"""
Docker Utility helpers for CLI tasks.
"""

from __future__ import annotations

import logging

import docker
from docker.errors import APIError, NotFound


logger = logging.getLogger(__name__)

# lazy singleton so we don't reconnect every call
_docker_client: docker.DockerClient | None = None


def _get_client() -> docker.DockerClient:
    # [FUNCTION DOC] HELPER: Return the module level Docker client, creating it from
    # the environment on first use (mutates the global singleton).
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def local_image_exists(
    image_name: str, client: docker.DockerClient | None = None
) -> bool:
    # [FUNCTION DOC] Check whether a local image exists. Returns False at once for an
    # empty name; otherwise looks it up with the given client (or the shared one) and
    # returns True on success. Not-found and API errors are logged as errors and
    # yield False, so a missing image is logged as an error too.
    """Return True if a local Docker image exists."""
    if not image_name:
        return False

    client = client or _get_client()
    try:
        client.images.get(image_name)
        return True
    except (NotFound, APIError) as e:
        logger.error(
            "Error when checking Docker image '%s': %s",
            image_name,
            e.explanation if hasattr(e, "explanation") else str(e),
        )
        return False
