# [MODULE DOC] Unit tests for docker_helper using mock clients: image existence results
# for found, not-found and API-error cases, and the lazily created shared client.
import unittest
from unittest import mock
from unittest.mock import MagicMock

import docker.errors as derr
from cli.lib.common.docker_helper import _get_client, local_image_exists


class TestDockerImageHelpers(unittest.TestCase):
    # [CLASS DOC] Test case for the docker helper functions; every test starts with the
    # module level client singleton reset to unset.
    def setUp(self):
        # [FUNCTION DOC] Patch the singleton client variable to None for the test and
        # register cleanup to restore it.
        # Reset the singleton in the target module
        patcher = mock.patch("cli.lib.common.docker_helper._docker_client", None)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_local_image_exists_true(self):
        # [FUNCTION DOC] A client whose lookup succeeds yields True.
        # Mock a docker client whose images.get returns an object (no exception)
        mock_client = MagicMock()
        mock_client.images.get.return_value = object()
        ok = local_image_exists("repo:tag", client=mock_client)
        self.assertTrue(ok)

    def test_local_image_exists_not_found_false(self):
        # [FUNCTION DOC] A client whose lookup raises the not-found error yields False.
        mock_client = MagicMock()
        # Raise docker.errors.NotFound
        mock_client.images.get.side_effect = derr.NotFound("nope")
        ok = local_image_exists("missing:latest", client=mock_client)
        self.assertFalse(ok)

    def test_local_image_exists_api_error_false(self):
        # [FUNCTION DOC] A client whose lookup raises a Docker API error yields False.
        mock_client = MagicMock()
        mock_client.images.get.side_effect = derr.APIError("boom", None)

        ok = local_image_exists("broken:tag", client=mock_client)
        self.assertFalse(ok)

    def test_local_image_exists_uses_lazy_singleton(self):
        # [FUNCTION DOC] With the environment client factory mocked, calling the client
        # getter twice must create the client once and return the same cached object.
        # Patch docker.from_env used by _get_client()
        with mock.patch(
            "cli.lib.common.docker_helper.docker.from_env"
        ) as mock_from_env:
            mock_docker_client = MagicMock()
            mock_from_env.return_value = mock_docker_client

            # First call should create and cache the client
            c1 = _get_client()
            self.assertIs(c1, mock_docker_client)
            mock_from_env.assert_called_once()

            # Second call should reuse cached client (no extra from_env calls)
            c2 = _get_client()
            self.assertIs(c2, mock_docker_client)
            mock_from_env.assert_called_once()  # still once

    def test_local_image_exists_without_client_param_calls_get_client_once(self):
        # [FUNCTION DOC] With the client getter mocked, two image checks without an
        # explicit client must each call the getter and look up each image name once.
        # KNOWN ISSUE: the test name says called once but the assertion expects two calls.
        # Ensure _get_client is called and cached; local_image_exists should reuse it
        with mock.patch("cli.lib.common.docker_helper._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # 1st call
            local_image_exists("repo:tag")
            # 2nd call
            local_image_exists("repo:tag2")

            # local_image_exists should call _get_client each time,
            # but your _get_client itself caches docker.from_env.
            self.assertEqual(mock_get_client.call_count, 2)
            self.assertEqual(mock_client.images.get.call_count, 2)
            mock_client.images.get.assert_any_call("repo:tag")
            mock_client.images.get.assert_any_call("repo:tag2")


if __name__ == "__main__":
    unittest.main()
