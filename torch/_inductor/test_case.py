# [MODULE DOC] Base test utilities for Inductor tests: a run_tests wrapper and a TestCase that enables
# the autograd and FX graph caches and isolates cache directories per test.
import contextlib
import os

from torch._dynamo.test_case import (
    run_tests as dynamo_run_tests,
    TestCase as DynamoTestCase,
)
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.utils import fresh_cache


def run_tests(needs: str | tuple[str, ...] = ()) -> None:
    # [FUNCTION DOC] Run the dynamo test runner, forwarding the needs requirement (string or tuple of
    # strings naming required capabilities). Returns None.
    dynamo_run_tests(needs)


class TestCase(DynamoTestCase):
    # [CLASS DOC] Dynamo-based TestCase for Inductor. State: an ExitStack holding context managers entered in
    # setUp and closed in tearDown. Enables the functorch autograd cache, the Inductor FX graph cache
    # (unless env vars set it) and a fresh cache directory (unless disabled by env vars).
    """
    A base TestCase for inductor tests. Enables FX graph caching and isolates
    the cache directory for each test.
    """

    def setUp(self) -> None:
        # [FUNCTION DOC] Enter the config patches: autograd cache on; fx_graph_cache on unless the FX graph cache
        # env vars are set; fresh_cache unless INDUCTOR_TEST_DISABLE_FRESH_CACHE is 1 or TORCH_COMPILE_DEBUG
        # is 1. Mutates global config until tearDown.
        super().setUp()
        self._inductor_test_stack = contextlib.ExitStack()
        self._inductor_test_stack.enter_context(
            functorch_config.patch(
                {
                    "enable_autograd_cache": True,
                }
            )
        )

        if (
            "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ
            and "TORCHINDUCTOR_FX_GRAPH_CACHE_DEFAULT" not in os.environ
        ):
            self._inductor_test_stack.enter_context(
                config.patch({"fx_graph_cache": True})
            )

        if (
            os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1"
            and os.environ.get("TORCH_COMPILE_DEBUG") != "1"
        ):
            self._inductor_test_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        # [FUNCTION DOC] Run parent tearDown and then undo every context entered in setUp.
        super().tearDown()
        self._inductor_test_stack.close()
