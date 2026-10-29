# mypy: allow-untyped-defs

# [MODULE DOC]
# Defines the abstract base fixture shared by all RPC test suites. It fixes the world size,
# computes the init method (TCP or file based) and declares the backend-specific hooks that
# concrete fixtures (for example the TensorPipe and faulty agent fixtures) must provide.
# No import-time side effects beyond imports.

import os
from abc import ABC, abstractmethod

import torch.testing._internal.dist_utils


class RpcAgentTestFixture(ABC):
    # [CLASS DOC]
    # Abstract mixin describing what an RPC test needs from an agent backend. Concrete
    # subclasses supply rpc_backend and rpc_backend_options plus the error-message regexes for
    # shutdown and timeout failures. It expects the test class to provide a file_name
    # attribute (used for the file based init method). Holds no state of its own.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of workers used by RPC tests, which is always 4.
        return 4

    @property
    def init_method(self):
        # [FUNCTION DOC]
        # Returns the rendezvous URL. If the environment variable RPC_INIT_WITH_TCP is "1",
        # builds a tcp URL from MASTER_ADDR and MASTER_PORT (raises KeyError if either is
        # unset); otherwise returns the file based init method.
        use_tcp_init = os.environ.get("RPC_INIT_WITH_TCP", None)
        if use_tcp_init == "1":
            master_addr = os.environ["MASTER_ADDR"]
            master_port = os.environ["MASTER_PORT"]
            return f"tcp://{master_addr}:{master_port}"
        else:
            return self.file_init_method

    @property
    def file_init_method(self):
        # [FUNCTION DOC]
        # Returns a file:// style init URL by filling the dist_utils template with this
        # test's file_name attribute.
        return torch.testing._internal.dist_utils.INIT_METHOD_TEMPLATE.format(
            file_name=self.file_name
        )

    @property
    @abstractmethod
    def rpc_backend(self):
        # [FUNCTION DOC]
        # Abstract property: subclasses return the backend type enum value to initialize RPC with.
        pass

    @property
    @abstractmethod
    def rpc_backend_options(self):
        # [FUNCTION DOC]
        # Abstract property: subclasses return the backend options object passed to RPC init.
        pass

    def setup_fault_injection(self, faulty_messages, messages_to_delay):  # noqa: B027
        # [FUNCTION DOC]
        # Default no-op hook called by the dist_init decorator with the requested failing
        # message names and delay mapping. Only the faulty agent fixture overrides it.
        """Method used by dist_init to prepare the faulty agent.

        Does nothing for other agents.
        """

    # Shutdown sequence is not well defined, so we may see any of the following
    # errors when running tests that simulate errors via a shutdown on the
    # remote end.
    @abstractmethod
    def get_shutdown_error_regex(self):
        # [FUNCTION DOC]
        # Abstract: returns a regex string matching any error an agent may emit when a peer
        # shuts down mid-test. Used with assertRaisesRegex.
        """
        Return various error message we may see from RPC agents while running
        tests that check for failures. This function is used to match against
        possible errors to ensure failures were raised properly.
        """

    @abstractmethod
    def get_timeout_error_regex(self):
        # [FUNCTION DOC]
        # Abstract: returns a (partial) regex string matching the error raised when an RPC
        # times out.
        """
        Returns a partial string indicating the error we should receive when an
        RPC has timed out. Useful for use with assertRaisesRegex() to ensure we
        have the right errors during timeout.
        """
