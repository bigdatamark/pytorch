# mypy: allow-untyped-defs

# [MODULE DOC]
# Test fixture that specializes the generic RPC agent test fixture for the TensorPipe
# backend. Defines one class, TensorPipeRpcAgentTestFixture, which supplies the backend
# type, backend options and expected error-message regexes. No import-time side effects
# beyond imports.

import torch.distributed.rpc as rpc
from torch.testing._internal.common_distributed import tp_transports
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


class TensorPipeRpcAgentTestFixture(RpcAgentTestFixture):
    # [CLASS DOC]
    # Concrete RpcAgentTestFixture for the TensorPipe agent. Holds no state of its own;
    # it relies on self.init_method from the base fixture. Meant to be mixed into generic
    # RPC test suites so they run against TensorPipe.
    @property
    def rpc_backend(self):
        # [FUNCTION DOC]
        # Returns the TensorPipe member of the RPC backend type enumeration.
        return rpc.backend_registry.BackendType["TENSORPIPE"]

    @property
    def rpc_backend_options(self):
        # [FUNCTION DOC]
        # Builds TensorPipe backend options from the backend registry, using the fixture's
        # init_method and the transports returned by tp_transports. Returns the options object.
        return rpc.backend_registry.construct_rpc_backend_options(
            self.rpc_backend, init_method=self.init_method, _transports=tp_transports()
        )

    def get_shutdown_error_regex(self):
        # [FUNCTION DOC]
        # Returns a regex string matched against shutdown-time errors. Currently built from
        # a single catch-all pattern, so it matches any message.
        # FIXME Once we consolidate the error messages returned by the
        # TensorPipe agent put some more specific regex here.
        error_regexes = [".*"]
        return "|".join([f"({error_str})" for error_str in error_regexes])

    def get_timeout_error_regex(self):
        # [FUNCTION DOC]
        # Returns the regex string expected in the error message of a timed-out RPC.
        return "RPC ran for more than"
