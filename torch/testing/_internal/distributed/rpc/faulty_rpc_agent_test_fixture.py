# mypy: allow-untyped-defs

# [MODULE DOC]
# Test fixture that runs RPC tests on the FAULTY_TENSORPIPE backend, which can drop or delay
# chosen message types. Defines the default lists of retryable (failable) message types and of
# delayed message types. Importing the private _testing module registers the faulty backend.

import torch.distributed.rpc as rpc
import torch.distributed.rpc._testing  # noqa: F401
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


# The following message types are currently retried in the RREF protocol and
# distributed autograd. Thus only these messages should be tested with the
# Faulty RPC Agent.
retryable_message_types = [
    "RREF_FORK_REQUEST",
    "RREF_CHILD_ACCEPT",
    "RREF_USER_DELETE",
    "CLEANUP_AUTOGRAD_CONTEXT_REQ",
]

# The following messages incur the corresponding delay in seconds while being
# processed in FaultyTensorPipeAgent's enqueueSend() function.
default_messages_to_delay = {
    "PYTHON_CALL": 1.5,  # Python UDF
    "SCRIPT_CALL": 1.5,  # Script/Builtin
}


class FaultyRpcAgentTestFixture(RpcAgentTestFixture):
    # [CLASS DOC]
    # Concrete fixture for the faulty TensorPipe agent. Instance state: messages_to_fail (list of
    # message type names the agent fails) and messages_to_delay (mapping of message type name
    # to delay seconds), both initialized from module defaults and overridable through
    # setup_fault_injection. Backend options use 8 worker threads and 3 failed sends.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Forwards all arguments to the parent initializer, then sets the default failing and
        # delayed message configuration. Note both attributes alias the module-level defaults,
        # so in-place mutation would leak across instances.
        super().__init__(*args, **kwargs)
        self.messages_to_fail = retryable_message_types
        self.messages_to_delay = default_messages_to_delay

    @property
    def rpc_backend(self):
        # [FUNCTION DOC]
        # Returns the FAULTY_TENSORPIPE backend type from the backend registry.
        return rpc.backend_registry.BackendType["FAULTY_TENSORPIPE"]

    @property
    def rpc_backend_options(self):
        # [FUNCTION DOC]
        # Builds faulty-agent backend options from the current init method with 8 worker
        # threads, 3 failed sends per faulty message and this fixture's failing and delayed
        # message configuration.
        return rpc.backend_registry.construct_rpc_backend_options(
            self.rpc_backend,
            init_method=self.init_method,
            num_worker_threads=8,
            num_fail_sends=3,
            messages_to_fail=self.messages_to_fail,
            messages_to_delay=self.messages_to_delay,
        )

    def setup_fault_injection(self, faulty_messages, messages_to_delay):
        # [FUNCTION DOC]
        # Overrides the failing message list and/or the delay mapping when the given value is
        # not None; a None argument keeps the current setting. An empty list or mapping is
        # honored. Mutates instance state only.
        if faulty_messages is not None:
            self.messages_to_fail = faulty_messages
        if messages_to_delay is not None:
            self.messages_to_delay = messages_to_delay

    def get_shutdown_error_regex(self):
        # [FUNCTION DOC]
        # Returns one regex alternation matching thread pool task exceptions and connection
        # reset or closed by peer errors.
        error_regexes = [
            "Exception in thread pool task",
            "Connection reset by peer",
            "Connection closed by peer",
        ]
        return "|".join([f"({error_str})" for error_str in error_regexes])

    def get_timeout_error_regex(self):
        # [FUNCTION DOC]
        # Returns the partial message text of an RPC timeout error.
        return "RPC ran for more than"
