# mypy: allow-untyped-defs
# [MODULE DOC] Internal RPC helpers for dynamic group membership: a store-based mutual
# exclusion context manager for join/leave and a wrapper that asks the current
# TensorPipe agent to update membership. Creates a module logger at import.
import logging
from contextlib import contextmanager
from typing import cast


logger = logging.getLogger(__name__)


@contextmanager
def _group_membership_management(store, name, is_join):
    # [FUNCTION DOC] HELPER: Context manager serializing rank join/leave across processes.
    # Loops trying to claim a shared token key in the store via compare-and-set with a token
    # unique to this name and operation. On success it yields, then clears the token key and
    # marks its own token "Done" so waiters proceed. Otherwise waits on the current holder's
    # token; if the wait raises RuntimeError it logs an error and re-raises.
    token_key = "RpcGroupManagementToken"
    join_or_leave = "join" if is_join else "leave"
    my_token = f"Token_for_{name}_{join_or_leave}"
    while True:
        # Retrieve token from store to signal start of rank join/leave critical section
        returned = store.compare_set(token_key, "", my_token).decode()
        if returned == my_token:
            # Yield to the function this context manager wraps
            yield
            # Finished, now exit and release token
            # Update from store to signal end of rank join/leave critical section
            store.set(token_key, "")
            # Other will wait for this token to be set before they execute
            store.set(my_token, "Done")
            break
        else:
            # Store will wait for the token to be released
            try:
                store.wait([returned])
            except RuntimeError:
                logger.error(
                    "Group membership token %s timed out waiting for %s to be released.",
                    my_token,
                    returned,
                )
                raise


def _update_group_membership(worker_info, my_devices, reverse_device_map, is_join):
    # [FUNCTION DOC] HELPER: Delegates to the current RPC agent (treated as a TensorPipeAgent)
    # to update group membership for the given worker info, its devices and reverse device
    # map; is_join selects join vs leave. Returns the agent's result.
    from . import api, TensorPipeAgent

    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    ret = agent._update_group_membership(
        worker_info, my_devices, reverse_device_map, is_join
    )
    return ret
