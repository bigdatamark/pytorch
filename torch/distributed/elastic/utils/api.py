#!/usr/bin/env python3

# [MODULE DOC] Small helpers for elastic launch: reading required environment variables, obtaining a bound localhost listening socket, and substituting a local-rank macro in command line arguments. No import-time side effects.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from string import Template
from typing import Any


def get_env_variable_or_raise(env_name: str) -> str:
    # [FUNCTION DOC] Return the value of the environment variable env_name; raises ValueError if it is not set.
    r"""
    Tries to retrieve environment variable. Raises ``ValueError``
    if no environment variable found.

    Args:
        env_name (str): Name of the env variable
    """
    value = os.environ.get(env_name, None)
    if value is None:
        msg = f"Environment variable {env_name} expected, but not set"
        raise ValueError(msg)
    return value


def get_socket_with_port() -> socket.socket:
    # [FUNCTION DOC] Create and return a TCP socket bound to localhost on an OS-chosen free port and listening (backlog zero). Tries each address family returned for localhost, closing sockets that fail to bind/listen.
    # Raises RuntimeError if no socket could be created. The caller owns (must close) the returned socket.
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError:
            s.close()
    raise RuntimeError("Failed to create a socket")


class macros:
    # [CLASS DOC] Namespace of macro placeholder strings usable in launcher arguments. Attribute local_rank is the placeholder for the local rank. Key method: substitute.
    """
    Defines simple macros for caffe2.distributed.launch cmd args substitution
    """

    local_rank = "${local_rank}"

    @staticmethod
    def substitute(args: list[Any], local_rank: str) -> list[str]:
        # [FUNCTION DOC] Static: return a new list where each string argument has the local_rank placeholder substituted with the given local_rank (unknown placeholders left untouched); non-string arguments are passed through unchanged.
        # # KNOWN ISSUE: return annotation says list of str but non-str items are preserved.
        args_sub = []
        for arg in args:
            if isinstance(arg, str):
                sub = Template(arg).safe_substitute(local_rank=local_rank)
                args_sub.append(sub)
            else:
                args_sub.append(arg)
        return args_sub
