#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC] Defines SubprocessHandler, a wrapper that launches a worker as a child process
# in its own session (on unix) and can later terminate its whole process group. Defines
# IS_WINDOWS from the platform at import time.
import os
import signal
import sys
from subprocess import Popen
from typing import Any

from torch.numa.binding import _maybe_wrap_command_args_with_numa_binding, NumaOptions


__all__ = ["SubprocessHandler"]

IS_WINDOWS = sys.platform == "win32"


def _get_default_signal() -> signal.Signals:
    # [FUNCTION DOC] HELPER: Returns the default termination signal: the Ctrl-C event on Windows, SIGTERM elsewhere.
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined]
    else:
        return signal.SIGTERM


class SubprocessHandler:
    # [CLASS DOC] Owns one child process (attribute proc, a Popen), its optional opened
    # stdout/stderr redirect files, and local_rank_id. The process is started in the constructor
    # and stopped via close(), which also closes the redirect files.
    """
    Convenience wrapper around python's ``subprocess.Popen``. Keeps track of
    meta-objects associated to the process (e.g. stdout and stderr redirect fds).
    """

    def __init__(
        self,
        entrypoint: str,
        args: tuple,
        env: dict[str, str],
        stdout: str | None,
        stderr: str | None,
        local_rank_id: int,
        numa_options: NumaOptions | None,
    ):
        # [FUNCTION DOC] Opens (for writing, truncating) the stdout/stderr files if paths are given,
        # merges the parent environment with env (env wins), stringifies the arguments, optionally
        # wraps the command with NUMA binding for the local rank's device, then launches the process.
        # Side effects: opens files and spawns a child process. Raises OS errors from open/spawn.
        self._stdout = open(stdout, "w") if stdout else None  # noqa: SIM115
        self._stderr = open(stderr, "w") if stderr else None  # noqa: SIM115
        # inherit parent environment vars
        env_vars = os.environ.copy()
        env_vars.update(env)

        args_str = (entrypoint, *[str(e) for e in args])
        args_str = _maybe_wrap_command_args_with_numa_binding(
            args_str,
            device_index=local_rank_id,
            numa_options=numa_options,
        )

        self.local_rank_id = local_rank_id

        self.proc: Popen = self._popen(args_str, env_vars)

    def _popen(self, args: tuple, env: dict[str, str]) -> Popen:
        # [FUNCTION DOC] HELPER: Starts the process with the given argument tuple and environment,
        # wired to the stored stdout/stderr files. On non-Windows it starts a new session so the
        # whole process group can be signalled later. Returns the Popen object.
        kwargs: dict[str, Any] = {}
        if not IS_WINDOWS:
            kwargs["start_new_session"] = True

        return Popen(
            # pyre-fixme[6]: Expected `Union[typing.Sequence[Union[_PathLike[bytes],
            #  _PathLike[str], bytes, str]], bytes, str]` for 1st param but got
            #  `Tuple[str, *Tuple[Any, ...]]`.
            args=args,
            env=env,
            stdout=self._stdout,
            stderr=self._stderr,
            **kwargs,
        )

    def close(self, death_sig: signal.Signals | None = None) -> None:
        # [FUNCTION DOC] Sends death_sig (default: platform default termination signal) to the
        # child: to the process itself on Windows, to its whole process group elsewhere. Then
        # closes the stdout/stderr files if open. Does not wait for the process to exit.
        # Raises OS errors if signalling fails (e.g. process group already gone).
        if not death_sig:
            death_sig = _get_default_signal()
        if IS_WINDOWS:
            self.proc.send_signal(death_sig)
        else:
            os.killpg(self.proc.pid, death_sig)
        if self._stdout:
            self._stdout.close()
        if self._stderr:
            self._stderr.close()
