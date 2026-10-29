# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC]
# Provides a context manager that profiles the current process with the external magic-trace
# tool. May download the tool at use time; no import-time side effects beyond imports.
import os
import signal
import subprocess
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def magic_trace(
    output: str = "trace.fxt", magic_trace_cache: str = "/tmp/magic-trace"
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager: downloads the magic-trace binary to magic_trace_cache
    # if missing (wget, chmod; prints progress), attaches it to the current process writing
    # a trace to output, and waits until it reports "Attached" before running the body. On
    # exit it interrupts the tracer, prints its remaining stderr, and raises ValueError if
    # it exited non-zero.
    # KNOWN ISSUE: the wait loop never ends if the tracer exits without printing "Attached".
    pid = os.getpid()
    if not os.path.exists(magic_trace_cache):
        print(f"Downloading magic_trace to: {magic_trace_cache}")
        subprocess.run(
            [
                "wget",
                "-O",
                magic_trace_cache,
                "-q",
                "https://github.com/janestreet/magic-trace/releases/download/v1.0.2/magic-trace",
            ]
        )
        subprocess.run(["chmod", "+x", magic_trace_cache])
    args = [magic_trace_cache, "attach", "-pid", str(pid), "-o", output]
    p = subprocess.Popen(args, stderr=subprocess.PIPE, encoding="utf-8")
    if p.stderr is None:
        raise AssertionError("Expected stderr to be non-None")
    while True:
        x = p.stderr.readline()
        print(x)
        if "Attached" in x:
            break
    try:
        yield
    finally:
        p.send_signal(signal.SIGINT)
        r = p.wait()
        if p.stderr is not None:
            print(p.stderr.read())
            p.stderr.close()
        if r != 0:
            raise ValueError(f"magic_trace exited abnormally: {r}")
