# [MODULE DOC]
# Entry point for the autotune subprocess launched by the inductor autotuning machinery.
# Parses the parent pid and the read/write pipe file descriptors from the command line,
# then runs the TuningProcess main loop over those pipes.
# Defines a logger and the main function; running as a script calls main.
# No side effects at import time other than creating the logger.
import argparse
import logging
import os

from torch._inductor.autotune_process import TuningProcess
from torch._inductor.compile_worker.utils import _async_compile_initializer


log = logging.getLogger(__name__)


def main() -> None:
    # [FUNCTION DOC]
    # Subprocess entry point. Parses the command-line options for the parent pid and the
    # read and write pipe file descriptors, and opens them as binary file objects.
    # Installs the parent-death watcher via the async compile initializer, then runs
    # TuningProcess.process_main on the pipes. Any exception is logged, not re-raised,
    # and both pipes are always closed on exit. Returns nothing.
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int)
    parser.add_argument("--read-fd", type=int)
    parser.add_argument("--write-fd", type=int)
    args = parser.parse_args()
    read_pipe = os.fdopen(args.read_fd, "rb")
    write_pipe = os.fdopen(args.write_fd, "wb")

    try:
        # Ensures the subprocess exits if the parent crashes:
        _async_compile_initializer(args.parent)
        TuningProcess.process_main(read_pipe, write_pipe)
    except Exception:
        log.exception("Uncaught exception in autotune subprocess")
    finally:
        read_pipe.close()
        write_pipe.close()


if __name__ == "__main__":
    main()
