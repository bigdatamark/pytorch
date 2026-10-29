#!/usr/bin/env python3
# [MODULE DOC] Entry point for running the model_dump package with the python -m flag.
# Calls main from the package with the command line arguments and exits the process with the
# returned status. Import-time side effect: runs immediately and terminates the interpreter.
import sys
from . import main

sys.exit(main(sys.argv))
