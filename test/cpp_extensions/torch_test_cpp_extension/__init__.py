# [PACKAGE DOC] Test-only out-of-tree device backend extension package (see RFC 122468).
# Exports the private entry point _autoload, which the backend autoload mechanism invokes.
"""
This is a device backend extension used for testing.
See this RFC: https://github.com/pytorch/pytorch/issues/122468
"""

import os


def _autoload():
    # [FUNCTION DOC] HELPER: Autoload entry point; sets the process environment variable
    # IS_CUSTOM_DEVICE_BACKEND_IMPORTED to 1 so tests can detect the backend was loaded.
    # Set the environment variable to true in this entrypoint
    os.environ["IS_CUSTOM_DEVICE_BACKEND_IMPORTED"] = "1"
