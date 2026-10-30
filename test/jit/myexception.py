# [MODULE DOC] Test helper defining a custom exception type in a separate module so tests can
# verify that the fully qualified exception class name is captured correctly.
r"""
Define exceptions used in test_exception.py. We define them in a
separate file on purpose to make sure the fully qualified exception class name
is captured correctly in suce cases.
"""


class MyKeyError(KeyError):
    # [CLASS DOC] HELPER: Custom exception deriving from the built-in missing-key error; adds no
    # behavior of its own.
    pass
