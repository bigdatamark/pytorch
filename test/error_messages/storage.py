# [MODULE DOC] Script that checks the error messages raised by invalid storage constructor
# arguments and storage methods (wrong argument types, bad offsets and sizes, bad index or value
# types, invalid resize and fill arguments). Every check runs at import time; a failure raises
# an assertion error and the printed messages are meant to be compared with expected output.
import torch


def check_error(desc, fn, *required_substrings):
    # [FUNCTION DOC] HELPER: Call the given callable and require that it raises an exception. On
    # failure the first argument of the exception is taken as the message, a separator, the
    # description and the message are printed, and each required substring must occur in the
    # message or an assertion error is raised. If the callable does not raise, an assertion
    # error naming the description is raised. Returns None on success.
    try:
        fn()
    except Exception as e:
        error_message = e.args[0]
        print("=" * 80)
        print(desc)
        print("-" * 80)
        print(error_message)
        print()
        for sub in required_substrings:
            if sub not in error_message:
                raise AssertionError(
                    f"Expected '{sub}' in error message: {error_message}"
                ) from None
        return
    raise AssertionError(f"given function ({desc}) didn't raise an error")


check_error("Wrong argument types", lambda: torch.FloatStorage(object()), "object")

check_error(
    "Unknown keyword argument", lambda: torch.FloatStorage(content=1234.0), "keyword"
)

check_error(
    "Invalid types inside a sequence",
    lambda: torch.FloatStorage(["a", "b"]),
    "list",
    "str",
)

check_error("Invalid size type", lambda: torch.FloatStorage(1.5), "float")

check_error(
    "Invalid offset", lambda: torch.FloatStorage(torch.FloatStorage(2), 4), "2", "4"
)

check_error(
    "Negative offset", lambda: torch.FloatStorage(torch.FloatStorage(2), -1), "2", "-1"
)

check_error(
    "Invalid size",
    lambda: torch.FloatStorage(torch.FloatStorage(3), 1, 5),
    "2",
    "1",
    "5",
)

check_error(
    "Negative size",
    lambda: torch.FloatStorage(torch.FloatStorage(3), 1, -5),
    "2",
    "1",
    "-5",
)

check_error("Invalid index type", lambda: torch.FloatStorage(10)["first item"], "str")


def assign():
    # [FUNCTION DOC] HELPER: Try to assign a string to a slice of a float storage of size ten.
    # The storage assignment is expected to raise a type error, which the caller verifies.
    torch.FloatStorage(10)[1:-1] = "1"


check_error("Invalid value type", assign, "str")

check_error(
    "resize_ with invalid type", lambda: torch.FloatStorage(10).resize_(1.5), "float"
)

check_error(
    "fill_ with invalid type", lambda: torch.IntStorage(10).fill_("asdf"), "str"
)

# TODO: frombuffer
